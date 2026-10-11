import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .data import load_graph, split_graph, Neighborhood
from .models import Teacher, Student
from .kernels import PairwiseKernel, ReverseReconstruction, hidden_loss, soft_loss
from .position import laplacian_pe


PRESETS = {
    "gcn": dict(hidden_dim=64, dropout=.8, weight_decay=.001),
    "sage": dict(hidden_dim=128, dropout=0., weight_decay=.0005),
    "gat": dict(hidden_dim=128, dropout=.6, weight_decay=.01),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def accuracy(logits, y, ids):
    if not len(ids):
        raise ValueError("Cannot evaluate an empty split")
    return float((logits[ids].argmax(-1) == y[ids]).float().mean())


def checked_step(loss, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss: change polynomial parameters or loss scale")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Non-finite gradients")
    optimizer.step()


@torch.no_grad()
def teacher_inference(model, graph, sampler, batch_size, device):
    model.eval()
    if model.kind != "sage":
        logits, hidden = model(graph.x.to(device), graph.edge_index.to(device))
        return logits.cpu(), [h.cpu() for h in hidden]
    x, hidden = graph.x, []
    for layer, conv in enumerate(model.convs):
        output = []
        for start in range(0, graph.num_nodes, batch_size):
            targets = np.arange(start, min(start + batch_size, graph.num_nodes))
            sources, (edges, count) = sampler.block(targets)
            block_x = x[torch.from_numpy(sources)].to(device)
            result = conv((block_x, block_x[:count]), edges.to(device))
            if layer < len(model.convs)-1:
                result = model.activate(result, layer)
            output.append(result.cpu())
        x = torch.cat(output)
        if layer < len(model.convs)-1:
            hidden.append(x)
    return x, hidden


@torch.no_grad()
def student_inference(model, x, pe, batch_size, device):
    model.eval()
    return torch.cat([model(x[i:i+batch_size].to(device),
                            None if pe is None else pe[i:i+batch_size].to(device))[0].cpu()
                      for i in range(0, len(x), batch_size)])


def train_teacher(model, graph, split, sampler, config, device, run_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("teacher_lr") or config["lr"],
                                 weight_decay=config["weight_decay"])
    best, stale, history, best_state = -1., 0, [], None
    for epoch in range(config["epochs"]):
        model.train()
        if model.kind == "sage":
            ids = split.train[torch.randperm(len(split.train))]
            losses = []
            for target in ids.split(config["batch_size"]):
                nodes, blocks = sampler.sample(target.numpy(), [config["fanout"]]*len(model.convs))
                blocks = [(edge.to(device), count) for edge, count in blocks]
                logits = model.sampled(graph.x[nodes].to(device), blocks)
                loss = F.cross_entropy(logits, graph.y[target].to(device))
                checked_step(loss, optimizer)
                losses.append(float(loss.detach()))
            loss_value = sum(losses) / len(losses)
        else:
            logits, _ = model(graph.x.to(device), graph.edge_index.to(device))
            loss = F.cross_entropy(logits[split.train.to(device)], graph.y[split.train].to(device))
            checked_step(loss, optimizer)
            loss_value = float(loss.detach())
        logits, _ = teacher_inference(model, graph, sampler, config["batch_size"], device)
        score = accuracy(logits, graph.y, split.val)
        history.append(dict(epoch=epoch+1, loss=loss_value, val_accuracy=score))
        if score > best:
            best, stale = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 0 or (epoch+1) % 25 == 0:
            print(f"teacher epoch={epoch+1} loss={loss_value:.5g} val={score:.4f}", flush=True)
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    torch.save(dict(state_dict=model.cpu().state_dict(), config=config), run_dir / "teacher.pt")
    model.to(device)
    (run_dir / "teacher_history.json").write_text(json.dumps(history, indent=2))
    return best


def train_student(model, graph, split, pe, teacher_logits, teacher_hidden, kernels,
                  reconstruction, config, device, run_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["student_weight_decay"])
    kernel_optimizer = None
    if config["kernel"] == "reverse" and config["method"] == "kmp":
        kernel_optimizer = torch.optim.Adam(list(kernels.parameters()) + list(reconstruction.parameters()),
                                            lr=config["lr"])
    # Hard targets are only training labels. Soft targets use disjoint observed nodes.
    soft_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    soft_mask[split.train] = False
    soft_ids = soft_mask.nonzero().flatten()
    best, stale, history, best_state, best_kernels = -1., 0, [], None, None
    best_reconstruction = None
    for epoch in range(config["epochs"]):
        model.train()
        pool = split.train if config["method"] == "mlp" else soft_ids
        pool = pool[torch.randperm(len(pool))]
        losses = []
        for ids in pool.split(config["batch_size"]):
            bx = graph.x[ids].to(device)
            bp = None if pe is None else pe[ids].to(device)
            logits, hidden = model(bx, bp)
            if config["method"] == "mlp":
                loss = F.cross_entropy(logits, graph.y[ids].to(device))
                kernel_value = logits.new_zeros(())
            else:
                # Independently sample labeled nodes; batch-sized kernel matrices only.
                hard_ids = split.train[torch.randperm(len(split.train))[:config["batch_size"]]]
                hard_logits, _ = model(graph.x[hard_ids].to(device),
                                      None if pe is None else pe[hard_ids].to(device))
                hard = F.cross_entropy(hard_logits, graph.y[hard_ids].to(device))
                soft = soft_loss(logits, teacher_logits[ids].to(device), config["temperature"])
                loss = config["theta"] * hard + (1-config["theta"]) * soft
                kernel_value = logits.new_zeros(())
                if config["method"] == "kmp":
                    teacher_h = [h[ids].to(device) for h in teacher_hidden]
                    if kernel_optimizer is not None:
                        # Step 1 updates W_k/decoder on detached states, Eq. (15).
                        kernel_optimizer.zero_grad(set_to_none=True)
                        re = reconstruction(teacher_h[-1], bx, kernels[-1], config["kernel_reduction"])
                        re += reconstruction(hidden[-1].detach(), bx, kernels[-1], config["kernel_reduction"])
                        checked_step(re, kernel_optimizer)
                        kernels.requires_grad_(False)
                    # Step 2 updates student; teacher and reverse kernel are fixed.
                    kernel_value = hidden_loss(hidden, teacher_h, kernels, config["kernel_reduction"])
                    loss = loss + config["gamma"] * kernel_value
            checked_step(loss, optimizer)
            if kernel_optimizer is not None:
                kernels.requires_grad_(True)
            losses.append((float(loss.detach()), float(kernel_value.detach())))
        logits = student_inference(model, graph.x, pe, config["batch_size"], device)
        score = accuracy(logits, graph.y, split.val)
        history.append(dict(epoch=epoch+1, loss=float(np.mean([x[0] for x in losses])),
                            hidden_loss=float(np.mean([x[1] for x in losses])), val_accuracy=score))
        if score > best:
            best, stale = score, 0
            best_state = copy.deepcopy(model.state_dict())
            best_kernels = copy.deepcopy(kernels.state_dict())
            if reconstruction is not None:
                best_reconstruction = copy.deepcopy(reconstruction.state_dict())
        else:
            stale += 1
        if epoch == 0 or (epoch+1) % 25 == 0:
            print(f"student epoch={epoch+1} loss={history[-1]['loss']:.5g} val={score:.4f}", flush=True)
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    kernels.load_state_dict(best_kernels)
    if reconstruction is not None:
        reconstruction.load_state_dict(best_reconstruction)
    torch.save(dict(state_dict=model.cpu().state_dict(), kernels=kernels.cpu().state_dict(),
                    reconstruction=best_reconstruction, config=config), run_dir / "student.pt")
    model.to(device)
    (run_dir / "student_history.json").write_text(json.dumps(history, indent=2))
    return best


def run(config):
    config = dict(config)
    seed_everything(config["seed"])
    torch.set_num_threads(config["threads"])
    preset = PRESETS[config["teacher"]]
    for key, value in preset.items():
        if config.get(key) is None:
            config[key] = value
    device = torch.device(config["device"])
    run_dir = Path(config["output"])
    if (run_dir / "config.json").exists():
        raise FileExistsError(f"Run directory already used: {run_dir}; choose a new --output")
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    graph = load_graph(config["dataset"], Path(config["data_root"]))
    split = split_graph(graph, config["seed"], config["train_per_class"], config["val_per_class"],
                        config["inductive_fraction"], config["split_protocol"])
    observed = split.observed
    config.update(input_dim=graph.x.shape[1], classes=int(graph.y.max())+1)
    if observed.num_nodes > 100000 and config["teacher"] != "sage":
        raise ValueError("Large graphs require sampled SAGE in this implementation")
    if config["pe_dim"] and len(split.heldout) and config["inductive_pe"] != "zero":
        raise ValueError("Inductive PE is unspecified in paper. Explicitly choose --inductive-pe zero or --pe-dim 0")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(dict(train=split.train, val=split.val, test=split.test,
                    observed_ids=split.observed_ids, heldout=split.heldout), run_dir / "split.pt")
    pe, pe_meta = None, None
    if config["pe_dim"]:
        print("Computing PE on observed graph", flush=True)
        pe, values, residual = laplacian_pe(observed.edge_index, observed.num_nodes,
                                          config["pe_dim"], config["seed"])
        pe_meta = dict(eigenvalues=values, max_residual=residual)
        torch.save(pe, run_dir / "pe.pt")
    sampler = Neighborhood(observed.edge_index, observed.num_nodes, config["seed"])
    teacher = Teacher(config["input_dim"], config["hidden_dim"], config["classes"],
                      config["teacher"], config["layers"], config["dropout"],
                      config["heads"], config["attention_dropout"], config["batch_norm"]).to(device)
    teacher_val = None
    if config["method"] != "mlp":
        teacher_val = train_teacher(teacher, observed, split, sampler, config, device, run_dir)
        teacher_logits, teacher_hidden = teacher_inference(teacher, observed, sampler,
                                                           config["batch_size"], device)
        torch.save(dict(logits=teacher_logits, hidden=teacher_hidden), run_dir / "teacher_targets.pt")
    else:
        teacher_logits, teacher_hidden = None, None
    model = Student(config["input_dim"], config["hidden_dim"], config["classes"], config["layers"],
                    config["student_dropout"], config["pe_dim"], config["pe_fusion"],
                    config["batch_norm"]).to(device)
    kernels = nn.ModuleList([PairwiseKernel(config["kernel"], config["hidden_dim"],
                                           config["heat_time"], config["degree"], config["offset"],
                                           config["random_terms"], config["random_dim"])
                             for _ in range(config["layers"]-1)]).to(device)
    if config["kernel"] == "reverse" and config["layers"] != 2:
        raise ValueError("Reverse-kernel reconstruction currently requires two-layer networks")
    reconstruction = (ReverseReconstruction(config["hidden_dim"], config["input_dim"]).to(device)
                      if config["kernel"] == "reverse" else None)
    student_val = train_student(model, observed, split, pe, teacher_logits, teacher_hidden,
                                kernels, reconstruction, config, device, run_dir)
    logits = student_inference(model, observed.x, pe, config["batch_size"], device)
    test_score = accuracy(logits, observed.y, split.test)
    heldout_score = None
    if len(split.heldout):
        zero_pe = torch.zeros(len(split.heldout), config["pe_dim"]) if pe is not None else None
        heldout_logits = student_inference(model, graph.x[split.heldout], zero_pe,
                                           config["batch_size"], device)
        heldout_score = accuracy(heldout_logits, graph.y[split.heldout], torch.arange(len(split.heldout)))
    result = dict(seed=config["seed"], dataset=config["dataset"], method=config["method"],
                  teacher=config["teacher"], kernel=config["kernel"], pe_dim=config["pe_dim"],
                  teacher_val_accuracy=teacher_val, student_val_accuracy=student_val,
                  observed_test_accuracy=test_score, inductive_test_accuracy=heldout_score,
                  teacher_observed_test_accuracy=(accuracy(teacher_logits, observed.y, split.test)
                                                 if teacher_logits is not None else None),
                  elapsed_seconds=time.perf_counter()-start, pe=pe_meta,
                  counts=dict(observed=observed.num_nodes, train=len(split.train), val=len(split.val),
                              observed_test=len(split.test), heldout=len(split.heldout)),
                  versions=dict(torch=torch.__version__, numpy=np.__version__))
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result
