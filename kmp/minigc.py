"""Exploratory MiniGC extension; features/readout are assumptions, not specified by paper."""
import argparse
import copy
import json
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from .models import Teacher, Student
from .kernels import PairwiseKernel, hidden_loss, soft_loss
from .train import seed_everything, checked_step


def make_graphs(count, seed):
    """Eight MiniGC families using NetworkX; independent seeded generation."""
    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(count):
        label, n = i % 8, int(rng.integers(10, 20))
        if label == 0:
            graph = nx.cycle_graph(n)
        elif label == 1:
            graph = nx.star_graph(n-1)
        elif label == 2:
            graph = nx.wheel_graph(n)
        elif label == 3:
            tail = int(rng.integers(2, n//2))
            graph = nx.lollipop_graph(n-tail, tail)
        elif label == 4:
            graph = nx.hypercube_graph(int(np.log2(n)))
        elif label == 5:
            rows = int(rng.integers(2, n//2))
            graph = nx.grid_2d_graph(rows, n//rows)
        elif label == 6:
            graph = nx.complete_graph(n)
        else:
            graph = nx.circular_ladder_graph(n//2)
        graph = nx.convert_node_labels_to_integers(graph)
        edges = list(graph.to_directed().edges()) + [(j, j) for j in graph.nodes()]
        # Degree including a self-loop is an explicit preprocessing choice.
        features = torch.tensor([[graph.degree(j)+1.] for j in graph.nodes()])
        graphs.append(Data(x=features, edge_index=torch.tensor(edges).T.contiguous(),
                           y=torch.tensor([label]), num_nodes=len(graph)))
    return graphs


class GraphClassifier(nn.Module):
    def __init__(self, teacher=False, hidden_dim=64):
        super().__init__()
        self.backbone = Teacher(1, hidden_dim, hidden_dim, "sage") if teacher else Student(1, hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 8)
        self.is_teacher = teacher

    def forward(self, batch):
        embeddings, hidden = (self.backbone(batch.x, batch.edge_index) if self.is_teacher
                              else self.backbone(batch.x))
        logits = self.readout(global_mean_pool(embeddings, batch.batch))
        return logits, hidden


def graph_hidden_loss(student_h, teacher_h, batch, kernel):
    # No artificial cross-graph node pairs in the mapping matrix.
    losses = [hidden_loss([student_h[0][batch.batch == i]], [teacher_h[0][batch.batch == i]],
                          [kernel], "mean") for i in range(batch.num_graphs)]
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, graphs):
    model.eval()
    batch = Batch.from_data_list(graphs)
    logits, _ = model(batch)
    return float((logits.argmax(1) == batch.y).float().mean())


def fit(model, training, validation, args, teacher=None, kernel=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, state, stale = -1., None, 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(training))
        for ids in order.split(args.batch_size):
            batch = Batch.from_data_list([training[i] for i in ids.tolist()])
            logits, hidden = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            if teacher is not None:
                with torch.no_grad():
                    targets, teacher_h = teacher(batch)
                loss = args.theta * loss + (1-args.theta) * soft_loss(logits, targets, args.temperature)
                if args.method == "kmp":
                    loss += args.gamma * graph_hidden_loss(hidden, teacher_h, batch, kernel)
            checked_step(loss, optimizer)
        score = evaluate(model, validation)
        if score > best:
            best, state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(state)
    return best


def main():
    p = argparse.ArgumentParser(description="Exploratory MiniGC graph classification extension")
    p.add_argument("--method", choices=["mlp", "glnn", "kmp"], default="kmp")
    p.add_argument("--kernel", choices=["gaussian", "sigmoid", "polynomial", "randomized"], default="gaussian")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--theta", type=float, default=.5)
    p.add_argument("--gamma", type=float, default=.1)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--output", type=Path, default=Path("runs/minigc"))
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim) < 1 or args.lr <= 0:
        p.error("Training sizes and learning rate must be positive")
    if not 0 <= args.theta <= 1 or args.gamma < 0 or args.temperature <= 0:
        p.error("Invalid distillation parameters")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    results = []
    for seed in args.seeds:
        seed_everything(seed)
        training, validation, testing = [make_graphs(n, seed*3+i) for i, n in enumerate([240, 60, 60])]
        teacher = GraphClassifier(True, args.hidden_dim)
        teacher_score = None
        if args.method != "mlp":
            fit(teacher, training, validation, args)
            teacher.eval().requires_grad_(False)
            teacher_score = evaluate(teacher, testing)
        student = GraphClassifier(False, args.hidden_dim)
        kernel = PairwiseKernel(args.kernel, args.hidden_dim)
        fit(student, training, validation, args, teacher if args.method != "mlp" else None, kernel)
        score = evaluate(student, testing)
        torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(), kernel=kernel.state_dict(),
                        hidden_dim=args.hidden_dim, seed=seed), args.output / f"seed_{seed}.pt")
        results.append(dict(seed=seed, teacher_test_accuracy=teacher_score, student_test_accuracy=score))
        print(results[-1], flush=True)
    scores = np.array([r["student_test_accuracy"] for r in results])*100
    config = vars(args).copy()
    config["output"] = str(args.output)
    summary = dict(protocol="Exploratory reconstruction, independent MiniGC-family generation and degree features",
                   config=config, results=results, mean_percent=float(scores.mean()),
                   std_percent=float(scores.std(ddof=1)) if len(scores)>1 else None)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
