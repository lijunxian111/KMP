import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from .train import run


def parser():
    p = argparse.ArgumentParser(description="Paper-based KMP reconstruction")
    p.add_argument("--config", type=Path)
    p.add_argument("--dataset", choices=["Cora", "Citeseer", "Pubmed", "Photo", "Computers", "ogbn-arxiv", "ogbn-products", "synthetic"], default="Cora")
    p.add_argument("--teacher", choices=["sage", "gcn", "gat"], default="sage")
    p.add_argument("--method", choices=["mlp", "glnn", "kmp"], default="kmp")
    p.add_argument("--kernel", choices=["gaussian", "sigmoid", "polynomial", "randomized", "reverse"], default="gaussian")
    p.add_argument("--kernel-reduction", choices=["sum", "mean"], default="sum")
    p.add_argument("--pe-dim", type=int, default=0)
    p.add_argument("--pe-fusion", choices=["concat", "mul"], default="concat")
    p.add_argument("--inductive-fraction", type=float, default=0.)
    p.add_argument("--inductive-pe", choices=["error", "zero"], default="error")
    p.add_argument("--split-protocol", choices=["random", "official"], default="random")
    p.add_argument("--train-per-class", type=int, default=20)
    p.add_argument("--val-per-class", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--batch-norm", action="store_true")
    p.add_argument("--student-dropout", type=float, default=0.)
    p.add_argument("--student-weight-decay", type=float, default=0.)
    p.add_argument("--fanout", type=int, default=5)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--attention-dropout", type=float, default=.6)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--teacher-lr", type=float, help="Optional separate Stage I learning rate")
    p.add_argument("--theta", type=float, default=.5)
    p.add_argument("--gamma", type=float, default=.1)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--heat-time", type=float, default=1.)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--offset", type=float, default=1.)
    p.add_argument("--random-terms", type=int, default=8)
    p.add_argument("--random-dim", type=int, default=64)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--output", default="runs/experiment")
    return p


def config_from_args(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        overrides = yaml.safe_load(preliminary.config.read_text()) or {}
        unknown = set(overrides) - {a.dest for a in p._actions}
        if unknown:
            p.error(f"Unknown configuration keys: {sorted(unknown)}")
        p.set_defaults(**overrides)
    config = vars(p.parse_args(argv))
    config.pop("config")
    if min(config["epochs"], config["patience"], config["batch_size"], config["threads"]) < 1:
        p.error("Epochs, patience, batch size and threads must be positive")
    if config["batch_norm"] and config["batch_size"] < 2:
        p.error("Batch normalization requires batch size >= 2")
    if not 0 <= config["theta"] <= 1 or config["gamma"] < 0:
        p.error("theta must be in [0,1] and gamma >= 0")
    if config["layers"] < 2 or config["pe_dim"] < 0 or config["temperature"] <= 0 or config["lr"] <= 0:
        p.error("Invalid layer count, PE dimension, temperature or learning rate")
    if config["teacher_lr"] is not None and config["teacher_lr"] <= 0:
        p.error("Teacher learning rate must be positive")
    return config


def main(argv=None):
    config = config_from_args(argv)
    seeds = config.pop("seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    output = Path(config["output"])
    results = []
    for seed in seeds:
        job = dict(config, seed=seed, output=str(output / f"seed_{seed}"))
        results.append(run(job))
    key = "inductive_test_accuracy" if config["inductive_fraction"] else "observed_test_accuracy"
    scores = np.array([r[key] for r in results]) * 100
    summary = dict(metric=key, runs=len(scores), seeds=seeds, mean_percent=float(scores.mean()),
                   std_percent=float(scores.std(ddof=1)) if len(scores)>1 else None,
                   std_convention="sample (ddof=1)", config=config,
                   status="Measured reconstruction results; not original author results")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
