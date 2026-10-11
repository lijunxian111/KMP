"""Explicit test-time Gaussian feature-noise protocol; paper leaves noise law unspecified."""
import argparse
import json
from pathlib import Path
import torch
from .data import load_graph
from .infer import load_student
from .train import student_inference, accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--fractions", nargs="+", type=float, default=[.1, .2, .3, .4, .5])
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if any(f < 0 for f in args.fractions):
        p.error("Noise fractions must be nonnegative")
    model, config = load_student(args.run / "student.pt")
    graph = load_graph(config["dataset"], Path(config["data_root"]))
    split = torch.load(args.run / "split.pt", weights_only=True)
    pe = torch.load(args.run / "pe.pt", weights_only=True) if config["pe_dim"] else None
    if len(split["heldout"]):
        ids = split["heldout"]
        x, y = graph.x[ids], graph.y[ids]
        if pe is not None:
            pe = torch.zeros(len(ids), config["pe_dim"])
        test = torch.arange(len(ids))
    else:
        x, y = graph.x[split["observed_ids"]], graph.y[split["observed_ids"]]
        test = split["test"]
    generator = torch.Generator().manual_seed(args.noise_seed)
    noise = torch.randn(x.shape, generator=generator)
    results = []
    for fraction in args.fractions:
        logits = student_inference(model, x + fraction*noise, pe, config["batch_size"], "cpu")
        results.append(dict(fraction=fraction, accuracy=accuracy(logits, y, test)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(protocol="test-time X + fraction * N(0,1)",
                                         noise_seed=args.noise_seed, results=results), indent=2))


if __name__ == "__main__":
    main()
