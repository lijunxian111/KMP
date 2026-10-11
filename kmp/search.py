"""Validation-only hyperparameter selection for Appendix Table 9."""
import argparse
import itertools
import json
from pathlib import Path
import yaml
from .cli import config_from_args
from .train import run


def candidates(grid):
    allowed = {"gamma", "temperature", "theta", "lr", "kernel", "pe_dim", "heat_time"}
    if not grid or set(grid)-allowed or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("Grid must contain nonempty lists of supported hyperparameters")
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def main():
    p = argparse.ArgumentParser(description="Select KMP hyperparameters using validation accuracy")
    p.add_argument("--grid", default="configs/grid.yaml", type=Path)
    p.add_argument("--base-config", default="configs/cora_sage.yaml", type=Path)
    p.add_argument("--output", default="runs/search", type=Path)
    p.add_argument("--tuning-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--execute", action="store_true", help="Without this flag, only print the experiment count")
    args = p.parse_args()
    grid = candidates(yaml.safe_load(args.grid.read_text()))
    if not args.execute:
        print(json.dumps(dict(candidates=len(grid), runs=len(grid)*len(args.tuning_seeds),
                              selection_metric="mean student validation accuracy"), indent=2))
        return
    config = config_from_args(["--config", str(args.base_config)])
    config.pop("seeds")
    scores = []
    for i, values in enumerate(grid):
        vals = []
        for seed in args.tuning_seeds:
            result = run(dict(config, **values, seed=seed,
                              output=str(args.output / f"candidate_{i}" / f"seed_{seed}")))
            vals.append(result["student_val_accuracy"])
        scores.append(dict(candidate=i, params=values, val_accuracy=sum(vals)/len(vals)))
    best = max(scores, key=lambda s: s["val_accuracy"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection.json").write_text(json.dumps(dict(best=best, scores=scores), indent=2))
    config.update(best["params"], seeds=list(range(10)), output=str(args.output / "selected"))
    (args.output / "selected.yaml").write_text(yaml.safe_dump(config))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
