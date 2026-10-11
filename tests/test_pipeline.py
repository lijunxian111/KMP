import json
import pytest
import torch
from kmp.cli import config_from_args
from kmp.train import run, student_inference
from kmp.data import load_graph, split_graph
from kmp.infer import load_student


@pytest.mark.parametrize("teacher,kernel,pe,inductive,method", [
    ("gcn", "gaussian", 4, 0., "kmp"),
    ("gat", "sigmoid", 0, 0., "kmp"),
    ("sage", "polynomial", 0, .2, "kmp"),
    ("sage", "randomized", 0, 0., "kmp"),
    ("sage", "reverse", 0, 0., "kmp"),
    ("sage", "gaussian", 4, .2, "kmp"),
    ("sage", "gaussian", 0, 0., "glnn"),
    ("sage", "gaussian", 0, 0., "mlp"),
])
def test_three_stage_pipeline_and_checkpoint_reload(tmp_path, teacher, kernel, pe, inductive, method):
    c = config_from_args(["--dataset", "synthetic", "--teacher", teacher, "--kernel", kernel,
                          "--pe-dim", str(pe), "--inductive-fraction", str(inductive),
                          "--inductive-pe", "zero", "--method", method,
                          "--hidden-dim", "8", "--epochs", "2", "--patience", "2",
                          "--kernel-reduction", "mean", "--output", str(tmp_path)])
    c.pop("seeds")
    c["seed"] = 0
    result = run(c)
    assert 0 <= result["observed_test_accuracy"] <= 1
    assert (tmp_path / "student.pt").exists()
    model, _ = load_student(tmp_path / "student.pt")
    graph = load_graph("synthetic", tmp_path)
    split = split_graph(graph, 0, inductive_fraction=inductive)
    position = torch.load(tmp_path / "pe.pt", weights_only=True) if pe else None
    logits = student_inference(model, split.observed.x, position, 31, "cpu")
    accuracy = float((logits[split.test].argmax(1) == split.observed.y[split.test]).float().mean())
    assert accuracy == result["observed_test_accuracy"]
    for h in json.loads((tmp_path / "student_history.json").read_text()):
        assert h["loss"] == h["loss"]
    if method != "mlp":
        assert (tmp_path / "teacher_targets.pt").exists()
