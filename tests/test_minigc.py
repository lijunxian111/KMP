import torch
from torch_geometric.data import Batch
from kmp.minigc import make_graphs, GraphClassifier, graph_hidden_loss
from kmp.kernels import PairwiseKernel


def test_minigc_families_seeded_and_graph_distillation_gradient():
    a, b = make_graphs(24, 7), make_graphs(24, 7)
    assert {int(g.y) for g in a} == set(range(8))
    for x, y in zip(a, b):
        torch.testing.assert_close(x.x, y.x)
        torch.testing.assert_close(x.edge_index, y.edge_index)
    batch = Batch.from_data_list(a[:8])
    teacher = GraphClassifier(True, 8).eval().requires_grad_(False)
    student = GraphClassifier(False, 8)
    logits, hidden = student(batch)
    with torch.no_grad():
        _, teacher_h = teacher(batch)
    loss = graph_hidden_loss(hidden, teacher_h, batch, PairwiseKernel("gaussian", 8))
    loss.backward()
    assert logits.shape == (8, 8)
    assert student.backbone.linears[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())


def test_student_graph_prediction_does_not_use_edges():
    batch = Batch.from_data_list(make_graphs(8, 3))
    student = GraphClassifier(False, 8).eval()
    first, _ = student(batch)
    batch.edge_index = torch.empty((2, 0), dtype=torch.long)
    second, _ = student(batch)
    torch.testing.assert_close(first, second)
