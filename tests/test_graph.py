from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pytest
import torch
from kmp.data import load_graph, split_graph, adjacency, Neighborhood
from kmp.position import laplacian_pe
from kmp.models import Teacher, Student
from kmp.train import teacher_inference


def test_split_is_disjoint_seeded_and_inductive_graph_has_no_heldout_nodes():
    graph = load_graph("synthetic", Path("unused"))
    split = split_graph(graph, 17, inductive_fraction=.2)
    again = split_graph(graph, 17, inductive_fraction=.2)
    torch.testing.assert_close(split.observed_ids, again.observed_ids)
    original_sets = [set(split.observed_ids[x].tolist()) for x in [split.train, split.val, split.test]]
    original_sets.append(set(split.heldout.tolist()))
    for i, a in enumerate(original_sets):
        for b in original_sets[i+1:]:
            assert not a & b
    assert len(set.union(*original_sets)) == graph.num_nodes
    assert split.observed.num_nodes == 192
    original_edges = split.observed_ids[split.observed.edge_index]
    assert not set(original_edges.flatten().tolist()) & original_sets[-1]
    for label in range(3):
        assert int((split.observed.y[split.train] == label).sum()) == 20
        assert int((split.observed.y[split.val] == label).sum()) == 30


def test_pe_satisfies_normalized_laplacian_equation_and_skips_all_zero_modes():
    # Two disconnected paths and an isolated vertex.
    edge = torch.tensor([[0,1,1,2,3,4,4,5], [1,0,2,1,4,3,5,4]])
    pe, values, residual = laplacian_pe(edge, 7, 3)
    a = adjacency(edge, 7)
    degree = np.asarray(a.sum(1)).flatten()
    inv = np.zeros(7)
    inv[degree>0] = degree[degree>0]**-.5
    lap = sp.eye(7)-sp.diags(inv) @ a @ sp.diags(inv)
    assert min(values) > 1e-7 and residual < 1e-6
    np.testing.assert_allclose(lap @ pe.numpy(), pe.numpy()*values, atol=1e-6)
    np.testing.assert_allclose(pe.T @ pe, np.eye(3), atol=1e-6)


def test_sparse_pe():
    ids = torch.arange(599)
    edge = torch.cat([torch.stack([ids, ids+1]), torch.stack([ids+1, ids])], 1)
    pe, _, residual = laplacian_pe(edge, 600, 3)
    assert pe.shape == (600, 3) and residual < 1e-5


def test_sage_layerwise_and_all_neighbor_sampling_equal_full_graph():
    torch.manual_seed(0)
    graph = load_graph("synthetic", Path("unused"))
    model = Teacher(12, 8, 3, "sage", layers=3).eval()
    sampler = Neighborhood(graph.edge_index, graph.num_nodes, 0)
    full, hidden = model(graph.x, graph.edge_index)
    batched, batched_h = teacher_inference(model, graph, sampler, 17, "cpu")
    torch.testing.assert_close(full, batched, atol=1e-6, rtol=1e-5)
    for a, b in zip(hidden, batched_h):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    target = torch.tensor([15, 2, 17, 11])
    nodes, blocks = sampler.sample(target.numpy(), [-1, -1, -1])
    sampled = model.sampled(graph.x[nodes], blocks)
    torch.testing.assert_close(full[target], sampled, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("fusion", ["concat", "mul"])
def test_student_pe_projection_and_graph_free_forward(fusion):
    student = Student(7, 8, 3, pe_dim=2, pe_fusion=fusion)
    logits, hidden = student(torch.randn(4, 7), torch.randn(4, 2))
    assert logits.shape == (4, 3) and hidden[0].shape == (4, 8)
    with pytest.raises(ValueError):
        student(torch.randn(4, 7))


def test_batch_norm_singleton_training_is_finite():
    from kmp.models import SafeBatchNorm
    bn = SafeBatchNorm(4).train()
    value = bn(torch.randn(1, 4))
    assert torch.isfinite(value).all()


def test_amazon_largest_component_preprocessing(monkeypatch):
    from torch_geometric.data import Data
    import kmp.data as module
    graph = Data(x=torch.ones(5, 3), y=torch.tensor([0, 0, 1, 1, 1]),
                 edge_index=torch.tensor([[0,1,2,3,3,4], [1,0,3,2,4,3]]))
    monkeypatch.setattr(module, "Amazon", lambda **kwargs: [graph])
    processed = module.load_graph("Photo", Path("unused"))
    assert processed.num_nodes == 3 and processed.edge_index.max() == 2
