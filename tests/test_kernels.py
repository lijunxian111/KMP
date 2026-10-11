import math
import pytest
import torch
from kmp.kernels import PairwiseKernel, hidden_loss, matrix_distance, soft_loss, ReverseReconstruction


@pytest.mark.parametrize("kind", ["gaussian", "sigmoid", "polynomial", "randomized", "reverse"])
def test_kernel_against_scalar_formula(kind):
    torch.manual_seed(7)
    h = torch.tensor([[.2, -.3], [.5, .4], [-.1, .6]], dtype=torch.float64)
    kernel = PairwiseKernel(kind, 2, heat_time=.7, random_terms=3, random_dim=4).double()
    result = kernel(h)
    expected = torch.empty_like(result)
    for i in range(3):
        for j in range(3):
            a, b = h[i], h[j]
            if kind == "gaussian":
                value = math.exp(-float(((a-b)**2).sum()) / 2.8)
            elif kind == "sigmoid":
                value = 1 / (1 + math.exp(-float(a.dot(b))))
            elif kind == "polynomial":
                value = (float(a.dot(b)) + 1)**2
            elif kind == "randomized":
                value = sum(math.exp(float(xi)) * float(torch.sigmoid(m @ a).dot(torch.sigmoid(m @ b)))
                            for m, xi in zip(kernel.matrices, kernel.xi)) / 3
            else:
                value = float(torch.sigmoid(kernel.projection.weight.detach() @ a).dot(
                              torch.sigmoid(kernel.projection.weight.detach() @ b)))
            expected[i, j] = value
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result, result.T)


def test_hidden_loss_is_layer_average_of_squared_frobenius_distances():
    s = [torch.tensor([[.2, .3], [.4, .5]], requires_grad=True),
         torch.tensor([[.1, .3], [.5, .4]], requires_grad=True)]
    t = [x.detach().clone().mul(1.3).requires_grad_() for x in s]
    kernels = [PairwiseKernel("gaussian", 2) for _ in s]
    expected = sum(((k(a) - k(b))**2).sum() for a, b, k in zip(s, t, kernels)) / 2
    actual = hidden_loss(s, t, kernels)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in s)
    assert all(x.grad is None for x in t)
    assert hidden_loss([s[0]], [s[0].detach()], [kernels[0]]).item() == 0


def test_kl_direction_and_temperature():
    s = torch.tensor([[1., -1.], [-2., 1.]], requires_grad=True)
    t = torch.tensor([[.3, .8], [1., -.2]], requires_grad=True)
    temperature = 2.
    p, q = (t / temperature).softmax(1), (s / temperature).softmax(1)
    expected = (p * (p.log()-q.log())).sum() / len(s) * temperature**2
    value = soft_loss(s, t, temperature)
    torch.testing.assert_close(value, expected)
    value.backward()
    assert s.grad is not None and t.grad is None


def test_reverse_reconstruction_dimension_and_gradients():
    kernel = PairwiseKernel("reverse", 3)
    recon = ReverseReconstruction(3, 7)
    h, x = torch.randn(5, 3), torch.randn(5, 7)
    value = recon(h, x, kernel)
    expected = ((recon.decoder(kernel(h) @ h) - x)**2).sum()
    torch.testing.assert_close(value, expected)
    value.backward()
    assert kernel.projection.weight.grad is not None
    assert recon.decoder.weight.grad is not None


def test_shape_validation():
    with pytest.raises(ValueError):
        hidden_loss([torch.zeros(2, 3)], [torch.zeros(3, 3)], [PairwiseKernel("gaussian", 3)])
    with pytest.raises(ValueError):
        matrix_distance(torch.zeros(2, 2), torch.zeros(3, 3))
