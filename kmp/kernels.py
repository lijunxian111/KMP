"""Pairwise kernels corresponding to paper Eqs. (9)--(15)."""
import torch
from torch import nn
import torch.nn.functional as F


class PairwiseKernel(nn.Module):
    def __init__(self, kind, dim, heat_time=1.0, degree=2, offset=1.0,
                 random_terms=8, random_dim=64):
        super().__init__()
        if kind not in {"gaussian", "sigmoid", "polynomial", "randomized", "reverse"}:
            raise ValueError(f"Unknown kernel: {kind}")
        if dim < 1 or heat_time <= 0 or degree < 1 or random_terms < 1 or random_dim < 1:
            raise ValueError("Kernel dimensions, heat time and degree must be positive")
        self.kind, self.dim = kind, dim
        self.heat_time, self.degree, self.offset = heat_time, degree, offset
        # The paper does not specify f_proj. Use a fixed identity affine map.
        self.register_buffer("sigmoid_scale", torch.tensor(1.0))
        self.register_buffer("sigmoid_bias", torch.tensor(0.0))
        if kind == "randomized":
            self.register_buffer("matrices", torch.randn(random_terms, random_dim, dim))
            self.register_buffer("xi", torch.randn(random_terms))
        if kind == "reverse":
            self.projection = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.projection.weight)

    def forward(self, h):
        if h.ndim != 2 or h.shape[1] != self.dim:
            raise ValueError("Kernel input must have shape [nodes, hidden_dim]")
        if self.kind == "gaussian":
            squared = h.square().sum(1)
            dist = (squared[:, None] + squared[None, :] - 2 * (h @ h.T)).clamp_min(0)
            return torch.exp(-dist / (4 * self.heat_time))
        if self.kind == "sigmoid":
            return torch.sigmoid(self.sigmoid_scale * (h @ h.T) + self.sigmoid_bias)
        if self.kind == "polynomial":
            return (h @ h.T + self.offset).pow(self.degree)
        if self.kind == "randomized":
            z = torch.sigmoid(torch.einsum("trd,nd->tnr", self.matrices, h))
            return torch.einsum("tnr,tmr,t->nm", z, z, self.xi.exp()) / len(self.xi)
        z = torch.sigmoid(self.projection(h))
        return z @ z.T


def matrix_distance(student, teacher, reduction="mean"):
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("Kernel matrices must have the same shape")
    if reduction not in {"sum", "mean"}:
        raise ValueError("Reduction must be sum or mean")
    error = (student - teacher).square()
    return error.sum() if reduction == "sum" else error.mean()


def hidden_loss(student_h, teacher_h, kernels, reduction="mean"):
    if not student_h or len(student_h) != len(teacher_h) or len(kernels) != len(student_h):
        raise ValueError("One shared kernel is required for each matched hidden layer")
    losses = []
    for s, t, kernel in zip(student_h, teacher_h, kernels):
        if s.shape != t.shape:
            raise ValueError("Teacher/student hidden dimensions and node order must match")
        # Teacher parameters are frozen; reverse-kernel parameters remain differentiable.
        losses.append(matrix_distance(kernel(s), kernel(t.detach()), reduction))
    return torch.stack(losses).mean()


def soft_loss(student_logits, teacher_logits, temperature=1.0):
    if temperature <= 0 or student_logits.shape != teacher_logits.shape:
        raise ValueError("Positive temperature and matching logits required")
    return F.kl_div(F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits.detach() / temperature, dim=-1),
                    reduction="batchmean") * temperature**2


class ReverseReconstruction(nn.Module):
    """Eq. (15), with an explicit decoder to resolve hidden/input width mismatch."""
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.decoder = (nn.Identity() if hidden_dim == input_dim
                        else nn.Linear(hidden_dim, input_dim, bias=False))

    def forward(self, h, x, kernel, reduction="sum"):
        reconstructed = self.decoder(kernel(h) @ h)
        return matrix_distance(reconstructed, x, reduction)
