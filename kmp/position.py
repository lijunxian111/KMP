"""Normalized Laplacian PE; no labels or dense adjacency required."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import torch
from .data import adjacency


def laplacian_pe(edge_index, n, dim, seed=0, tolerance=1e-8):
    if dim < 1 or dim >= n:
        raise ValueError("PE dimension must be between 1 and number of nodes - 1")
    a = adjacency(edge_index, n)
    a = a.maximum(a.T)
    degree = np.asarray(a.sum(1)).flatten()
    inverse = np.zeros(n)
    inverse[degree > 0] = degree[degree > 0] ** -.5
    laplacian = sp.eye(n, format="csr") - sp.diags(inverse) @ a @ sp.diags(inverse)
    _, labels = connected_components(a, directed=False)
    # Eq. (4) has eigenvalue 1, rather than 0, for isolated vertices.
    nullity = len(np.unique(labels[degree > 0]))
    count = dim + nullity
    if count >= n and n > 512:
        raise ValueError("Too many disconnected components for sparse PE at requested dimension")
    if n <= 512:
        values, vectors = np.linalg.eigh(laplacian.toarray())
    else:
        values, vectors = eigsh(laplacian, k=count, which="SM", tol=tolerance,
                                v0=np.random.default_rng(seed).normal(size=n))
    order = np.argsort(values)
    order = order[values[order] > 1e-7][:dim]
    if len(order) != dim:
        raise ValueError("Not enough nontrivial eigenvectors; lower PE dimension")
    values, vectors = values[order], vectors[:, order]
    # Deterministic sign; repeated-eigenvalue rotations remain basis dependent.
    pivots = np.argmax(np.abs(vectors), axis=0)
    vectors *= np.sign(vectors[pivots, np.arange(dim)])[None, :]
    residual = np.linalg.norm(laplacian @ vectors - vectors * values, axis=0).max()
    if not np.isfinite(residual) or residual > 1e-5:
        raise RuntimeError(f"PE eigenvector residual too large: {residual}")
    return torch.tensor(vectors, dtype=torch.float32), values.tolist(), float(residual)
