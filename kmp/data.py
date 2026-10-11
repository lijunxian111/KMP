from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from torch_geometric.utils import to_undirected, remove_self_loops, subgraph


@dataclass
class GraphSplit:
    observed: Data
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    observed_ids: torch.Tensor
    heldout: torch.Tensor


def load_graph(name, root):
    if name in {"Cora", "Citeseer", "Pubmed"}:
        graph = Planetoid(root=str(root), name=name, transform=NormalizeFeatures())[0]
    elif name in {"Photo", "Computers"}:
        graph = Amazon(root=str(root), name=name, transform=NormalizeFeatures())[0]
        graph = LargestConnectedComponents(num_components=1)(graph)
    elif name in {"ogbn-arxiv", "ogbn-products"}:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=name, root=str(root))
        graph = dataset[0]
        graph.y = graph.y.flatten()
        graph.official_split = dataset.get_idx_split()
    elif name == "synthetic":
        rng = np.random.default_rng(123)
        y = np.repeat(np.arange(3), 80)
        x = rng.normal(0, .2, (240, 12))
        x[np.arange(240), y] += 1
        rows, cols = np.where(rng.random((240, 240)) < np.where(y[:, None] == y, .12, .006))
        graph = Data(x=torch.tensor(x, dtype=torch.float32), y=torch.tensor(y),
                     edge_index=torch.tensor(np.stack([rows, cols]), dtype=torch.long))
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    graph.edge_index = to_undirected(remove_self_loops(graph.edge_index)[0],
                                    num_nodes=graph.num_nodes)
    return graph


def split_graph(graph, seed, train_per_class=20, val_per_class=30,
                inductive_fraction=0.0, protocol="random"):
    if not 0 <= inductive_fraction < 1 or min(train_per_class, val_per_class) < 1:
        raise ValueError("Invalid split settings")
    if protocol not in {"random", "official"}:
        raise ValueError("Unknown split protocol")
    if protocol == "official" and inductive_fraction:
        raise ValueError("Official OGB + inductive withholding is not defined; use random")
    rng = np.random.default_rng(seed)
    train, val, test, heldout = [], [], [], []
    if protocol == "official":
        if not hasattr(graph, "official_split"):
            raise ValueError("Official split is supported for OGB only")
        train = graph.official_split["train"].tolist()
        val = graph.official_split["valid"].tolist()
        test = graph.official_split["test"].tolist()
    else:
        labels = graph.y.numpy()
        for label in np.unique(labels):
            ids = rng.permutation(np.where(labels == label)[0])
            count = int(len(ids) * inductive_fraction)
            heldout.extend(ids[:count].tolist())
            ids = ids[count:]
            if len(ids) <= train_per_class + val_per_class:
                raise ValueError(f"Class {label} has too few observed nodes for this split")
            train.extend(ids[:train_per_class].tolist())
            val.extend(ids[train_per_class:train_per_class+val_per_class].tolist())
            test.extend(ids[train_per_class+val_per_class:].tolist())
    heldout = torch.tensor(sorted(heldout), dtype=torch.long)
    observed_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    observed_mask[heldout] = False
    observed_ids = observed_mask.nonzero().flatten()
    remap = torch.full((graph.num_nodes,), -1, dtype=torch.long)
    remap[observed_ids] = torch.arange(len(observed_ids))
    edges, _ = subgraph(observed_ids, graph.edge_index, relabel_nodes=True,
                        num_nodes=graph.num_nodes)
    observed = Data(x=graph.x[observed_ids], y=graph.y[observed_ids], edge_index=edges)
    return GraphSplit(observed, remap[torch.tensor(train)], remap[torch.tensor(val)],
                      remap[torch.tensor(test)], observed_ids, heldout)


def adjacency(edge_index, n):
    source, target = edge_index.cpu().numpy()
    matrix = sp.coo_matrix((np.ones(len(source)), (target, source)), shape=(n, n)).tocsr()
    matrix.data[:] = 1
    return matrix


class Neighborhood:
    """Seeded CSR sampler without optional torch-sparse/pyg-lib extensions."""
    def __init__(self, edge_index, n, seed):
        self.csr = adjacency(edge_index, n)
        self.rng = np.random.default_rng(seed)

    def block(self, targets, fanout=-1):
        targets = np.asarray(targets, dtype=np.int64)
        sources = targets.tolist()
        lookup = {int(node): i for i, node in enumerate(sources)}
        rows, cols = [], []
        for i, target in enumerate(targets):
            neighbors = self.csr.indices[self.csr.indptr[target]:self.csr.indptr[target+1]]
            if fanout >= 0 and len(neighbors) > fanout:
                neighbors = self.rng.choice(neighbors, fanout, replace=False)
            for node in neighbors:
                node = int(node)
                if node not in lookup:
                    lookup[node] = len(sources)
                    sources.append(node)
                rows.append(lookup[node])
                cols.append(i)
        edges = torch.tensor([rows, cols], dtype=torch.long)
        return np.asarray(sources), (edges, len(targets))

    def sample(self, targets, fanouts):
        nodes, blocks = np.asarray(targets), []
        for fanout in reversed(fanouts):
            nodes, block = self.block(nodes, fanout)
            blocks.insert(0, block)
        return torch.from_numpy(nodes), blocks
