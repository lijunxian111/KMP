import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class SafeBatchNorm(nn.BatchNorm1d):
    def forward(self, x):
        if self.training and len(x) == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, training=False,
                                momentum=self.momentum, eps=self.eps)
        return super().forward(x)


class Teacher(nn.Module):
    def __init__(self, input_dim, hidden_dim, classes, kind="sage", layers=2,
                 dropout=0.0, heads=3, attention_dropout=0.6, batch_norm=False):
        super().__init__()
        if layers < 2 or kind not in {"gcn", "sage", "gat"}:
            raise ValueError("Teacher needs at least two layers and a valid GNN kind")
        self.kind, self.dropout = kind, dropout
        widths = [input_dim] + [hidden_dim] * (layers - 1) + [classes]
        convs = []
        for i, (src, dst) in enumerate(zip(widths, widths[1:])):
            if kind == "gcn":
                convs.append(GCNConv(src, dst, cached=False))
            elif kind == "sage":
                convs.append(SAGEConv(src, dst))
            else:
                # Average three heads, retaining the paper's 128-dimensional hidden state.
                convs.append(GATConv(src, dst, heads=heads if i < layers-1 else 1,
                                     concat=False, dropout=attention_dropout))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([SafeBatchNorm(hidden_dim) if batch_norm else nn.Identity()
                                   for _ in range(layers - 1)])

    def activate(self, h, layer):
        return F.relu(self.norms[layer](h))

    def forward(self, x, edge_index):
        hidden = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activate(x, i)
                hidden.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x, hidden

    def sampled(self, x, blocks):
        if self.kind != "sage" or len(blocks) != len(self.convs):
            raise ValueError("Sampled training currently supports GraphSAGE only")
        for i, (conv, block) in enumerate(zip(self.convs, blocks)):
            edges, count = block
            x = conv((x, x[:count]), edges)
            if i < len(self.convs)-1:
                x = self.activate(x, i)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Student(nn.Module):
    def __init__(self, input_dim, hidden_dim, classes, layers=2, dropout=0.0,
                 pe_dim=0, pe_fusion="concat", batch_norm=False):
        super().__init__()
        if layers < 2 or pe_fusion not in {"concat", "mul"}:
            raise ValueError("Invalid student layers or PE fusion")
        self.pe_dim, self.pe_fusion, self.dropout = pe_dim, pe_fusion, dropout
        self.pe_embedding = nn.Linear(pe_dim, input_dim) if pe_dim else None
        first_dim = input_dim * (2 if pe_dim and pe_fusion == "concat" else 1)
        widths = [first_dim] + [hidden_dim] * (layers-1) + [classes]
        self.linears = nn.ModuleList([nn.Linear(a, b) for a, b in zip(widths, widths[1:])])
        self.norms = nn.ModuleList([SafeBatchNorm(hidden_dim) if batch_norm else nn.Identity()
                                   for _ in range(layers-1)])

    def forward(self, x, pe=None):
        if self.pe_embedding is not None:
            if pe is None or pe.shape != (len(x), self.pe_dim):
                raise ValueError("Student requires a matching cached PE tensor")
            position = self.pe_embedding(pe)
            x = torch.cat([x, position], dim=1) if self.pe_fusion == "concat" else x * position
        hidden = []
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears)-1:
                x = F.relu(self.norms[i](x))
                hidden.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x, hidden
