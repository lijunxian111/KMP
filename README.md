# KMP — Teaching MLP More Graph Information

Official Paper-based reconstruction of **Teaching MLP More Graph Information: A Three-stage Multitask Knowledge Distillation Framework**  
**Junxian Li**, [Bin Shi](https://gr.xjtu.edu.cn/shibin/), [Erfei Cui](https://scholar.google.com/citations?user=MRTB_-wAAAAJ&hl=en&oi=ao), [Hua Wei](https://scholar.google.com/citations?hl=en&user=F1CEAKwAAAAJ) and Qinghua Zheng, [arXiv:2403.01079v1](https://arxiv.org/abs/2403.01079).

This implementation was rebuilt from the paper by codex. Missing details use documented GNN conventions. The initial version was done in 2023.06, and many details cannot be traced back after years. Feel so sorry about that.  

Train a GNN teacher, transfer soft predictions and pairwise hidden-layer kernels to an MLP, then run the student on features and optional cached positional encodings without message passing.

## Install

Python 3.10+; CPU works for the small datasets. CUDA is recommended for large graphs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

For OGB datasets: `python -m pip install -e '.[ogb]'`.

## Run

```bash
# Short offline end-to-end check
python -m kmp --dataset synthetic --epochs 3 --pe-dim 4 --output runs/smoke

# Cora, SAGE teacher, KMP without PE, paper's epoch/patience settings, 10 seeds
python -m kmp --config configs/cora_sage.yaml

# KMP + Laplacian PE
python -m kmp --config configs/cora_sage.yaml --pe-dim 16 --output runs/cora_sage_pe

# Baselines (same splits and teacher seed)
python -m kmp --config configs/cora_sage.yaml --method glnn --output runs/cora_glnn
python -m kmp --config configs/cora_sage.yaml --method mlp --output runs/cora_mlp

# Other teachers and kernels
python -m kmp --dataset Citeseer --teacher gcn --kernel randomized --output runs/citeseer_gcn
python -m kmp --dataset Cora --teacher gat --kernel sigmoid --output runs/cora_gat

# Inductive setting: remove held-out nodes and all incident edges before training
python -m kmp --dataset Cora --inductive-fraction .2 --output runs/cora_inductive

# Experimental reverse kernel; documented decoder assumption resolves Eq. (15)
python -m kmp --dataset Cora --kernel reverse --kernel-reduction mean --output runs/cora_reverse
```

Each seed saves configuration, split indices, teacher/student checkpoints, frozen teacher targets, optional PE, validation histories and measured metrics. A single seed has no standard-deviation estimate.

Supported node datasets: `Cora`, `Citeseer`, `Pubmed`, `Photo`, `Computers`, `ogbn-arxiv`, `ogbn-products`. OGB downloads can be large. Large-graph teachers are restricted to sampled SAGE; GCN/GAT do not silently fall back to incorrect local normalization. SAGE requires no optional compiled sampling libraries; the portable CSR sampler prioritizes correctness over throughput.

## Stage III: inference without edges

Prepare feature tensors in the same row order as the cached PE and save them with `torch.save`.

```bash
python -m kmp.infer --checkpoint runs/cora_sage_pe/seed_0/student.pt \
  --features features.pt --pe runs/cora_sage_pe/seed_0/pe.pt --output predictions.pt
```

Omit `--pe` for a student trained without PE. A PE-trained student requires matching precomputed PE. New-node PE is unspecified in the paper; inductive PE is rejected unless the explicit `--inductive-pe zero` fallback is selected.

## Grid search and robustness

```bash
# Print the size of Appendix Table 9's 1,620-point grid
python -m kmp.search --grid configs/grid.yaml

# Execute it; selects by validation accuracy, then writes selected.yaml
python -m kmp.search --grid configs/grid.yaml --tuning-seeds 0 --execute
python -m kmp --config runs/search/selected.yaml

# Feature-noise evaluation of a saved student, no retraining
python -m kmp.robustness --run runs/cora_sage_pe/seed_0 --output runs/noise.json
```

The full grid is expensive. Extend its YAML with a list of kernels to select the best kernel by validation accuracy. γ sensitivity can be run by narrowing the grid to `gamma` only. For a comparison, use identical splits, tuning budget and noise seeds for every baseline.

## Verification and paper fidelity

 The defaults implement squared Frobenius **mean** and distinguish heat time from KL temperature. Optional sum reduction changes γ's scale. GraphSAGE fanout, GNN dropout and weight decay follow Table 8; PE dimension, student regularization and other omitted details are explicit assumptions.

**Local validation:** 29 tests passed. Ten-seed Cora/SAGE accuracy is 75.34±2.53% for GLNN, and 76.10±1.63% for the mean-reduction control. 

## Citation

If you find our paper and reproduction useful, please kindly cite:  

```ruby
@article{li2024teaching,
  title={Teaching MLP More Graph Information: A Three-stage Multitask Knowledge Distillation Framework},
  author={Li, Junxian and Shi, Bin and Cui, Erfei and Wei, Hua and Zheng, Qinghua},
  journal={arXiv preprint arXiv:2403.01079},
  year={2024}
}
```

## Exploratory graph classification

```bash
python -m kmp.minigc --method kmp --output runs/minigc_kmp
python -m kmp.minigc --method glnn --output runs/minigc_glnn
```

This extension uses the eight [MiniGC graph families](https://www.dgl.ai/dgl_docs/generated/dgl.data.MiniGCDataset.html), 240/60/60 graphs, a SAGE teacher, degree features and mean graph readout. Features, architecture, readout and graph-level kernel aggregation were not given in the paper. Graph generation is independent of DGL's exact random sequence; hypercubes/grids round requested node counts down. It is an exploratory reproduction of §5.10's idea. The student forward pass uses precomputed degree features and the graph's node membership, without edges or message passing.
