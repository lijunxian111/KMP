# Paper correspondence and implementation assumptions

Source: [arXiv:2403.01079v1](https://arxiv.org/html/2403.01079v1), submitted 2 March 2024. The author's original code is unavailable. This repository reconstructs the described algorithms; it does not recover the historical implementation or certify every reported number.

## Correspondence

| Paper | Implementation | Status |
|---|---|---|
| Three stages, §4.1 | `train_teacher`, `train_student`, `kmp.infer` | Implemented |
| Eq. (4), normalized Laplacian | `position.laplacian_pe` | Implemented, sparse eigensolver for >512 nodes |
| Eqs. (7)–(8), PE projection/fusion | `models.Student` | Both concatenation and elementwise multiplication |
| Eqs. (9)–(10), four pairwise kernels | `kernels.PairwiseKernel` | Implemented with documented kernel parameters |
| Eq. (11), matrix distance | `matrix_distance` | Squared Frobenius norm (entrywise sum); optional mean is an explicit alternate scale |
| Eq. (12), average across hidden layers | `hidden_loss` | Implemented, same node ordering and hidden width |
| Eq. (13), true/soft target loss | `train_student`, `soft_loss` | Training labels only; teacher→student KL |
| Eqs. (14)–(15), reverse kernel | `PairwiseKernel(reverse)`, `ReverseReconstruction` | Experimental reconstruction with a decoder; two-layer networks only |
| Table 8 | `train.PRESETS` | Teacher depth, width, dropout, weight decay, heads and fanout |
| Table 9 | `configs/grid.yaml`, `kmp.search` | Full grid defined; validation-only selection |
| MLP / GLNN / KMP / KMP+PE | `--method`, `--pe-dim` | Implemented |
| §5.9, feature noise | `kmp.robustness` | Explicit test-time Gaussian-noise assumption |
| §5.10, MiniGC graph classification | `kmp.minigc` | Separate exploratory extension; paper omits architecture/features |

## Assumptions needed because the paper does not specify them

- **Data split:** 20 training nodes and 30 validation nodes **per class** sampled without replacement, all remaining observed nodes for testing. The paper's wording for the 30 validation nodes is ambiguous; per-class sampling follows common low-label graph benchmarks. Random seeds are saved and shared by teacher and student. OGB can alternatively use `--split-protocol official`, which is explicitly a different protocol.
- **Student:** same hidden width and depth as the teacher, ReLU hidden activations; dropout and weight decay default to zero. Matching uses post-activation, pre-dropout hidden states. Batch normalization is optional because Table 8 says “if used”. Singleton batches use running statistics.
- **Teacher:** GCN is full-batch; GAT is full-batch and averages three hidden attention heads to retain total hidden width 128 (head-width conventions are unspecified). SAGE trains with fanout 5 at each layer, batch size 512, and evaluates using all neighbors in exact layerwise batches. GCN/GAT training on graphs over 100,000 nodes is deliberately rejected; large-graph support here is SAGE only.
- **PE:** default comparison is KMP without PE, consistent with Tables 1–4. Use `--pe-dim 16` for a common explicit choice for KMP+PE. Linear projection maps the k-dimensional PE to the original feature width d. All zero Laplacian eigenvalues are excluded, including disconnected components; isolated vertices follow the literal I−D⁻¹ᐟ²AD⁻¹ᐟ² convention with diagonal 1. Signs are canonicalized; repeated eigenvalues can still yield rotated bases across libraries. PE preprocessing requires graph structure and is not included in claims of graph-free inference.
- **Inductive setting:** a stratified 20% withholding is available through `--inductive-fraction .2`; no withholding is the default. Withheld features, labels and edges never enter teacher training, teacher target generation, student training or PE calculation. The paper does not define PE for new nodes. By default inductive PE raises an error; `--inductive-pe zero` explicitly evaluates a zero-PE fallback and must not be described as exact reproduction of Table 5.
- **Hard versus soft supervision:** hard targets use labeled training nodes; soft targets use disjoint observed nodes, including validation/test features but never their labels. This is a transductive protocol. Hard-node batches are independently sampled. θ is the Eq. (13) mixing weight, not a second random split of labeled nodes. Table 9's label-percentage wording is ambiguous.
- **KL temperature:** KL(teacher || student), averaged per node and multiplied by temperature², the standard knowledge-distillation convention. The precise direction, normalization and temperature scaling are not expanded in Eq. (13).
- **Hidden-distance normalization:** Eq. (11) is interpreted as squared Frobenius **sum**, and the loss is averaged over matched hidden layers as Eq. (12) specifies. `--kernel-reduction mean` is available but changes the meaning of γ by the square of batch size. The Gaussian heat time T and soft-logit temperature t are separate.
- **Sigmoid kernel:** f_proj is a fixed identity affine transformation, i.e. sigmoid(hᵢᵀhⱼ). The paper does not state its scale, bias or optimization rule. The stored scale and bias are 1 and 0.
- **Randomized kernel:** shared fixed Gaussian matrices M_r, fixed ξ_r drawn from N(0,1), sigmoid activation, 8 terms of output width 64 by default. The paper specifies Gaussian matrices but does not give term count, feature dimension or the distribution of ξ. All buffers are saved in checkpoints.
- **Polynomial kernel:** (hᵢᵀhⱼ + c)^d, with c=1 and d=2 by default; the typesetting in Eq. (10) is ambiguous. Features are not silently normalized. Finite losses and gradients are checked; very large polynomial losses may need smaller lr or degree/offset.
- **Reverse kernel:** shared W_k, sigmoid activation and Eq. (14) Gram matrix. Eq. (15) multiplies an m×m matrix by an m×hidden_dim tensor, producing m×hidden_dim, while X can have input_dim columns. A learned linear decoder resolves that mismatch, and is an explicit addition to the literal equation. Alternating optimization updates W_k/decoder using detached teacher and student states, then updates the student with W_k fixed. Reconstruction uses raw node features, without PE. This path is an exploratory interpretation, not claimed to reproduce Figure 4 exactly.
- **Learning rates:** by default `--lr` applies to both teacher and student; `--teacher-lr` can hold Stage I learning rate fixed while tuning Stage II. The reduced local search varied the common learning rate and records it explicitly.
- **Stopping/selection:** at most 1000 epochs and patience 50 on validation accuracy. Strict improvements reset patience. Restore the best validation checkpoint before testing. Grid selection uses mean validation accuracy only; test results never select hyperparameters. Ten-seed reports use sample standard deviation (ddof=1), which the paper does not specify.
- **Amazon components:** retain the largest connected component. This is an inference from Table 7: its Amazon node/edge counts are smaller than the commonly distributed full graphs, consistent with component filtering. Exact historical preprocessing is unavailable; record the actual observed size before claiming numerical equivalence. Other datasets retain all vertices.
- **Feature preprocessing/noise:** row-normalized features for Planetoid/Amazon, native features for OGB. Noise study uses test-time X + p·N(0,1) with fixed cached PE; the paper does not specify noise distribution, scale or whether it applies during training.

- **MiniGC extension:** eight graph families generated independently with NetworkX; 240 training, 60 validation, 60 test graphs. Requested sizes are 10–19 as in the public MiniGC generator's exclusive upper bound; hypercubes and grids can have fewer nodes after rounding. Node features are degrees including self loops, readout is mean pooling plus a linear 8-class classifier. Kernels compare nodes within each graph, with mean matrix reduction and mean across graphs. These are explicit choices because §5.10 supplies no features, architecture or graph-level loss. They are not asserted to match the old implementation.

## What verification means

Scalar-formula tests validate kernel values, matrix-loss scaling and KL direction; integration tests validate all teacher kinds, all kernels, PE, inductive withholding, baselines and checkpoint reload. A CSR-sampling test compares exact all-neighbor SAGE outputs against ordinary whole-graph inference. These checks establish the tested implementation's behavior, not an absence of all possible bugs.

Measured benchmark results belong in `verification/`; original paper values must always be labeled separately. Neither a successful smoke test nor numerical agreement on one dataset proves reproduction of all seven datasets, all teachers, all hyperparameters and all paper figures. CUDA/OGB-scale execution must be validated on a suitable GPU machine.
