# Verification report

Executed 11 October 2026 on macOS ARM64, CPU, Python 3.12.14, PyTorch 2.14.1, PyG 2.8.0.post1, NumPy 2.5.4 and SciPy 1.18.1. Exact installed packages: [`requirements-tested.txt`](../requirements-tested.txt).

## Code checks

**29 tests passed** ([log](pytest.txt)): independent scalar kernel formulas, squared-distance scaling, KL direction and temperature, detached teacher gradients, reconstruction dimensions, normalized Laplacian residuals, split disjointness and withheld-node removal, exact all-neighbor SAGE sampling/inference equivalence, all three teachers, all kernels, MLP/GLNN/KMP pipelines, checkpoint reload, graph-free student predictions, MiniGC extension, configuration overrides and grid definition.

Additional executed checks: actual Cora GCN/GAT three-epoch runs; Cora inductive SAGE/KMP training; selected-configuration smoke run; absolute Gaussian-noise evaluation; MiniGC extension with 10 seeds. For each default-comparison seed, all teacher state tensors were exactly equal between GLNN, KMP and KMP+PE. Thus the default comparison controls teacher weights as well as splits.

## Measured Cora results

Random splits, 20 labeled training and 30 validation nodes per class; seeds 0–9; at most 1000 epochs; patience 50; batch size 512; best validation checkpoints restored. Mean ± sample standard deviation, test accuracy in percent. PE dimension is 16. All results below were measured, not copied from the paper. Per-seed records and actual configurations are in [`cora_benchmarks.json`](cora_benchmarks.json).

| Method | Accuracy (%) |
|---|---|
| SAGE teacher (default lr=.01) | 76.31 ± 1.73 |
| MLP | 51.12 ± 1.82 |
| GLNN | 75.34 ± 2.53 |
| KMP, sum (default) | 71.83 ± 1.81 |
| KMP+PE, sum (default) | 72.71 ± 2.60 |
| KMP, sum (validation-selected) | 71.46 ± 1.83 |
| KMP+PE, sum (validation-selected) | 72.46 ± 2.71 |
| KMP, mean (normalization control) | 76.10 ± 1.63 |
| KMP+PE, mean (normalization control) | 75.96 ± 2.53 |

The default SAGE learning rate is .01. θ=.5, Gaussian heat time=1 and KL temperature=1. Hidden widths follow Table 8. The two sum-reduction selections used a predeclared 36-setting grid per PE mode: four kernels × γ in {.1,.3,1} × lr in {.01,.005,.001}; tuning seed 0; selection by student validation accuracy only. [`tuning_no_pe.json`](tuning_no_pe.json) and [`tuning_pe.json`](tuning_pe.json) contain every candidate's validation score. Both selected Gaussian γ=.3; selected lr is .005 without PE and .01 with PE. Learning rate affected both stages. The selected parameters were then fixed for 10-seed measurement.

The mean-reduction rows are a separate normalization control, not a kernel choice selected using test accuracy. Mean and sum losses differ by the number of matrix entries, so their γ values are not directly comparable. Both interpretations are exposed in the code and described in [FAITHFULNESS.md](../docs/FAITHFULNESS.md).

**The paper's original numbers are not fully reproduced.** For context, the paper reports Cora/SAGE GLNN 78.99±1.90, KMP 79.48±1.48 and KMP+PE 80.97±1.03. These refer to the historical setup, including its selected kernels and hyperparameters. Missing original details and our smaller validation search prevent a claim of numerical equivalence. The observed normalization sensitivity should be resolved against the historical loss definition if it can be recovered.

## Additional measurements and limits

The Cora inductive smoke run withheld 539 nodes before every training/preprocessing stage and obtained 65.49% on those nodes (one seed, not a variance estimate). The independently generated MiniGC extension obtained 91.50±3.88% over 10 seeds; its explicit degree features/readout assumptions differ from the unspecified historical architecture, so this does not reproduce Table 6.

Cora was processed by PyG: 2708 nodes, 1433 input features and 10556 directed adjacency entries after cleanup. Original raw-file Git blob SHAs are recorded in [`cora_git_blob_shas.json`](cora_git_blob_shas.json). Training tensors/checkpoints remain local and are not committed to this source repository.

**Not executed:** the full 1620-setting Table 9 grid, six remaining node datasets, OGB-scale/CUDA runs, all teacher/kernel/setting combinations and complete original robustness figures. GitHub CI configuration is supplied, but this report describes local verification; it does not assert a remote CI result.

## Re-run

From the repository root after installation:

```bash
python -m pytest -q
python -m kmp --config configs/cora_sage.yaml
python -m kmp --config configs/cora_selected_no_pe.yaml
python -m kmp --config configs/cora_selected_pe.yaml
python -m kmp --config configs/cora_mean_pe0.yaml
python -m kmp --config configs/cora_mean_pe16.yaml
```

These commands download the normal public datasets into `data/`. The original local measurements used an equivalent cached `work/data/` path because direct GitHub downloads timed out and raw files were retrieved through the authenticated GitHub connector. Metrics JSON files preserve the actual run configuration, data-path convention and software versions.
