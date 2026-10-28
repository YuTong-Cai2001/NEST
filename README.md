# NEST: Zero-shot cross-domain dynamic graph anomaly detection with normality-guided learning and evolving-neighborhood scoring

NEST performs zero-shot and few-shot anomalous-node detection across dynamic graphs. Source normality guides representation learning; evolving target neighborhoods provide the reference for anomaly scoring. Few-shot adaptation selects a bounded score correction using support labels while keeping the backbone frozen.

[View the paper's framework (Figure 1, PDF)](Figure_1.pdf)

## Installation and data

```bash
git clone https://github.com/YuTong-Cai2001/NEST.git
cd NEST
pip install -r requirements.txt
```

Use Python 3.10 or 3.11 and a PyTorch build compatible with your CUDA environment. Place DGraphFin under `data/DGraphFin2/` and FiGraph under `data/FiGraph-main/data/`. The trainer also accepts `--dgraphfin_path` and `--figraph_path`. Datasets and trained checkpoints are not included.

Obtain the source data from the [official DGraph website](https://dgraph.xinye.com/) (DGraphFin2 release) and the target data from the [FiGraph authors' repository](https://github.com/XNetLab/FiGraph). These datasets were created and released by their respective authors; please cite Huang et al., *DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection* (NeurIPS 2022), and Wang et al., *FiGraph: A Dynamic Heterogeneous Graph Dataset for Financial Anomaly Detection* (WWW Companion 2025), and follow their dataset terms.

The source loader expects `dgraphfin.npz`, `dgraphfinv2_edge_timestamp.npy`, and `dgraphfinv2_node_timestamp.npy` in the DGraphFin2 directory. The FiGraph directory should contain year folders `2018/` through `2022/`, each with `edgesYEAR.csv` and `ListedCompanyFeatures772_YEAR.csv` from the official release.


## Training and evaluation

Run commands from the repository root. On Windows, the launcher trains the three paper seeds, selects checkpoints by source training loss, and evaluates zero-shot transfer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Seeds 42,43,44 -Epochs 50 -Variants full -SkipFewShot
```

The main protocol uses all retained source windows (currently 20), all valid binary source labels, and the lowest total source training loss for checkpoint selection. Target labels are used only for final evaluation after checkpoint selection. Checkpoints are written to `outputs/nest_fullseq_loss_full_seed*/checkpoints/best_source_loss.pth`. All component, scoring, and few-shot analyses use these source-loss-selected checkpoints. A separate supplementary main comparison uses `--source_protocol stratified_validation`: the common class-stratified 30% training / 10% validation split and highest source-validation ROC-AUC. Evaluate fixed-query few-shot correction with the main-protocol checkpoints:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_few_shot_curve.ps1 -Seeds 42,43,44 -Shots 5,10,15,20 -NormalMultipliers 1,2,3,5
```

New measurements are written under `results/reproduced/`. Omitting `-SkipFewShot` in the training command runs the fixed-query few-shot evaluation after each full-model run. The training launcher also supports the paper's component variants: `svd_alignment,no_descriptors,no_residual,no_self,no_proto,dual`.

The same core entry points can be run directly in any shell; this example uses seed 42:

```bash
python train_anomaly_gfm.py --mode dynamic --epochs 50 --batch_size 1 --sequence_length 0 --source_protocol all_labels --seed 42 --run_name nest_fullseq_loss_full_seed42 --results_csv results/reproduced/fullseq_loss_zero_shot_runs.csv
python few_shot_curve.py --model_path outputs/nest_fullseq_loss_full_seed42/checkpoints/best_source_loss.pth --seed 42 --shots 5,10,15,20 --normal_multipliers 1,2,3,5 --results_csv results/reproduced/fullseq_loss_few_shot_grid_runs.csv
```

## Core files

| Files | Role |
| --- | --- |
| `anomaly_gfm_complete_model.py` | Spatial-temporal model, source objectives, and target scoring |
| `data_processor.py` | Dataset loading and shared descriptors |
| `train_anomaly_gfm.py` | Source training and zero-shot evaluation |
| `model_adapter.py`, `sparse_graph_utils.py` | Sparse-graph execution dependencies used by training |
| `few_shot_finetune.py`, `few_shot_curve.py` | Frozen-feature score correction and fixed-query evaluation |
| `summarize_few_shot_curve.py` | Mean and population standard deviation of few-shot results |
| `plot_paper_experiment_figures.py` | Recreate Figures 2-4 from the published per-seed CSVs |

Existing class names and experiment keys are retained for checkpoint compatibility.

The current evaluation attempts support-only correction at every requested shot count, including five shots. Support-only cross-validation can select the unchanged zero-shot ranking. Different support ratios have separate result keys and separate aggregates; the zero-shot reference is summarized once per seed. New artifacts are saved beside the new result CSV, rather than overwriting artifacts beside the source checkpoint. The paper-result CSVs below now use the approved full-source-sequence protocol.

To regenerate Figures 2-4 from the published results without training:

```bash
python plot_paper_experiment_figures.py
```

The script writes PDFs, PNG previews, and a statistics audit to `figures/generated/`.

## Paper results

The main results below use all 20 retained source windows, all valid binary source labels, and minimum source training-loss selection. The source-validation protocol appears only in a separate supplementary main comparison; every NEST component, scoring, reference, and few-shot experiment uses the main source-loss checkpoints. Earlier five-window experiments are preserved outside this publication checkout.

Means and population standard deviations over seeds 42, 43, and 44:

| Evaluation | ROC-AUC | Average precision |
| --- | --- | --- |
| Zero-shot, 5,132 labeled target nodes | 0.7705 +/- 0.0129 | 0.0886 +/- 0.0102 |
| Zero-shot on the fixed few-shot query set | 0.76639 +/- 0.01481 | 0.07956 +/- 0.01632 |
| 10 anomalous + 50 normal support nodes on that query set | 0.76986 +/- 0.01427 | 0.08112 +/- 0.01548 |

Few-shot evaluation reserves a maximum support pool of 20 anomalous and 100 normal nodes. The remaining 5,012 query nodes, including 103 anomalies, stay fixed across shot counts within each seed. Ten-shot adaptation uses 10 anomalous and 50 normal support nodes. Its zero-shot reference is recomputed after excluding the entire reserved pool.

`results/` contains the current paper measurements. The component CSV retains its historical filename for script compatibility; its rows now describe the 20-window experiments:

| CSV | Contents |
| --- | --- |
| `zero_shot_runs.csv` | Three full-model zero-shot runs |
| `main_comparison_runs.csv` | Three-seed main comparison, including adapted baselines |
| `source_validation_comparison_runs.csv` | Supplementary main comparison with source-validation selection |
| `component_ablation_final_20261007_table3_runs.csv` | Final matched component comparison, 21 rows |
| `scoring_ablation_runs.csv` | Matched-checkpoint scoring comparison |
| `reference_scoring_runs.csv` | Crossed source-compactness / target-reference comparison |
| `few_shot_curve_runs.csv` | 48 fixed-query runs: four shots, four support ratios, three seeds |
| `few_shot_curve_mean_std.csv` | Few-shot aggregates |

Run identifiers, protocol fields, and checkpoint paths identify the reported experiments. Two previously completed 20-window seed-42 checkpoints are reused and audited; the remaining runs were recomputed for this suite. Referenced checkpoints are not bundled. Baseline CSV rows record the adapted baseline runs; this repository contains the NEST implementation.

Authors: Yutong Cai, Xiang Feng, and Huiqun Yu.

