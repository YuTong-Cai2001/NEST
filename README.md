# NEST: Normality-guided learning and evolving-neighborhood scoring for cross-domain dynamic graph anomaly detection

NEST performs zero-shot and few-shot anomalous-node detection across dynamic graphs. Source normality guides representation learning; evolving target neighborhoods provide the reference for anomaly scoring. Few-shot adaptation selects a bounded score correction using support labels while keeping the backbone frozen.

[View the paper's framework (Figure 1, PDF)](Figure_1.pdf)

## Installation and data

```bash
git clone https://github.com/YuTong-Cai2001/NEST.git
cd NEST
pip install -r requirements.txt
```

Use Python 3.10 or 3.11 and a PyTorch build compatible with your CUDA environment. Place DGraphFin under `data/DGraphFin2/` and FiGraph under `data/FiGraph-main/data/`. The trainer also accepts `--dgraphfin_path` and `--figraph_path`. Datasets and trained checkpoints are not included.

## Training and evaluation

Run commands from the repository root. On Windows, the launcher trains the three paper seeds, selects checkpoints by source training loss, and evaluates zero-shot transfer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Seeds 42,43,44 -Epochs 50 -Variants full -SkipFewShot
```

Checkpoints are written to `outputs/nest_paper_full_seed*/`. Evaluate fixed-query few-shot correction with those checkpoints:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_few_shot_curve.ps1 -Seeds 42,43,44 -Shots 5,10,20
```

New measurements are written under `results/reproduced/`. Omitting `-SkipFewShot` in the training command runs the fixed-query few-shot evaluation after each full-model run. The training launcher also supports the paper's component variants: `svd_alignment,no_descriptors,no_residual,no_self,no_proto,dual`.

The same core entry points can be run directly in any shell; this example uses seed 42:

```bash
python train_anomaly_gfm.py --mode dynamic --epochs 50 --batch_size 1 --sequence_length 5 --seed 42 --run_name nest_paper_full_seed42 --results_csv results/reproduced/zero_shot_runs.csv
python few_shot_curve.py --model_path outputs/nest_paper_full_seed42/checkpoints/best_source_loss.pth --seed 42 --shots 5,10,20 --results_csv results/reproduced/few_shot_curve_runs.csv
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

Existing class names and experiment keys are retained for checkpoint compatibility.

## Paper results

Means and population standard deviations over seeds 42, 43, and 44:

| Evaluation | ROC-AUC | Average precision |
| --- | --- | --- |
| Zero-shot, 5,132 labeled target nodes | 0.7704 +/- 0.0132 | 0.0926 +/- 0.0114 |
| Zero-shot on the fixed few-shot query set | 0.7665 +/- 0.0146 | 0.0834 +/- 0.0188 |
| 10-shot correction on that same query set | 0.7700 +/- 0.0136 | 0.0845 +/- 0.0182 |

Few-shot evaluation reserves a maximum support pool of 20 anomalous and 100 normal nodes. The remaining 5,012 query nodes, including 103 anomalies, stay fixed across shot counts within each seed. Ten-shot adaptation uses 10 anomalous and 50 normal support nodes. Its zero-shot reference is recomputed after excluding the entire reserved pool.

`results/` contains the paper's measurements:

| CSV | Contents |
| --- | --- |
| `zero_shot_runs.csv` | Three full-model zero-shot runs |
| `main_comparison_runs.csv` | Three-seed main comparison, including adapted baselines |
| `component_ablation_final_20261007_table3_runs.csv` | Final matched component comparison, 21 rows |
| `scoring_ablation_runs.csv` | Matched-checkpoint scoring comparison |
| `reference_scoring_runs.csv` | Crossed source-compactness / target-reference comparison |
| `few_shot_curve_runs.csv` | Nine fixed-query few-shot runs |
| `few_shot_curve_mean_std.csv` | Few-shot aggregates |

Historical run identifiers and checkpoint paths in archived CSVs identify the original experiments. Referenced checkpoints are not bundled. Baseline CSV rows record the adapted baseline runs; this repository contains the NEST implementation.

Authors: Yutong Cai, Xiang Feng, and Huiqun Yu.

