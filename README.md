# NEST: Normality-guided learning and evolving-neighborhood scoring for cross-domain dynamic graph anomaly detection

## Overview

NEST is a framework for zero-shot and few-shot anomalous-node detection across dynamic graphs with heterogeneous node attributes. It separates the role of source normality in representation learning from the reference used to score target anomalies: source-domain supervision shapes the encoder, while evolving target neighborhoods provide the comparison context at inference.

The manuscript implementation is available directly on [`main`](https://github.com/YuTong-Cai2001/NEST/tree/main). Existing experiment identifiers retain their historical spelling so that saved checkpoints and result files remain usable. See [RUNNING.md](RUNNING.md) for reproduction commands and [MANUSCRIPT_VERSION.md](MANUSCRIPT_VERSION.md) for the verified code and result provenance.

## Method

1. **Shared inputs for heterogeneous graphs.** Sixteen distributional attribute descriptors and eight structural-temporal descriptors provide a common 24-dimensional input for DGraphFin and FiGraph.
2. **Normality-guided representation learning.** A dynamic GNN and a causal TCN are coupled by residual interaction and hierarchical fusion. Source training combines supervised detection, normal-prototype compactness, and a temporal class-mean objective.
3. **Evolving-neighborhood zero-shot scoring.** The frozen backbone compares normalized target-node states with their contemporaneous neighbor means. Fixed linear recency weights aggregate disagreement across snapshots, and empirical percentile ranks produce graph-relative anomaly scores. Target labels are not used to fit the encoder or the zero-shot scoring rule.
4. **Support-only few-shot correction.** The zero-shot ranking anchors a bounded correction selected by support-only cross-validation. Candidates include a logistic score readout, a support-prototype readout, and retaining the zero-shot ranking. The backbone remains frozen, and query labels are reserved for evaluation.

The normal prototype is a source representation constraint, not the default target anomaly-scoring reference. Prototype transport and alternative score features remain available for the scoring analyses and support-selected corrections; they are not the default zero-shot score.

## Manuscript results

Results below are means and population standard deviations over seeds 42, 43, and 44 for DGraphFin-to-FiGraph transfer.

| Evaluation | ROC-AUC | Average precision |
| --- | --- | --- |
| Zero-shot main evaluation: 5,132 labeled target nodes | 0.7704 +/- 0.0132 | 0.0926 +/- 0.0114 |
| Zero-shot reference on the fixed few-shot query set | 0.7665 +/- 0.0146 | 0.0834 +/- 0.0188 |
| 10-shot correction on the same query set | 0.7700 +/- 0.0136 | 0.0845 +/- 0.0182 |

The main evaluation includes 123 anomalous and 5,009 normal nodes from the final target snapshot. Few-shot evaluation reserves a maximum support pool of 20 anomalous and 100 normal nodes, leaving 5,012 query nodes, including 103 anomalies. The 10-shot setting uses 10 anomalous and 50 normal support nodes. Within each seed, the query set excludes the entire maximum support pool and remains fixed across the 5-, 10-, and 20-shot settings. The zero-shot query reference is therefore recomputed on that reduced evaluation population rather than copied from the main table.

## Installation

Clone the renamed repository; its default `main` branch contains the manuscript implementation:

```bash
git clone https://github.com/YuTong-Cai2001/NEST.git
cd NEST
```

Use Python 3.10 or 3.11. Install a PyTorch build compatible with your CUDA environment, then install the remaining requirements:

```bash
pip install -r requirements.txt
python smoke_test.py
```

Both `single: OK` and `dual: OK` should appear in the smoke-test output. The code uses the existing module and class names, including `AnomalyGFMModel`, for checkpoint compatibility; these identifiers do not refer to a different manuscript method.

## Data

- **Source: DGraphFin.** The source has 17-dimensional raw attributes and timestamped interactions. The manuscript uses the existing source sampling and temporal-window construction in the training pipeline.
- **Target: FiGraph.** The target sequence covers 2018-2022 and has 772-dimensional raw attributes before descriptor construction. Inference uses the observed target graph; ROC-AUC and AP are computed only for final-snapshot nodes whose released label is exactly 0 or 1. Background entities labeled `-1` are not treated as normal examples.

Place the released datasets at the paths expected by `data_processor.py` and the launchers. Raw datasets are not distributed as part of this repository.

## Usage

All commands below are run from the repository root (called `ProtoDGAD` in the original local workspace).

### Zero-shot training and evaluation

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -RunPrefix nest_paper -Seeds 42,43,44 -Epochs 50 -Variants full -SkipFewShot
```

This trains the full model, restores the checkpoint with the lowest source training loss, and evaluates the frozen model on FiGraph. It writes separate checkpoints under `outputs/nest_paper_full_seed*/`. No training is required merely to rename the method.

### Fixed-query few-shot evaluation

For the retained manuscript checkpoints:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_few_shot_curve.ps1 -CheckpointPrefix spad_ablation_no_reg_seed -Seeds 42,43,44 -Shots 5,10,20
python validate_few_shot_curve.py
```

For checkpoints produced by the zero-shot command above, use `-CheckpointPrefix nest_paper_full_seed` instead. The launcher keeps the backbone frozen, uses nested support sets, and selects score corrections using support labels only. At five shots, the current protocol retains the zero-shot anchor because correction fitting requires at least 10 anomalous support nodes.

The separate `few_shot_adapter.py` and its launcher implement an alternative residual-adapter experiment, not the few-shot procedure reported in the current manuscript.

## Main files

| File | Purpose |
| --- | --- |
| `anomaly_gfm_complete_model.py` | Dynamic GNN, causal TCN, residual interaction, fusion, and source objectives |
| `data_processor.py` | Graph loading and shared descriptor construction |
| `train_anomaly_gfm.py` | Source training and zero-shot target evaluation |
| `few_shot_finetune.py` | Frozen-feature extraction and support-only score correction |
| `few_shot_curve.py` | Fixed-query, nested-support few-shot evaluation |
| `evaluate_reference_scoring.py` | Crossed comparison of training compactness and scoring reference |
| `run_final_component_ablation.py` | Final-objective component-ablation launcher |
| `validate_few_shot_curve.py` | Checks saved query and support-set consistency |

## Paper and citation

Paper title: **NEST: Normality-guided learning and evolving-neighborhood scoring for cross-domain dynamic graph anomaly detection**.

Authors: Yutong Cai, Xiang Feng, and Huiqun Yu. Publication details will be added when available.

## License and contact

This project is licensed under the MIT License. For questions, please open an issue in the [NEST repository](https://github.com/YuTong-Cai2001/NEST).

