# SPAD: Single-Prototype Adaptation for Dynamic Graph Anomaly Detection

> The executable, repaired experiment instructions are in [RUNNING.md](RUNNING.md). The pipeline now defaults to a strict single normal prototype; `--prototype_mode dual` is reserved for the ablation.

## Overview

SPAD is a framework for zero-shot and few-shot anomalous-node detection in dynamic graphs. It combines normal-population alignment with temporal consistency learned from source-domain dynamic graphs.

![Framework Overview](docs/framework.png)

## Key Features

- **Zero-Shot Anomaly Detection**: Can detect anomaly nodes in dynamic graphs without any labeled samples from the target domain
- **Parameter-Efficient Few-Shot Adaptation**: Freezes the transferable SPAD backbone and updates a 513-parameter residual prototype adapter from sparse target annotations
- **Temporal Modeling**: Explicitly captures temporal dependencies and structural evolution patterns
- **Dual-Alignment Mechanism**: Combines self-evolution inconsistency modeling with population-level alignment
- **Directional Prototype-Transport Residual**: Scores nodes that fail to follow the label-free source-to-target normal-population shift
- **Dynamic Neighbourhood Inconsistency**: Detects nodes whose representations repeatedly diverge from their incident neighbours across graph snapshots
- **Recency-Aware Scale-Free Scoring**: Gives later snapshots linearly greater influence and calibrates the result by empirical percentile rank without target-fitted weights

## Method Introduction

SPAD is developed based on two core principles:

1. **Self-Evolution Inconsistency**: Anomaly nodes' evolution directions are inconsistent with their historical self-evolution trajectories
2. **Global Evolution Inconsistency**: Anomaly nodes are inconsistent with overall population evolution trends

The framework explicitly models these principles through:
- **Temporal Consistency Constraints**: Captures self-evolution inconsistency
- **Single-Prototype Population Alignment**: Characterizes global evolution inconsistency
- **Robust Target Prototype Transport**: Estimates the target normal centre from informative, temporally stable majority nodes
- **Directional Transport Residual**: Uses `d_target - d_source` as the zero-shot anomaly ranking score, without target labels or a target-fitted mixing weight
- **Recency-Aware Local Inconsistency**: Aggregates representation-to-neighbour disagreement using fixed chronological weights `1,...,T`
- **Coefficient-Free Rank Calibration**: Converts the resulting local evolution score to its target-graph empirical percentile without using target labels

This approach learns transferable discriminative priors under source domain supervision, enabling effective zero-shot transfer to target domains.

## Performance

| Setting | AUC | AP |
|---------|-----|-----|
| Zero-Shot engineering check (DGraphFin → FiGraph 2022, seed 42) | 0.7882 | 0.1084 |

*Verified end-to-end from the final inference path using a 50-epoch seed-42
run. Report multi-seed mean and standard deviation in the paper after
the complete experiment launcher finishes; the old unverified 0.9537/0.0180
placeholder has been removed. Metrics are computed only on FiGraph nodes with
ground-truth labels 0/1; `Label=-1` background nodes are excluded rather than
silently treated as normal. Results produced with those two protocols are not
directly comparable.*

## Project Structure

```
ProtoDGAD/
├── anomaly_gfm_complete_model.py    # SPAD temporal graph model
├── data_processor.py                # Data loading and preprocessing
├── few_shot_finetune.py            # Few-shot fine-tuning implementation
├── few_shot_adapter.py             # Frozen-backbone residual prototype adapter
├── run_few_shot_adapter_curve.ps1  # 5/10/20-shot, three-seed adapter launcher
├── summarize_few_shot_adapter.py   # Adapter mean/std aggregation
├── few_shot_curve.py               # Fixed-query nested-support shot curve
├── run_few_shot_experiments.ps1    # Three-seed few-shot launcher
├── run_few_shot_curve.ps1          # 5/10/20-shot three-seed launcher
├── summarize_few_shot.py           # Paired mean/std aggregation
├── model_adapter.py                # Sparse graph model adapter
├── sparse_graph_utils.py           # Sparse graph processing utilities
├── train_anomaly_gfm.py            # Main training script (zero-shot)
├── utils.py                        # Utility functions
├── plot_results.py                 # CSV aggregation and paper figures
├── run_experiments.ps1             # Reproducible experiment launcher
├── results/                        # Raw and aggregated CSV results
└── outputs/                        # Checkpoints and fitted feature aligners
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric 2.0+
- SciPy
- NumPy
- scikit-learn

### Setup

```bash
# From the ProtoDGAD directory, after installing a CUDA-compatible PyTorch build
pip install -r requirements.txt
python smoke_test.py
```

## Datasets

### Source Domain: DGraphFin

DGraphFin is a financial transaction dataset containing temporal graph data with anomaly labels. It contains:
- Multiple temporal snapshots
- Node features (17 dimensions)
- Binary anomaly labels

### Target Domain: FiGraph

FiGraph is another financial dataset with different feature dimensions (772 dimensions) used for zero-shot evaluation. Feature alignment is applied to enable cross-domain transfer.

FiGraph uses `Label=-1` for background nodes that do not require
classification. SPAD reports AUC/AP on the labeled target-company subset only.
For example, the 2018 snapshot contains 225 anomalies among 3,596 labeled
companies (random AP about 0.0626), not among every background node in the
global heterogeneous graph.

## Usage

### Zero-Shot Detection

Run zero-shot anomaly detection from DGraphFin (source) to FiGraph (target):

```bash
python train_anomaly_gfm.py --mode dynamic --start_year 2018 --end_year 2022 --epochs 50 --batch_size 1 --seed 42 --method_version dipt --run_name spad_dipt_full_seed42
```

**Parameters:**
- `--mode`: Graph mode (use "dynamic")
- `--start_year`: Starting year for training data
- `--end_year`: Ending year for training data
- `--epochs`: Number of training epochs

### Few-Shot Adaptation

Run the paper-facing parameter-efficient target-domain adaptation from the
three retained zero-shot checkpoints:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_few_shot_adapter_curve.ps1
```

This script will:
1. Load the pre-trained checkpoint from zero-shot training
2. Reserve a nested maximum support pool and run 5/10/20-shot settings for seeds 42/43/44
3. Freeze the SPAD backbone and update only a 513-parameter residual adapter
4. Select the update count by five-fold support-only cross-validation
5. Evaluate on the fixed disjoint query set and write paired AUC/AP results

The candidate update range is 1--250 epochs with patience 40; the selected
epoch is the median of the five fold-specific best epochs. Query labels are
used only for final evaluation and never for model or epoch selection.

## Key Components

### Model Architecture

The `AnomalyGFMModel` implements:
- **TemporalConvNet**: Temporal convolution for sequence processing
- **GraphNeuralNetwork**: Graph structure encoding with residual connections
- **HierarchicalFusionNetwork**: Multi-level feature fusion
- **Dual Alignment**: Temporal consistency and single-normal-prototype alignment

### Feature Alignment

To enable cross-domain transfer between DGraphFin (17-dim) and FiGraph (772-dim), the framework implements:
- **SVD-based alignment**: Projects features to a common space
- **Linear projection**: Learns domain-invariant representations

### Loss Functions

- **Normal-Prototype Compactness Loss**: Concentrates normal-node representations around a transferable one-class prototype
- **Temporal Consistency Loss**: Ensures smooth temporal evolution

## Main Contributions

1. **Unified Framework**: First unified framework for zero-shot/few-shot anomaly detection in dynamic graphs

2. **Dual-Alignment Design**: Novel temporal graph representation learning with dual-alignment constraints
   - Temporal consistency constraints
   - Single-prototype population alignment

3. **Theoretical Foundation**: Strong theoretical support with proof of effectiveness

4. **Practical Applicability**: Efficient and practical technical pathway for industrial-level dynamic graph anomaly detection

## Citation

If you use this code or method in your research, please cite our paper once published:

*To be updated after paper publication.*

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

