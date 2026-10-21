# Running NEST

NEST combines normality-guided source learning with evolving-neighborhood target scoring. The current manuscript uses frozen-backbone, support-only score correction for few-shot adaptation, not the alternative residual-adapter experiment.

The paper-compatible implementation is on `main` in [the NEST repository](https://github.com/YuTong-Cai2001/NEST/tree/main). No experimental branch is needed. Run commands from the repository root, named `ProtoDGAD` in the existing local workspace or `NEST` after a fresh clone. See [MANUSCRIPT_VERSION.md](MANUSCRIPT_VERSION.md) for the verified source hashes and archived results.

## 1. Environment and smoke test

Use Python 3.10 or 3.11 and a CUDA-compatible PyTorch installation. Install the remaining requirements and check both prototype modes:

```powershell
conda activate pytorch
where.exe python
pip install -r requirements.txt
python smoke_test.py
```

The test should report `single: OK` and `dual: OK`. When launching PowerShell from an activated Conda prompt, use `-NoProfile` so a shell profile does not replace the active interpreter. The experiment launchers print `Using Python:`; `-PythonExecutable` can select an interpreter explicitly.

For a short code-path check, use a separate run prefix to avoid replacing a full-run checkpoint:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -RunPrefix nest_smoke -Seeds 42 -Epochs 2 -Variants full -SkipFewShot
```

This is a smoke test, not a manuscript result.

## 2. Full zero-shot experiment

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -RunPrefix nest_paper -Seeds 42,43,44 -Epochs 50 -Variants full -SkipFewShot
```

The launcher uses dynamic mode, batch size 1, the last five retained source windows, and the `dipt` method-version tag. The default source objective contains supervised detection, normal-prototype compactness with weight 0.3, and temporal class-mean consistency with weight 0.5. It has no additional residual regularizer. Checkpoint selection minimizes source-domain training loss.

The frozen target score aggregates normalized node-to-neighbor disagreement with chronological weights `1,...,T`, then converts scores to empirical percentile ranks. Neither direct source-prototype distance nor directional prototype transport is the default zero-shot ranking.

The manuscript's retained full-model runs are `spad_ablation_no_reg_seed42`, `spad_ablation_no_reg_seed43`, and `spad_ablation_no_reg_seed44`. These historical run names are unchanged. New runs using the command above are saved separately as `nest_paper_full_seed42`, `nest_paper_full_seed43`, and `nest_paper_full_seed44`.

## 3. Fixed-query few-shot experiment

For the retained full-model checkpoints:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_few_shot_curve.ps1 -CheckpointPrefix spad_ablation_no_reg_seed -Seeds 42,43,44 -Shots 5,10,20
python validate_few_shot_curve.py
```

If the checkpoints were generated with the `nest_paper` prefix above, replace `-CheckpointPrefix spad_ablation_no_reg_seed` with `-CheckpointPrefix nest_paper_full_seed`.

For each seed, the procedure:

1. Reserves 20 anomalous and 100 normal target nodes as the maximum support pool.
2. Fixes the query set to the remaining 5,012 labeled nodes, including 103 anomalies, and recomputes the zero-shot reference on that set.
3. Uses nested support sets containing `k` anomalous and `5k` normal nodes for `k=5,10,20`.
4. Keeps the backbone frozen and selects a bounded logistic or support-prototype correction by five-fold support-only cross-validation. Retaining the zero-shot ranking is also a candidate.
5. Evaluates the selected correction on the same query set. Query labels do not enter fitting or selection.

The minimum anomalous-support count for correction fitting is 10, so the five-shot setting retains the zero-shot anchor. Each seed has its own random support pool, but the query nodes and zero-shot reference are fixed across shot counts within that seed.

The current manuscript reports a mean query ROC-AUC increase from 0.7665 to 0.7700 at 10 shots, using 10 anomalous and 50 normal support nodes. This is distinct from the main zero-shot ROC-AUC of 0.7704, which is evaluated on all 5,132 validly labeled final-snapshot nodes.

The scripts `few_shot_adapter.py`, `run_few_shot_adapter_curve.ps1`, and `summarize_few_shot_adapter.py` concern a separate residual-adapter trial. They are not the source of the current paper's few-shot results.

## 4. Component and scoring analyses

The final-objective component-ablation launcher is:

```powershell
python run_final_component_ablation.py
python summarize_final_component_ablation.py
```

It runs the six retained variants over seeds 42, 43, and 44. Existing final-ablation prefixes remain unchanged so their CSV rows and checkpoints can be matched to the saved experiments. Use `python evaluate_reference_scoring.py --help` for the separate crossed comparison of source compactness and target scoring reference.

## 5. Outputs and evaluation populations

- Source-selected checkpoints and fitted feature aligners: `outputs/<run_name>/`.
- Zero-shot measurements: `results/zero_shot_runs.csv`.
- Fixed-query few-shot measurements: `results/few_shot_curve_runs.csv`.
- Few-shot means and population standard deviations: `results/few_shot_curve_mean_std.csv`.
- Final-objective ablation measurements: `results/component_ablation_final_20261007_runs.csv`.

Metrics use only FiGraph labels exactly equal to 0 or 1; background entities labeled `-1` are excluded from metric calculation, not reclassified as normal. The observed target graph remains available for representation extraction and neighborhood scoring. Excluding the support pool from query metrics does not remove those nodes or their edges from the target graph.

Historical identifiers such as `spad_*`, `ProtoDGAD`, `AnomalyGFMModel`, and `dipt` are retained where they are part of executable paths, saved checkpoint formats, or experiment keys. The method described by the current manuscript and these instructions is NEST.
