# Running the strengthened SPAD pipeline

Run every command from the `ProtoDGAD` directory.

## 1. Environment

Create a fresh Python 3.10 or 3.11 environment. Install the PyTorch build that matches the CUDA version on the machine first, then install the remaining dependencies:

```powershell
conda activate pytorch
where python
pip install -r requirements.txt
python smoke_test.py
```

The smoke test must report `single: OK` and `dual: OK` before real training.
On this machine, the existing `pytorch` environment already passed this test
with PyTorch 2.9.0, CUDA 12.8, and CUDA available.

If the launcher is called from a Conda CMD prompt, use `-NoProfile` so a
nested PowerShell profile cannot replace the active interpreter:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -RunPrefix spad_dipt_smoke -Seeds 42 -Epochs 2 -Variants full -SkipFewShot
```

The launcher prints `Using Python: ...` before doing any work. It must point
to the intended Conda environment. You can also pin it explicitly with
`-PythonExecutable "C:\Users\xixi\.conda\envs\pytorch\python.exe"`.

## 2. Quick validation

This checks one full run. It is not a final paper result:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -RunPrefix spad_dipt_smoke -Seeds 42 -Epochs 2 -Variants full -SkipFewShot
```

Keep the separate `spad_dipt_smoke` prefix: using the formal `spad_dipt`
prefix for a two-epoch check would replace the seed-42 formal checkpoint and
CSV row with the short-run result.

## 3. Minimal paper experiments

Fastest useful one-seed zero-shot run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Seeds 42 -Epochs 50 -Variants full -SkipFewShot
```

To produce both the full zero-shot result and its few-shot adaptation, omit
`-SkipFewShot`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Seeds 42 -Epochs 50 -Variants full
```

Then add the six retained paper-facing ablations without rerunning the full model:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Seeds 42 -Epochs 50 -Variants svd_alignment,no_descriptors,no_residual,no_self,no_proto,dual -SkipFewShot
```

If time permits, repeat the complete set for three seeds:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Seeds 42,43,44 -Epochs 50
```

If the three zero-shot checkpoints already exist (as in the retained
`spad_ablation_no_reg_seed42/43/44` formal runs), run the paper-facing
parameter-efficient 5/10/20-shot experiment and its mean/std aggregation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_few_shot_adapter_curve.ps1
```

This freezes the SPAD backbone, trains only a 513-parameter residual prototype
adapter and selects the number of updates using five-fold support-only cross
validation. The search covers epochs 1--250 with patience 40; there is no
hard-coded one-epoch limit. The fixed query set excludes the maximum support
pool, so 5/10/20-shot results are directly paired within each seed.

The default script runs the full model, the six retained component/design ablations, few-shot fine-tuning of the full model, and figure generation. `-Variants` lets you stage that work. The `no_transport` and `no_local` switches remain available for diagnostic scoring studies but are not part of the paper-facing component table.

The strengthened full model is saved under the `spad_dipt_*` prefix and uses
the `dipt` method-version tag.  This keeps its CSV rows and checkpoints
separate from the older `spad_*` and intermediate `spad_di2_*` runs.

## 4. Outputs

- Checkpoints and fitted feature aligners: `outputs/<run_name>/`
- Zero-shot measurements: `results/zero_shot_runs.csv`
- Few-shot measurements: `results/few_shot_runs.csv` (new full runs) or
  `results/few_shot_final_runs.csv` (retained formal checkpoints)
- Few-shot three-seed summary: `results/few_shot_mean_std.csv`
- Fixed-query shot-curve measurements: `results/few_shot_curve_runs.csv`
- Fixed-query shot-curve summary: `results/few_shot_curve_mean_std.csv`
- Parameter-efficient adapter measurements: `results/few_shot_adapter_trial.csv`
- Parameter-efficient adapter summary: `results/few_shot_adapter_mean_std.csv`
- Generated paper figures and aggregated tables: `figures/generated/`

The target labels are not used for checkpoint selection. The final target evaluation is performed once after restoring the checkpoint with the lowest source-domain training loss.

Evaluation uses only nodes whose FiGraph labels are exactly 0 or 1. Nodes with
`Label=-1` are unlabeled background entities and are excluded; interpreting
them as normal can greatly inflate AUC and produces numbers that are not
comparable with the corrected protocol. Every formal run prints the number of
excluded unlabeled nodes and writes `evaluation_protocol=labeled_target_nodes_only`
to the zero-shot CSV.

As an end-to-end code-path verification, the seed-42 50-epoch run evaluated
on the final 2022 snapshot with the final SPAD score obtained ROC-AUC `0.7882`
and AP `0.1084`.  Source
prototype and temporal objectives learn the transferable representation; the
target score aggregates dynamic neighbourhood disagreement with fixed linear
recency weights and empirical percentile-rank calibration.  Target labels and
target-fitted fusion weights are not used to compute it.  This is a single-run
engineering check, not a substitute for the requested multi-seed paper results.
