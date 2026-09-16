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

To produce both the full zero-shot result and its few-shot result, omit
`-SkipFewShot`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Seeds 42 -Epochs 50 -Variants full
```

Then add the four ablations without rerunning the full model:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Seeds 42 -Epochs 50 -Variants no_self,no_proto,no_reg,no_transport,dual -SkipFewShot
```

If time permits, repeat the complete set for three seeds:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Seeds 42,43,44 -Epochs 50
```

The default script runs the full model, three loss ablations, the dual-prototype ablation, few-shot fine-tuning of the full model, and figure generation. `-Variants` lets you stage that work.

The strengthened full model is saved under the `spad_dipt_*` prefix and uses
the `dipt` method-version tag.  This keeps its CSV rows and checkpoints
separate from the older `spad_*` and intermediate `spad_di2_*` runs.

## 4. Outputs

- Checkpoints and fitted feature aligners: `outputs/<run_name>/`
- Zero-shot measurements: `results/zero_shot_runs.csv`
- Few-shot measurements: `results/few_shot_runs.csv`
- Generated paper figures and aggregated tables: `figures/generated/`

The target labels are not used for checkpoint selection. The final target evaluation is performed once after restoring the checkpoint with the lowest source-domain training loss.

As an end-to-end code-path verification, the seed-42 50-epoch run evaluated
with the final directional prototype-transport residual obtained ROC-AUC
`0.7654` and AP `0.0843`.  The score is the coefficient-free difference
between target-transported and source-prototype distances; target labels are
not used to fit it.  This is a single-run engineering check, not a substitute
for the requested multi-seed paper results.
