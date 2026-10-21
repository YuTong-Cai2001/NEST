"""Rerun final-objective ablations without modifying model/trainer code.

Run sequentially to avoid GPU memory contention. Each run has its own console
log and checkpoint directory. Successful runs are resumable by their CSV row.
"""
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = "spad_final_ablation_20261007"
CSV = ROOT / "results/component_ablation_final_20261007_runs.csv"
OUT = ROOT / "outputs" / PREFIX
VARIANTS = {
    "no_self": ["--lambda_self", "0"],
    "no_proto": ["--lambda_proto", "0"],
    "svd_alignment": ["--alignment_mode", "svd"],
    "no_descriptors": ["--disable_structural_temporal_descriptors"],
    "no_residual": ["--disable_residual_fusion"],
    "dual": ["--prototype_mode", "dual"],
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = ["train_anomaly_gfm.py", "anomaly_gfm_complete_model.py", "data_processor.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    manifest = {"started": datetime.now().isoformat(), "python": sys.executable,
                "seeds": [42, 43, 44], "epochs": 50, "source_sha256": hashes,
                "results_csv": str(CSV), "runs": []}
    for seed in (42, 43, 44):
        for variant, flags in VARIANTS.items():
            name = f"{PREFIX}_{variant}_seed{seed}"
            command = [sys.executable, "-u", "train_anomaly_gfm.py", "--mode", "dynamic",
                       "--epochs", "50", "--batch_size", "1", "--sequence_length", "5",
                       "--start_year", "2018", "--end_year", "2022", "--seed", str(seed),
                       "--method_version", "dipt", "--results_csv", str(CSV),
                       "--run_name", name] + flags
            done = set()
            if CSV.exists():
                with CSV.open(encoding="utf-8-sig", newline="") as f:
                    done = {r["run_name"] for r in csv.DictReader(f)}
            log = OUT / f"{name}.log"
            record = {"run_name": name, "command": command, "log": str(log)}
            if name in done:
                print(f"SKIP successful run {name}", flush=True)
                record["status"] = "already_complete"
            else:
                print(f"START {datetime.now().isoformat()} {name}", flush=True)
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                            stderr=subprocess.STDOUT)
                record["exit_code"] = result.returncode
                record["status"] = "complete" if result.returncode == 0 else "failed"
                print(f"END {datetime.now().isoformat()} {name}: {record['status']}", flush=True)
            manifest["runs"].append(record)
            (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if record["status"] == "failed":
                raise RuntimeError(f"Training failed; inspect {log}")
    assert hashes == {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    print("All 18 final-objective ablation runs completed; training sources unchanged.", flush=True)

if __name__ == "__main__":
    main()
