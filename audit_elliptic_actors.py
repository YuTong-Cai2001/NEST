"""Inspect official Elliptic++ files without fitting or changing SPAD.

Downloads are checked against Git LFS SHA-256 pointers from a pinned upstream
commit. Raw CSVs remain unchanged. Analysis outputs are JSON, not model input.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
COMMIT = "08fe6aded83afb97bf5a79a71130f542ca783c2e"
FILES = {
    "wallets_features.csv": "Actors Dataset/wallets_features.csv",
    "wallets_classes.csv": "Actors Dataset/wallets_classes.csv",
    "AddrTx_edgelist.csv": "Actors Dataset/AddrTx_edgelist.csv",
    "TxAddr_edgelist.csv": "Actors Dataset/TxAddr_edgelist.csv",
    "txs_features.csv": "Transactions Dataset/txs_features.csv",
}
CORE = ("anomaly_gfm_complete_model.py", "train_anomaly_gfm.py", "data_processor.py",
        "sparse_graph_utils.py", "model_adapter.py", "few_shot_finetune.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "SPAD-Elliptic-Actors-Audit", **(headers or {})}), timeout=60)


def download(name, raw_dir):
    relative = urllib.parse.quote(FILES[name], safe="/")
    pointer_url = f"https://raw.githubusercontent.com/git-disl/EllipticPlusPlus/{COMMIT}/{relative}"
    with request(pointer_url) as response:
        pointer = response.read().decode("utf-8")
    oid = re.search(r"oid sha256:([0-9a-f]{64})", pointer)
    size = re.search(r"size (\d+)", pointer)
    if not oid or not size:
        raise ValueError(f"Not a Git LFS pointer: {name}")
    expected_sha, expected_size = oid[1], int(size[1])
    destination = raw_dir / name
    url = f"https://media.githubusercontent.com/media/git-disl/EllipticPlusPlus/{COMMIT}/{relative}?download=1"
    if destination.exists():
        if destination.stat().st_size != expected_size or sha(destination) != expected_sha:
            raise ValueError(f"Existing raw file does not match upstream: {destination}")
        print(f"VERIFIED existing {name}", flush=True)
    else:
        partial = raw_dir / (name + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError(f"Unexpected partial length: {partial}")
        if offset < expected_size:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            print(f"DOWNLOAD {name}: {offset}/{expected_size}", flush=True)
            with request(url, headers) as response:
                if offset:
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server did not honor resume offset")
                elif response.status != 200:
                    raise ValueError(f"Expected complete download, got status {response.status}")
                with partial.open("ab" if offset else "wb") as handle:
                    last_report = time.monotonic()
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        offset += len(block)
                        if time.monotonic() - last_report > 15:
                            print(f"PROGRESS {name}: {offset}/{expected_size}", flush=True)
                            last_report = time.monotonic()
        if partial.stat().st_size != expected_size or sha(partial) != expected_sha:
            raise ValueError(f"Download integrity check failed: {name}")
        partial.replace(destination)
        print(f"VERIFIED download {name}", flush=True)
    return {"name": name, "upstream_commit": COMMIT, "pointer_url": pointer_url,
            "download_url": url, "bytes": expected_size, "sha256": expected_sha,
            "path": str(destination)}


def features_audit(raw_dir):
    print("AUDIT reading wallet observations and labels", flush=True)
    f = pd.read_csv(raw_dir / "wallets_features.csv", dtype={"address": str})
    classes = pd.read_csv(raw_dir / "wallets_classes.csv", dtype={"address": str})
    assert classes["address"].is_unique
    assert set(classes["class"].unique()) <= {1, 2, 3}
    assert f["Time step"].notna().all()
    assert np.equal(f["Time step"], f["Time step"].astype(int)).all()
    assert f["Time step"].between(1, 49).all()
    labels = classes.set_index("address")["class"]
    f["_audit_class"] = f["address"].map(labels)
    assert f["_audit_class"].notna().all()
    feature_cols = [c for c in f.columns if c not in ("address", "Time step", "_audit_class")]
    rows_per_address = f.groupby("address", sort=False).size()
    steps_per_address = f.groupby("address", sort=False)["Time step"].nunique()
    repeated_ids = steps_per_address[steps_per_address > 1].index
    repeated = f[f["address"].isin(repeated_ids)]
    changes = (repeated.groupby("address", sort=False)[feature_cols].nunique(dropna=False) > 1).sum()
    per_time = []
    for t in range(1, 50):
        part = f[f["Time step"] == t]
        per_time.append({"time_step": t, "rows": len(part), "unique_addresses": part["address"].nunique(),
                         "class_counts": {str(k): int(v) for k, v in part["_audit_class"].value_counts().items()}})
    result = {
        "feature_rows": len(f), "unique_feature_addresses": f["address"].nunique(),
        "class_rows": len(classes), "class_counts": {str(k): int(v) for k, v in classes["class"].value_counts().items()},
        "feature_columns": list(feature_cols), "usable_numeric_field_count_before_temporal_review": len(feature_cols),
        "time_step_count": f["Time step"].nunique(),
        "duplicate_address_time_rows": int(f.duplicated(["address", "Time step"]).sum()),
        "addresses_with_multiple_rows": int((rows_per_address > 1).sum()),
        "addresses_in_multiple_observed_steps": len(repeated_ids),
        "multi_step_addresses_by_class": {str(k): int(v) for k, v in labels.reindex(repeated_ids).value_counts().items()},
        "distinct_step_histogram": {str(k): int(v) for k, v in steps_per_address.value_counts().sort_index().items()},
        "nonfinite_counts": {c: int((~np.isfinite(f[c].to_numpy(dtype=float))).sum()) for c in feature_cols},
        "multi_step_addresses_with_feature_changes": {c: int(changes.get(c, 0)) for c in feature_cols},
        "per_time_step": per_time,
        "global_history_fields_requiring_review": [c for c in feature_cols if c in (
            "last_block_appeared_in", "lifetime_in_blocks", "num_timesteps_appeared_in", "total_txs")],
        "label_column_in_feature_file": any(c.lower() in ("class", "label") for c in feature_cols),
    }
    return result


def graph_audit(raw_dir):
    print("AUDIT reading transaction-time mapping and address incidence", flush=True)
    tx = pd.read_csv(raw_dir / "txs_features.csv", usecols=["txId", "Time step"])
    assert tx["txId"].is_unique
    time_by_tx = tx.set_index("txId")["Time step"]
    incoming = pd.read_csv(raw_dir / "AddrTx_edgelist.csv", dtype={"input_address": str})
    outgoing = pd.read_csv(raw_dir / "TxAddr_edgelist.csv", dtype={"output_address": str})
    for frame in (incoming, outgoing):
        frame["time"] = frame["txId"].map(time_by_tx)
        if frame["time"].isna().any():
            raise ValueError("Address incidence refers to transactions absent from the published time mapping")
    observations = pd.concat([
        incoming[["input_address", "time"]].rename(columns={"input_address": "address"}),
        outgoing[["output_address", "time"]].rename(columns={"output_address": "address"}),
    ], ignore_index=True).drop_duplicates()
    classes = pd.read_csv(raw_dir / "wallets_classes.csv", dtype={"address": str}).set_index("address")["class"]
    observations["class"] = observations["address"].map(classes)
    if observations["class"].isna().any():
        raise ValueError("Missing official label for an observed address")
    steps = observations.groupby("address", sort=False)["time"].nunique()
    ins = incoming.groupby("txId").size()
    outs = outgoing.groupby("txId").size()
    products = ins.mul(outs, fill_value=0).astype(np.int64)
    per_time = []
    for t in range(1, 50):
        current = observations[observations["time"] == t]
        per_time.append({"time_step": t, "active_addresses": len(current),
                         "class_counts": {str(k): int(v) for k, v in current["class"].value_counts().items()},
                         "address_pair_edge_occurrences": int(products.reindex(tx.loc[tx["Time step"] == t, "txId"], fill_value=0).sum())})
    # Review all rolling windows, never select windows using detection scores.
    rolling = []
    for end in range(5, 50):
        window = observations[observations["time"].between(end - 4, end)]
        nsteps = window.groupby("address", sort=False)["time"].nunique()
        latest = observations[observations["time"] == end].copy()
        latest["observed_steps_in_window"] = latest["address"].map(nsteps)
        rolling.append({"end": end, "start": end - 4, "universe": len(nsteps),
                        "latest_labeled_nodes": int(latest["class"].isin((1, 2)).sum()),
                        "latest_illicit_nodes": int((latest["class"] == 1).sum()),
                        "latest_labeled_with_history": int((latest["class"].isin((1, 2)) & (latest["observed_steps_in_window"] >= 2)).sum()),
                        "latest_illicit_with_history": int(((latest["class"] == 1) & (latest["observed_steps_in_window"] >= 2)).sum())})
    return {
        "transactions_with_time": len(tx), "incoming_relations": len(incoming), "outgoing_relations": len(outgoing),
        "address_time_occurrences": len(observations), "unique_observed_addresses": len(steps),
        "distinct_step_histogram": {str(k): int(v) for k, v in steps.value_counts().sort_index().items()},
        "illicit_distinct_step_histogram": {str(k): int(v) for k, v in steps[classes.reindex(steps.index).to_numpy() == 1].value_counts().sort_index().items()},
        "reconstructible_address_pair_occurrences": int(products.sum()), "per_time_step": per_time,
        "all_rolling_five_step_windows": rolling,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("features", "graph"), default="features")
    args = parser.parse_args()
    raw_dir = ROOT / "data/EllipticPlusPlus"
    out = ROOT / "outputs/elliptic_actors_audit_20261020"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    before = {name: sha(ROOT / name) for name in CORE}
    names = list(FILES) if args.stage == "graph" else list(FILES)[:2]
    manifest_path = out / "download_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(lambda n: download(n, raw_dir), names):
            previous[record["name"]] = record
            save_json(manifest_path, previous)
    report = {"upstream_commit": COMMIT, "stage": args.stage,
              "features": features_audit(raw_dir), "core_sha256_before": before,
              "training_performed": False, "target_metrics_computed": False}
    if args.stage == "graph":
        report["graph"] = graph_audit(raw_dir)
    report["core_sha256_after"] = {name: sha(ROOT / name) for name in CORE}
    assert report["core_sha256_before"] == report["core_sha256_after"]
    report["status"] = "complete"
    save_json(out / f"{args.stage}_audit.json", report)
    print("AUDIT_COMPLETE " + str(out / f"{args.stage}_audit.json"), flush=True)
    print(json.dumps({k: report["features"][k] for k in (
        "feature_rows", "unique_feature_addresses", "time_step_count",
        "addresses_in_multiple_observed_steps", "multi_step_addresses_by_class")}), flush=True)


if __name__ == "__main__":
    main()
