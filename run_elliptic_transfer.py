"""Fixed DGraphFin -> Elliptic++ Actors main experiment (full SPAD only).

The existing model, descriptors, supervised source epoch and default inference
path are reused unchanged. Released wallet attributes are retrospective static
summaries, not causally available per-snapshot measurements. No target labels
are used for representation fitting, window selection or checkpoint selection.
"""

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_gfm_complete_model import AnomalyGFMModel
from data_processor import AnomalyGFMDataProcessor
from evaluate_reference_scoring import (
    file_sha256, model_state_sha256, ranking_metrics, score_metrics,
    stable_percentile,
)
from run_reverse_transfer import (
    atomic_json, class_counts, core_hashes, input_signature,
    normalize_target_graph, restore_rng, rng_state,
)
from sparse_graph_utils import SparseGraphProcessor
from train_anomaly_gfm import (
    AnomalyGFMTrainer, TemporalGraphDataset, set_global_seed, sparse_collate_fn,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs/spad_dgraphfin_to_elliptic_actors_20261020"
PROVENANCE = ROOT.parent / "outputs/spad_final_ablation_20261007/component_results.json"
SEEDS = (42, 43, 44)
STEPS = (45, 46, 47, 48, 49)
OFFICIAL_SHA256 = {
    "wallets_features.csv": "317daca2810c355ddfdb8c0dab34cf11d1aa90567fe975090c7e5a901386eb77",
    "wallets_classes.csv": "4e5132c99f941666bf1fefd4100a1428d339c9252ec6987909e1adf8eac902f9",
    "AddrTx_edgelist.csv": "f5f903f752387f66a1bccaeff54e293b2e8470fcddf5eb56b88aa06fd23a8f3b",
    "TxAddr_edgelist.csv": "9f5afbdde7bc3d91fb7a4655be55799d6504cd0063ae55a0753a5a41189932b8",
    "txs_features.csv": "2db326ec8ddb68f1d810c1834e1ff62e0a8300378f0984a1e3b2ca82a439821b",
}


def fixed_protocol():
    return {
        "schema": 1, "direction": "DGraphFin_to_EllipticPlusPlus_Actors",
        "seeds": list(SEEDS), "variant": "full", "epochs": 50,
        "backup_commit": "b7f364b", "target_time_steps": list(STEPS),
        "time_step_selection": "final five official steps, fixed before scoring",
        "target_universe": "lexically ordered union of all incident wallets in steps 45-49; no subsampling",
        "graph": "input wallet -> transaction -> output wallet; preserve direction and pair multiplicities in CSR weights; existing diagonal and D^-1/2 A D^-1/2 preprocessing",
        "target_attributes": "55 released static numeric summaries, one vector per wallet; exclude address, Time step and class",
        "attribute_presence": "use the released vector only when wallet participates in that snapshot; zero absent rows",
        "feature_availability": "retrospective observed-sequence detection, not online forecasting; published summaries can include full-history information",
        "target_transform": "fit existing invariant transform on step 45 nonzero rows without labels, freeze for steps 46-49",
        "target_structural_descriptors": "existing eight descriptors on the fixed five-step graph sequence; first-step initialization unchanged",
        "source_preparation": "unchanged DGraphFin loader and source sampling: max 100000 nodes, all source anomalies plus highest-degree source normals; max 20 sampled seven-unit windows; train on final five",
        "source_transform": "same deterministic invariant source transform and descriptors as FiGraph main experiment; verify against saved source scalers",
        "source_objective": "BCE + 0.3 prototype compactness + 0.5 temporal consistency; no additional regularizer",
        "selection": "50 epochs, minimum total source training loss; all three training runs finish before target evaluation",
        "evaluation_population": "all step-49 participating wallets with official class exactly 1 or 2",
        "label_mapping": {"1": 1, "2": 0, "3": -1},
        "unknown_nodes": "retained in graph computation and full-universe rank, excluded from metrics",
        "inference": "unchanged frozen SPAD default anomaly_scores; recency weights 1..5 and full-universe empirical percentile",
        "metrics": "ROC-AUC and average precision; population standard deviation ddof=0 across seeds; no score sign tuning",
        "raw_score_metrics": "also record raw recency-disagreement metrics to expose ordinal-tie or sigmoid effects, not for score selection",
        "upstream_commit": "08fe6aded83afb97bf5a79a71130f542ca783c2e",
    }


def join_incidence(incoming, outgoing, transactions, steps=STEPS):
    """Construct presence and directed pair occurrences without label inputs."""
    if not transactions["txId"].is_unique:
        raise ValueError("Transaction-time map has duplicate transaction IDs")
    times = transactions.set_index("txId")["Time step"]
    left = incoming.copy()
    right = outgoing.copy()
    for frame in (left, right):
        mapped = frame["txId"].map(times)
        if mapped.isna().any():
            raise ValueError("Incidence references an unknown transaction")
        frame["time"] = mapped.astype(np.int64)
    left = left[left["time"].isin(steps)]
    right = right[right["time"].isin(steps)]
    presence = pd.concat([
        left[["input_address", "time"]].rename(columns={"input_address": "address"}),
        right[["output_address", "time"]].rename(columns={"output_address": "address"}),
    ], ignore_index=True).drop_duplicates()
    pairs = left.merge(right[["txId", "output_address"]], on="txId", how="inner",
                       validate="many_to_many")
    return presence, pairs[["input_address", "output_address", "time"]]


def wallet_table(frame, ids):
    columns = [c for c in frame if c not in ("address", "Time step")]
    if any(c.lower() in ("class", "label") for c in columns):
        raise ValueError("A class/label field entered the wallet feature table")
    # Duplicate rows reflect transactions, not independent wallet-time samples.
    selected = frame[frame["address"].isin(ids)]
    values = selected.groupby("address", sort=False)[columns].nunique(dropna=False)
    if not (values <= 1).all().all():
        raise ValueError("Expected the verified static attributes per wallet")
    table = selected.drop_duplicates("address").set_index("address").reindex(ids)
    if table[columns].isna().any().any():
        raise ValueError("A modeled wallet lacks released numeric features")
    numeric = table[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite published wallet features")
    return numeric, columns


def map_target_labels(classes, ids, latest_presence):
    if not classes["address"].is_unique:
        raise ValueError("Duplicate wallet class records")
    official = classes.set_index("address")["class"].reindex(ids)
    if official.isna().any() or not official.isin((1, 2, 3)).all():
        raise ValueError("Unexpected/missing official wallet class")
    labels = official.map({1: 1, 2: 0, 3: -1}).to_numpy(dtype=np.int64)
    active = np.isin(ids, np.asarray(latest_presence))
    mask = active & np.isin(labels, (0, 1))
    return labels, mask, active


def check_source_transform(processor):
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    references = [r for r in provenance["checks"]
                  if r["run_name"].startswith("spad_ablation_no_reg_seed")]
    assert len(references) == 3
    for reference in references:
        path = Path(reference["checkpoint"]).parent.parent / "feature_aligner.pkl"
        with path.open("rb") as handle:
            old = pickle.load(handle)
        for key in ("source_scaler", "source_descriptor_scaler"):
            for field in ("mean_", "var_", "scale_"):
                np.testing.assert_allclose(getattr(getattr(processor, key), field),
                                           getattr(old[key], field), rtol=0, atol=0)
    return [r["run_name"] for r in references]


def prepare(out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Existing experiment output: use --resume, never overwrite")
    out.mkdir(parents=True, exist_ok=True)
    protocol = fixed_protocol()
    protocol.update({"core_sha256": core_hashes(), "entry_point_sha256": file_sha256(__file__),
                     "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    atomic_json(out / "protocol.json", protocol)
    raw_dir = ROOT / "data/EllipticPlusPlus"
    raw_hashes = {str((raw_dir / n).relative_to(ROOT)): file_sha256(raw_dir / n)
                  for n in OFFICIAL_SHA256}
    assert all(raw_hashes[str((raw_dir / n).relative_to(ROOT))] == expected
               for n, expected in OFFICIAL_SHA256.items()), "Official input hash mismatch"
    for name in ("dgraphfin.npz", "dgraphfinv2_edge_timestamp.npy"):
        path = ROOT / "data/DGraphFin2" / name
        raw_hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    print("PREPARE official inputs verified; protocol frozen", flush=True)
    tx = pd.read_csv(raw_dir / "txs_features.csv", usecols=["txId", "Time step"])
    incoming = pd.read_csv(raw_dir / "AddrTx_edgelist.csv")
    outgoing = pd.read_csv(raw_dir / "TxAddr_edgelist.csv")
    presence, pairs = join_incidence(incoming, outgoing, tx)
    ids = np.asarray(sorted(presence["address"].unique()), dtype=str)
    index = pd.Index(ids)
    frame = pd.read_csv(raw_dir / "wallets_features.csv")
    raw_attributes, attribute_columns = wallet_table(frame, ids)
    assert len(attribute_columns) == 55
    target_graphs, masks, graph_stats = [], [], []
    mapping = np.arange(len(ids), dtype=np.int64)
    for step in STEPS:
        current = pairs[pairs["time"] == step]
        edges = np.column_stack([index.get_indexer(current["input_address"]),
                                 index.get_indexer(current["output_address"])])
        if (edges < 0).any():
            raise ValueError("Lost target wallet endpoint")
        graph, count = normalize_target_graph(edges, mapping, len(ids))
        assert count == len(current), "Truncated target neighborhoods"
        target_graphs.append(graph)
        masks.append(np.isin(ids, presence.loc[presence["time"] == step, "address"]))
        graph_stats.append({"time_step": step, "raw_pair_occurrences": count,
                            "raw_self_pair_occurrences": int((edges[:, 0] == edges[:, 1]).sum()),
                            "normalized_csr_nnz_including_diagonal": int(graph.nnz),
                            "active_wallets": int(masks[-1].sum())})
    # IDs, graph construction and presence are fixed before classes are read.
    classes = pd.read_csv(raw_dir / "wallets_classes.csv")
    labels, evaluation_mask, latest_active = map_target_labels(
        classes, ids, presence.loc[presence["time"] == STEPS[-1], "address"])
    assert len(ids) == 93555 and latest_active.sum() == 12199
    assert class_counts(labels[evaluation_mask]) == {"0": 3813, "1": 652}
    del frame, tx, incoming, outgoing, presence, pairs, classes
    gc.collect()
    processor = AnomalyGFMDataProcessor({
        "alignment_mode": "invariant", "target_dim": 16, "seed": 42,
        "dgraphfin_path": str(ROOT / "data/DGraphFin2"),
        "cache_dir": str(ROOT / "cache"), "enable_cache": True,
    })
    print("PREPARE original DGraphFin source loader and sampling", flush=True)
    source_raw = processor.load_dgraphfin_data()
    snapshots = processor.create_temporal_snapshots(source_raw, window_size=7)
    sequence = processor.prepare_temporal_sequences(source_raw, snapshots)
    source_ids = np.asarray(sequence["selected_node_ids"], dtype=np.int64)
    source_labels = np.asarray(source_raw["labels"])[source_ids].astype(np.int64)
    source_graphs = sequence["temporal_adjacencies"][0]
    source_reference = sequence["temporal_sequences"][0][0]
    first_target = np.where(masks[0][:, None], raw_attributes, 0).astype(np.float32)
    source_aligned, _ = processor.fit_feature_aligner(source_reference, first_target)
    transform_checks = check_source_transform(processor)
    source_features = processor.append_structural_temporal_features(
        [source_aligned] * len(source_graphs), source_graphs)
    # Static published attributes are zeroed for absent nodes, as in FiGraph.
    aligned_static = processor.transform_target_features(raw_attributes)
    target_aligned = [np.where(mask[:, None], aligned_static, 0).astype(np.float32)
                      for mask in masks]
    target_features = processor.append_structural_temporal_features(target_aligned, target_graphs)
    processor.save_feature_aligner(str(out / "feature_aligner.pkl"))
    source = {"features": source_features[-5:], "graphs": source_graphs[-5:],
              "labels": source_labels, "ids": source_ids}
    target = {"features": target_features, "graphs": target_graphs, "labels": labels,
              "ids": ids, "evaluation_mask": evaluation_mask,
              "presence_masks": masks}
    for domain in (source, target):
        for features in domain["features"]:
            assert features.shape == (len(domain["ids"]), 24)
            assert np.isfinite(features).all()
        for graph in domain["graphs"]:
            assert graph.shape == (len(domain["ids"]),) * 2
            assert np.isfinite(graph.data).all()
    data = {"source": source, "target": target}
    with (out / "prepared_inputs.pkl.tmp").open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "prepared_inputs.pkl.tmp").replace(out / "prepared_inputs.pkl")
    np.savez_compressed(out / "target_population.npz", wallet_addresses=ids,
                        labels=labels, evaluation_mask=evaluation_mask,
                        latest_active=latest_active, presence_masks=np.stack(masks))
    stats = {
        "status": "prepared", "raw_sha256": raw_hashes,
        "protocol_sha256": file_sha256(out / "protocol.json"),
        "prepared_inputs_sha256": file_sha256(out / "prepared_inputs.pkl"),
        "feature_aligner_sha256": file_sha256(out / "feature_aligner.pkl"),
        "population_sha256": file_sha256(out / "target_population.npz"),
        "source_signature": input_signature(source), "target_signature": input_signature(target),
        "source_num_nodes": len(source_ids), "source_class_counts": class_counts(source_labels),
        "source_transform_exactly_matches": transform_checks,
        "source_graph_nnz": [int(g.nnz) for g in source["graphs"]],
        "target_num_nodes": len(ids), "target_class_counts": class_counts(labels),
        "target_evaluation_class_counts": class_counts(labels[evaluation_mask]),
        "target_anomaly_prevalence": float(labels[evaluation_mask].mean()),
        "target_attribute_columns": attribute_columns, "target_graphs": graph_stats,
        "target_latest_labeled_with_multistep_presence": int((evaluation_mask & (np.stack(masks).sum(0) >= 2)).sum()),
        "core_unchanged": core_hashes() == protocol["core_sha256"],
    }
    assert stats["core_unchanged"]
    atomic_json(out / "preparation.json", stats)
    print("PREPARED " + json.dumps({k: stats[k] for k in (
        "source_num_nodes", "source_class_counts", "target_num_nodes",
        "target_evaluation_class_counts", "target_anomaly_prevalence")}), flush=True)
    return data, stats


def verify_prepared(out):
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    stats = json.loads((out / "preparation.json").read_text(encoding="utf-8"))
    assert all(protocol[key] == value for key, value in fixed_protocol().items())
    assert protocol["entry_point_sha256"] == file_sha256(__file__)
    assert protocol["core_sha256"] == core_hashes()
    for name, key in (("protocol.json", "protocol_sha256"),
                      ("prepared_inputs.pkl", "prepared_inputs_sha256"),
                      ("feature_aligner.pkl", "feature_aligner_sha256"),
                      ("target_population.npz", "population_sha256")):
        assert file_sha256(out / name) == stats[key], name
    for name, expected in stats["raw_sha256"].items():
        assert file_sha256(ROOT / name) == expected, name
    with (out / "prepared_inputs.pkl").open("rb") as handle:
        data = pickle.load(handle)
    for domain in ("source", "target"):
        assert input_signature(data[domain]) == stats[f"{domain}_signature"]
    return data, stats


def training_config(out, seed):
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    reference = next(r for r in provenance["checks"]
                     if r["run_name"] == f"spad_ablation_no_reg_seed{seed}")
    config = dict(reference["config"])
    assert config["num_epochs"] == 50 and config["lambda_proto"] == .3
    assert config["lambda_self"] == .5 and config.get("lambda_reg", 0) == 0
    config.update({"seed": seed, "run_name": f"dgraphfin_to_elliptic_actors_seed{seed}",
                   "output_dir": str(out / f"seed{seed}"),
                   "source_dataset": "DGraphFin", "target_dataset": "EllipticPlusPlus-Actors",
                   "results_csv": str(out / "unused_legacy_path.csv")})
    return config


def train_one(source, out, seed):
    config = training_config(out, seed)
    run_dir = Path(config["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    complete = run_dir / "training_complete.json"
    if complete.exists():
        record = json.loads(complete.read_text(encoding="utf-8"))
        assert record["config"] == config and record["epochs_completed"] == 50
        assert file_sha256(record["checkpoint"]) == record["checkpoint_sha256"]
        return record
    # Only source arrays enter this function; it has no target data argument.
    with (run_dir / "training.log").open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            trainer = AnomalyGFMTrainer(config)
            trainer.feature_dim = 24
            trainer.feature_aligner_path = str(out / "feature_aligner.pkl")
            dataset = TemporalGraphDataset({"temporal_sequences": [source["features"]],
                "temporal_adjacencies": [source["graphs"]], "valid_timestamps": [0]},
                5, labels=source["labels"])
            trainer.train_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                collate_fn=sparse_collate_fn, num_workers=0)
            trainer.create_model()
            initial = model_state_sha256(trainer.model)
            history, best, best_epoch, start = [], float("inf"), -1, 0
            last = run_dir / "last_training_state.pth"
            if last.exists():
                state = torch.load(last, map_location="cpu", weights_only=False)
                assert state["config"] == config and state["initial_state_sha256"] == initial
                trainer.model.load_state_dict(state["model"])
                trainer.optimizer.load_state_dict(state["optimizer"])
                trainer.scheduler.load_state_dict(state["scheduler"])
                history, best, best_epoch, start = (state[k] for k in (
                    "history", "best", "best_epoch", "next_epoch"))
                restore_rng(state["rng"])
                del state
            for epoch in range(start, 50):
                started = time.perf_counter()
                metrics = trainer.train_epoch_sparse()
                loss = float(metrics["total_loss"])
                assert np.isfinite(loss)
                expected = metrics["anomaly_loss"] + .3 * metrics["prototype_loss"] + .5 * metrics["consistency_loss"]
                assert abs(loss - expected) < 2e-6, "Unexpected source objective"
                if loss < best:
                    best, best_epoch = loss, epoch
                    trainer.save_model("best_source_loss.pth")
                trainer.scheduler.step(loss)
                history.append({"epoch": epoch, "seconds": time.perf_counter() - started,
                    **{k: float(v) for k, v in metrics.items() if np.isscalar(v)}})
                state = {"config": config, "model": trainer.model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "scheduler": trainer.scheduler.state_dict(), "rng": rng_state(),
                    "history": history, "best": best, "best_epoch": best_epoch,
                    "next_epoch": epoch + 1, "initial_state_sha256": initial}
                temporary = last.with_suffix(".pth.tmp")
                torch.save(state, temporary)
                temporary.replace(last)
                atomic_json(run_dir / "source_training_history.json", history)
                print("SOURCE_EPOCH " + json.dumps(history[-1]), flush=True)
                if (epoch + 1) % 10 == 0:
                    sys.__stdout__.write(f"SOURCE_PROGRESS seed={seed} epoch={epoch + 1}/50 best_loss={best:.6f}\n")
                    sys.__stdout__.flush()
            checkpoint = run_dir / "checkpoints/best_source_loss.pth"
            record = {"seed": seed, "config": config, "epochs_completed": len(history),
                "best_source_loss": best, "best_epoch_zero_based": best_epoch,
                "initial_state_sha256": initial, "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "training_seconds": sum(r["seconds"] for r in history),
                "resolved_model_config": trainer.resolved_model_config}
            assert record["epochs_completed"] == 50
            atomic_json(complete, record)
            del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"TRAINED seed={seed} best_epoch={best_epoch + 1} source_loss={best:.6f}", flush=True)
    return record


def evaluate_one(target, out, trained):
    seed = trained["seed"]
    run_dir = out / f"seed{seed}"
    result_path = run_dir / "evaluation.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["checkpoint_sha256"] == trained["checkpoint_sha256"]
        assert file_sha256(result["scores_path"]) == result["scores_sha256"]
        return result
    set_global_seed(seed)
    assert file_sha256(trained["checkpoint"]) == trained["checkpoint_sha256"]
    saved = torch.load(trained["checkpoint"], map_location="cpu", weights_only=False)
    model = AnomalyGFMModel(saved["resolved_model_config"])
    assert all(k.startswith("base_model.") for k in saved["model_state_dict"])
    model.load_state_dict({k[len("base_model."):]: v for k, v in saved["model_state_dict"].items()}, strict=True)
    model.to("cuda").requires_grad_(False).eval()
    before = model_state_sha256(model)
    tensor = torch.as_tensor(np.stack(target["features"]), device="cuda").unsqueeze(0)
    processor = SparseGraphProcessor(device="cuda")
    graphs = []
    for graph in target["graphs"]:
        edge_index, edge_attr = processor.sparse_to_pyg_format(graph)
        graphs.append({"edge_index": edge_index, "edge_attr": edge_attr,
                       "num_nodes": len(target["ids"])})
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = model(tensor, graphs, tensor[:, -1])
    assert before == model_state_sha256(model)
    score = outputs["anomaly_scores"].cpu().numpy().reshape(-1)
    raw = outputs["recency_neighbor_inconsistency"].cpu().numpy().reshape(-1)
    logits = outputs["anomaly_logits"].cpu().numpy().reshape(-1)
    assert len(score) == len(target["ids"]) and np.isfinite(score).all()
    # Validate that the actual model took the default percentile-local branch.
    scale = (torch.nn.functional.softplus(model.score_scale_raw) + 1e-6).item()
    expected_logits = scale * (stable_percentile(raw) - .5)
    np.testing.assert_allclose(logits, expected_logits, rtol=1e-5, atol=2e-5)
    mask, labels = target["evaluation_mask"], target["labels"]
    metrics = ranking_metrics(labels, score, mask)
    np.savez_compressed(run_dir / "target_scores.npz", anomaly_score=score,
                        anomaly_logit=logits, raw_recency_disagreement=raw)
    result = {"seed": seed, "checkpoint": trained["checkpoint"],
        "checkpoint_sha256": trained["checkpoint_sha256"], "metrics": metrics,
        "raw_disagreement_metrics_and_ties": score_metrics(labels, raw, mask),
        "modeled_nodes": len(labels), "evaluated_nodes": int(mask.sum()),
        "anomalous_evaluated_nodes": int((labels[mask] == 1).sum()),
        "scores_path": str(run_dir / "target_scores.npz"),
        "scores_sha256": file_sha256(run_dir / "target_scores.npz"),
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "model_state_unchanged": True, "default_local_percentile_branch_verified": True}
    atomic_json(result_path, result)
    print("RESULT " + json.dumps(result), flush=True)
    del model, saved, outputs, tensor, graphs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    out = args.output_dir.resolve()
    data, stats = verify_prepared(out) if args.resume else prepare(out)
    if args.prepare_only:
        return
    manifest_path = out / "main_experiment_results.json"
    manifest = {"status": "training", "direction": fixed_protocol()["direction"],
        "seeds": list(SEEDS), "protocol_sha256": stats["protocol_sha256"],
        "core_sha256_before": core_hashes(), "training": [], "runs": [],
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0)}}
    atomic_json(manifest_path, manifest)
    try:
        for seed in SEEDS:
            print(f"TRAIN_START seed={seed}", flush=True)
            manifest["training"].append(train_one(data["source"], out, seed))
            atomic_json(manifest_path, manifest)
        manifest["status"] = "evaluating"
        atomic_json(manifest_path, manifest)
        for trained in manifest["training"]:
            manifest["runs"].append(evaluate_one(data["target"], out, trained))
            atomic_json(manifest_path, manifest)
        manifest["summary"] = {"ddof": 0, **{
            metric: {"mean": float(np.mean([r["metrics"][metric] for r in manifest["runs"]])),
                     "std": float(np.std([r["metrics"][metric] for r in manifest["runs"]], ddof=0)),
                     "values": [r["metrics"][metric] for r in manifest["runs"]]}
            for metric in ("auc", "ap")}}
        manifest["core_sha256_after"] = core_hashes()
        assert manifest["core_sha256_before"] == manifest["core_sha256_after"]
        assert all(file_sha256(r["checkpoint"]) == r["checkpoint_sha256"]
                   for r in manifest["training"])
        manifest["status"] = "complete"
        atomic_json(manifest_path, manifest)
        print("SUMMARY " + json.dumps(manifest["summary"]), flush=True)
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = repr(error)
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
