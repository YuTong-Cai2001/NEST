"""FiGraph -> DGraphFin: a fixed-protocol, three-seed reference comparison.

Only this entry point is new. Model, source loss, optimizer construction,
training epoch, and shared descriptors are imported unchanged. The target
population is selected from observed edges without consulting its labels.
All six source models finish training before any target score is evaluated.
"""

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from data_processor import AnomalyGFMDataProcessor
from evaluate_reference_scoring import (
    SOURCE_FILES, SCORES, array_sha256, file_sha256, model_state_sha256,
    ranking_metrics, score_metrics, summarize,
)
from sparse_graph_utils import SparseGraphProcessor
from train_anomaly_gfm import (
    AnomalyGFMTrainer, TemporalGraphDataset, set_global_seed, sparse_collate_fn,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs/spad_reverse_transfer_20261020"
SEEDS = (42, 43, 44)
VARIANTS = ("full", "no_proto")
SCHEMA = 1


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def core_hashes():
    return {name: file_sha256(ROOT / name) for name in SOURCE_FILES}


def retained_windows(timestamps, width=7, maximum=20):
    """Match the existing nonempty-window sampling, without any label input."""
    timestamps = np.asarray(timestamps).reshape(-1)
    if not len(timestamps) or not np.isfinite(timestamps).all():
        raise ValueError("Empty or nonfinite edge timestamps")
    origin = timestamps.min()
    bins = np.floor((timestamps - origin) / width).astype(np.int64)
    ids = np.unique(bins)
    if len(ids) > maximum:
        ids = ids[::len(ids) // maximum][:maximum]
    if len(ids) < 5:
        raise ValueError("At least five nonempty target windows are required")
    return origin, bins, ids


def select_target_nodes(observed_edges, num_nodes, cap):
    """Top endpoint degree, with deterministic original-ID tie breaking.

    The signature deliberately has no labels or class-count arguments.
    Both unknown/background and binary-labeled nodes can be selected.
    """
    edges = np.asarray(observed_edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Expected edge array [E, 2]")
    if not len(edges) or edges.min() < 0 or edges.max() >= num_nodes:
        raise ValueError("Missing edges or out-of-range node identifiers")
    degree = np.bincount(edges.reshape(-1), minlength=num_nodes)
    candidates = np.flatnonzero(degree)
    order = np.lexsort((candidates, -degree[candidates]))
    selected = np.sort(candidates[order[:min(cap, len(order))]])
    if not len(selected):
        raise ValueError("Empty target population")
    return selected, degree


def normalize_target_graph(edges, mapping, count):
    """Preserve DGraphFin's stored direction and existing normalization."""
    mapped = mapping[edges]
    mapped = mapped[(mapped >= 0).all(axis=1)]
    graph = sp.csr_matrix((np.ones(len(mapped), dtype=np.float32),
                           (mapped[:, 0], mapped[:, 1])),
                          shape=(count, count), dtype=np.float32)
    graph.setdiag(1.0)
    rows = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    inv = np.maximum(rows, 1.0) ** np.float32(-0.5)
    coo = graph.tocoo()
    coo.data *= inv[coo.row] * inv[coo.col]
    return coo.tocsr(), int(len(mapped))


def class_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def input_signature(data):
    return {
        "feature_hashes": [array_sha256(f) for f in data["features"]],
        "graph_hashes": [{"shape": list(g.shape), "nnz": int(g.nnz),
                          "indptr": array_sha256(g.indptr),
                          "indices": array_sha256(g.indices),
                          "data": array_sha256(g.data)} for g in data["graphs"]],
        "node_ids": array_sha256(data["ids"]),
        "labels": array_sha256(data["labels"]),
    }


def plan(cap):
    return {
        "schema": SCHEMA, "source": "FiGraph", "target": "DGraphFin",
        "source_years": [2018, 2019, 2020, 2021, 2022],
        "source_labels": "2022 binary labels only; -1 background is masked",
        "source_population": "all nodes in the five-year edge-endpoint union",
        "target_population": "top endpoint degree over the 20 observed retained windows; no label input",
        "target_node_cap": cap, "target_degree_ties": "ascending original node ID",
        "target_window_width": 7, "target_max_retained_windows": 20,
        "target_window_sampling": "nonempty windows[::floor(W/20)][:20] when W>20",
        "model_input": "last five retained windows; descriptors computed before truncation",
        "target_direction": "preserve stored directed edges; local score accumulates both endpoints",
        "target_attributes": "released static attributes repeated over retained snapshots",
        "feature_alignment": "unchanged invariant 16 statistics + 8 structural-temporal descriptors",
        "standardization": "fit separately on first source and target snapshot, then freeze",
        "evaluation": "all selected target nodes with labels exactly 0 or 1; no support removal",
        "scores": list(SCORES), "score_direction": "higher is more anomalous; never label-fitted",
        "rank_population": "all selected target nodes, including background",
        "selection": "minimum total source training loss; target metrics only after all source training",
        "seeds": list(SEEDS), "epochs": 50,
        "lambda_proto": {"full": 0.3, "no_proto": 0.0},
        "lambda_self": 0.5, "no_proto_scope": "remove compactness only; prototype remains in BCE head",
    }


def prepare(out, cap):
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError("Preparation already exists; use --resume to verify and reuse")
    # This immutable plan is written before either domain's labels are loaded.
    before = core_hashes()
    protocol = plan(cap)
    protocol["core_sha256"] = before
    protocol["backup_commit"] = "9776c21c09dd67379ecd43775286a26e2162955e"
    protocol["entry_point_sha256"] = file_sha256(__file__)
    protocol["created_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(protocol_path, protocol)
    processor = AnomalyGFMDataProcessor({
        "alignment_mode": "invariant", "target_dim": 16, "seed": 42,
        "enable_cache": False,
        "figraph_path": str(ROOT / "data/FiGraph-main/data"),
        "dgraphfin_path": str(ROOT / "data/DGraphFin2"),
    })
    raw_paths = [ROOT / "data/DGraphFin2" / name for name in
                 ("dgraphfin.npz", "dgraphfinv2_edge_timestamp.npy")]
    for year in range(2018, 2023):
        base = ROOT / "data/FiGraph-main/data" / str(year)
        raw_paths.extend([base / f"edges{year}.csv",
                          base / f"ListedCompanyFeatures772_{year}.csv"])
    print("Hashing original data files", flush=True)
    raw_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in raw_paths}
    print("Loading FiGraph source sequence with the unchanged loader", flush=True)
    figraph = processor.load_figraph_temporal_sequence(2018, 2022)
    source_raw = figraph["temporal_sequences"][0]
    source_graphs = figraph["temporal_adjacencies"][0]
    source_labels = np.asarray(figraph["temporal_labels"][0][-1], dtype=np.int64)
    assert source_raw[0].shape == (192468, 772)
    assert class_counts(source_labels) == {"-1": 187336, "0": 5009, "1": 123}
    print("Loading DGraphFin; selecting target nodes without using labels", flush=True)
    raw = processor.load_dgraphfin_data()
    edges = np.asarray(raw["edge_index"], dtype=np.int64)
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T
    timestamps = np.asarray(raw["edge_timestamp"]).reshape(-1)
    if len(edges) != len(timestamps):
        raise ValueError("Timestamp and edge counts differ")
    origin, bins, window_ids = retained_windows(timestamps)
    observed = np.isin(bins, window_ids)
    selected, degree = select_target_nodes(edges[observed], raw["num_nodes"], cap)
    mapping = np.full(raw["num_nodes"], -1, dtype=np.int64)
    mapping[selected] = np.arange(len(selected), dtype=np.int64)
    target_graphs, edge_counts = [], []
    for window in window_ids:
        g, n = normalize_target_graph(edges[bins == window], mapping, len(selected))
        target_graphs.append(g)
        edge_counts.append(n)
    target_raw = np.asarray(raw["node_features"][selected], dtype=np.float32)
    # Target class membership is only read AFTER the sampling IDs are fixed.
    target_labels = np.asarray(raw["labels"][selected], dtype=np.int64)
    if set(np.unique(target_labels)) - {0, 1, 2, 3}:
        raise ValueError("Unexpected DGraphFin target labels")
    if not all(np.any(target_labels == c) for c in (0, 1)):
        raise ValueError("Fixed target sample lacks both classes; do not tune selection")
    print("Fitting domain-specific feature transforms, without labels", flush=True)
    _, target_aligned = processor.fit_feature_aligner(source_raw[0], target_raw)
    source_aligned = [processor.transform_source_features(x) for x in source_raw]
    source_features = processor.append_structural_temporal_features(source_aligned, source_graphs)
    target_features = processor.append_structural_temporal_features(
        [target_aligned] * len(target_graphs), target_graphs)
    processor.save_feature_aligner(str(out / "feature_aligner.pkl"))
    source = {"features": source_features[-5:], "graphs": source_graphs[-5:],
              "labels": source_labels, "ids": np.asarray(figraph["global_node_ids"], dtype=str)}
    target = {"features": target_features[-5:], "graphs": target_graphs[-5:],
              "labels": target_labels, "ids": selected}
    for domain in (source, target):
        for f in domain["features"]:
            assert f.shape == (len(domain["ids"]), 24) and np.isfinite(f).all()
        for g in domain["graphs"]:
            assert g.shape == (len(domain["ids"]),) * 2 and np.isfinite(g.data).all()
    data = {"source": source, "target": target}
    with (out / "prepared_inputs.pkl.tmp").open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "prepared_inputs.pkl.tmp").replace(out / "prepared_inputs.pkl")
    np.savez_compressed(out / "target_population.npz", global_node_ids=selected,
                        labels=target_labels, evaluation_mask=np.isin(target_labels, (0, 1)))
    final_nonself = target_graphs[-1].copy()
    final_nonself.setdiag(0)
    final_nonself.eliminate_zeros()
    stats = {
        "status": "prepared", "raw_sha256": raw_hashes,
        "protocol_sha256": file_sha256(protocol_path),
        "prepared_inputs_sha256": file_sha256(out / "prepared_inputs.pkl"),
        "feature_aligner_sha256": file_sha256(out / "feature_aligner.pkl"),
        "population_sha256": file_sha256(out / "target_population.npz"),
        "source_signature": input_signature(source), "target_signature": input_signature(target),
        "source_num_nodes": len(source_labels), "source_class_counts": class_counts(source_labels),
        "target_num_nodes": len(selected), "target_class_counts": class_counts(target_labels),
        "source_graph_nnz": [g.nnz for g in source["graphs"]],
        "target_retained_window_ids": window_ids.tolist(),
        "target_model_window_ids": window_ids[-5:].tolist(),
        "target_window_bounds": [[float(origin + w * 7), float(min(origin + w * 7 + 6, timestamps.max()))]
                                 for w in window_ids],
        "target_edges_per_retained_window": edge_counts,
        "target_minimum_selected_degree": int(degree[selected].min()),
        "target_final_snapshot_nonself_incident_nodes": int(np.count_nonzero(
            np.asarray(final_nonself.sum(axis=0)).ravel() + np.asarray(final_nonself.sum(axis=1)).ravel())),
        "core_unchanged": before == core_hashes(),
    }
    assert stats["core_unchanged"]
    atomic_json(out / "preparation.json", stats)
    print("PREPARED " + json.dumps({k: stats[k] for k in (
        "source_num_nodes", "source_class_counts", "target_num_nodes",
        "target_class_counts", "target_model_window_ids")}), flush=True)
    return data, stats


def verify_prepared(out):
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    stats = json.loads((out / "preparation.json").read_text(encoding="utf-8"))
    assert protocol["schema"] == SCHEMA
    assert protocol["entry_point_sha256"] == file_sha256(__file__), "Entry point changed after protocol freeze"
    assert protocol["core_sha256"] == core_hashes(), "Original model/trainer/processor changed"
    assert stats["protocol_sha256"] == file_sha256(out / "protocol.json")
    for name, key in (("prepared_inputs.pkl", "prepared_inputs_sha256"),
                      ("feature_aligner.pkl", "feature_aligner_sha256"),
                      ("target_population.npz", "population_sha256")):
        assert file_sha256(out / name) == stats[key], f"Changed {name}"
    for name, expected in stats["raw_sha256"].items():
        assert file_sha256(ROOT / name) == expected, f"Original data changed: {name}"
    with (out / "prepared_inputs.pkl").open("rb") as handle:
        data = pickle.load(handle)
    assert input_signature(data["source"]) == stats["source_signature"]
    assert input_signature(data["target"]) == stats["target_signature"]
    return data, stats


def training_config(out, variant, seed):
    provenance = json.loads((ROOT.parent / "outputs/spad_final_ablation_20261007/component_results.json").read_text(encoding="utf-8"))
    record = next(r for r in provenance["checks"] if r["run_name"] == "spad_ablation_no_reg_seed42")
    config = dict(record["config"])
    assert config["num_epochs"] == 50 and config.get("lambda_reg", 0) == 0
    config.update({"seed": seed, "lambda_proto": .3 if variant == "full" else 0.,
                   "run_name": f"figraph_to_dgraphfin_{variant}_seed{seed}",
                   "output_dir": str(out / f"{variant}_seed{seed}"),
                   "source_dataset": "FiGraph-2018-2022", "target_dataset": "DGraphFin",
                   "results_csv": str(out / "unused_legacy_result_path.csv")})
    return config


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def train_one(data, out, variant, seed):
    config = training_config(out, variant, seed)
    run_dir = Path(config["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "training_complete.json"
    if done_path.exists():
        result = json.loads(done_path.read_text(encoding="utf-8"))
        assert result["config"] == config and result["epochs_completed"] == 50
        assert file_sha256(result["checkpoint"]) == result["checkpoint_sha256"]
        print(f"SKIP verified source training {variant} seed={seed}", flush=True)
        return result
    started = time.perf_counter()
    with (run_dir / "training.log").open("a", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            trainer = AnomalyGFMTrainer(config)
            trainer.feature_dim = 24
            trainer.feature_aligner_path = str(out / "feature_aligner.pkl")
            src = data["source"]
            temporal = {"temporal_sequences": [src["features"]],
                        "temporal_adjacencies": [src["graphs"]], "valid_timestamps": [0]}
            dataset = TemporalGraphDataset(temporal, 5, labels=src["labels"])
            trainer.train_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                                              collate_fn=sparse_collate_fn, num_workers=0)
            trainer.create_model()
            initial_hash = model_state_sha256(trainer.model)
            history, best, best_epoch, start_epoch = [], float("inf"), -1, 0
            last = run_dir / "last_training_state.pth"
            if last.exists():
                state = torch.load(last, map_location="cpu", weights_only=False)
                assert state["config"] == config
                trainer.model.load_state_dict(state["model"])
                trainer.optimizer.load_state_dict(state["optimizer"])
                trainer.scheduler.load_state_dict(state["scheduler"])
                history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
                start_epoch = state["next_epoch"]
                assert state["initial_state_sha256"] == initial_hash
                restore_rng(state["rng"])
                print(f"RESUME epoch {start_epoch}", flush=True)
                del state
            for epoch in range(start_epoch, 50):
                epoch_start = time.perf_counter()
                metrics = trainer.train_epoch_sparse()
                loss = float(metrics["total_loss"])
                if not np.isfinite(loss):
                    raise ValueError("Nonfinite source loss")
                if loss < best:
                    best, best_epoch = loss, epoch
                    trainer.save_model("best_source_loss.pth")
                trainer.scheduler.step(loss)
                row = {"epoch": epoch, "seconds": time.perf_counter() - epoch_start,
                       **{k: float(v) for k, v in metrics.items() if np.isscalar(v)}}
                history.append(row)
                print("SOURCE_EPOCH " + json.dumps(row), flush=True)
                # Exact RNG/optimizer resume is independent of target labels.
                state = {"config": config, "model": trainer.model.state_dict(),
                         "optimizer": trainer.optimizer.state_dict(),
                         "scheduler": trainer.scheduler.state_dict(), "rng": rng_state(),
                         "history": history, "best": best, "best_epoch": best_epoch,
                         "next_epoch": epoch + 1, "initial_state_sha256": initial_hash}
                temporary = run_dir / "last_training_state.pth.tmp"
                torch.save(state, temporary)
                temporary.replace(last)
                atomic_json(run_dir / "source_training_history.json", history)
                torch.cuda.empty_cache()
            checkpoint = run_dir / "checkpoints/best_source_loss.pth"
            result = {"variant": variant, "seed": seed, "config": config,
                      "epochs_completed": len(history), "best_source_loss": best,
                      "best_epoch_zero_based": best_epoch,
                      "initial_state_sha256": initial_hash,
                      "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                      "training_seconds": sum(r["seconds"] for r in history),
                      "wall_seconds_this_invocation": time.perf_counter() - started,
                      "resolved_model_config": trainer.resolved_model_config}
            assert result["epochs_completed"] == 50
            atomic_json(done_path, result)
            del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"TRAINED {variant} seed={seed} epoch={best_epoch} source_loss={best:.6f}", flush=True)
    return result


def evaluate_one(data, out, trained):
    from anomaly_gfm_complete_model import AnomalyGFMModel
    variant, seed = trained["variant"], trained["seed"]
    run_dir = out / f"{variant}_seed{seed}"
    result_path = run_dir / "evaluation.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["checkpoint_sha256"] == trained["checkpoint_sha256"]
        assert file_sha256(result["scores_path"]) == result["scores_sha256"]
        return result
    set_global_seed(seed)
    checkpoint = torch.load(trained["checkpoint"], map_location="cpu", weights_only=False)
    assert file_sha256(trained["checkpoint"]) == trained["checkpoint_sha256"]
    model = AnomalyGFMModel(checkpoint["resolved_model_config"])
    state = checkpoint["model_state_dict"]
    assert all(k.startswith("base_model.") for k in state)
    model.load_state_dict({k[len("base_model."):]: v for k, v in state.items()}, strict=True)
    model.to("cuda").requires_grad_(False).eval()
    before = model_state_sha256(model)
    target = data["target"]
    processor = SparseGraphProcessor(device="cuda")
    tensor = torch.as_tensor(np.stack(target["features"]), device="cuda").unsqueeze(0)
    graphs = []
    for g in target["graphs"]:
        ei, ea = processor.sparse_to_pyg_format(g)
        graphs.append({"edge_index": ei, "edge_attr": ea, "num_nodes": len(target["ids"])})
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model(tensor, graphs, tensor[:, -1])
    assert before == model_state_sha256(model)
    raw = {SCORES[0]: output["learned_prototype_distances"].cpu().numpy().reshape(-1),
           SCORES[1]: output["recency_neighbor_inconsistency"].cpu().numpy().reshape(-1)}
    default = output["anomaly_scores"].cpu().numpy().reshape(-1)
    labels = target["labels"]
    mask = np.isin(labels, (0, 1))
    metrics = {name: score_metrics(labels, values, mask) for name, values in raw.items()}
    score_path = run_dir / "target_scores.npz"
    np.savez_compressed(score_path, **raw, default_model_score=default)
    result = {"variant": variant, "seed": seed,
              "lambda_proto": .3 if variant == "full" else 0., "lambda_self": .5,
              "checkpoint": trained["checkpoint"], "checkpoint_sha256": trained["checkpoint_sha256"],
              "scores": metrics, "default_model_metrics": ranking_metrics(labels, default, mask),
              "scores_path": str(score_path), "scores_sha256": file_sha256(score_path),
              "modeled_nodes": len(labels), "evaluated_nodes": int(mask.sum()),
              "anomalous_evaluated_nodes": int(np.sum(labels[mask] == 1)),
              "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
              "model_state_unchanged": True}
    atomic_json(result_path, result)
    print("RESULT " + json.dumps(result), flush=True)
    del model, output, tensor, graphs, checkpoint, state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-nodes", type=int, default=100000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this planned experiment")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if args.resume:
        data, stats = verify_prepared(out)
        protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
        assert protocol["target_node_cap"] == args.target_nodes
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Refusing to overwrite existing experiment; use --resume")
        data, stats = prepare(out, args.target_nodes)
    if args.prepare_only:
        return
    before = core_hashes()
    manifest_path = out / "reverse_transfer_results.json"
    manifest = {"status": "training", "direction": "FiGraph_to_DGraphFin",
                "seeds": list(SEEDS), "variants": list(VARIANTS), "scores": list(SCORES),
                "protocol_sha256": stats["protocol_sha256"], "core_sha256_before": before,
                "runtime": {"python": sys.version, "torch": torch.__version__,
                            "gpu": torch.cuda.get_device_name(0)},
                "training": [], "runs": []}
    atomic_json(manifest_path, manifest)
    try:
        for seed in SEEDS:
            for variant in VARIANTS:
                print(f"TRAIN_START {variant} seed={seed}", flush=True)
                trained = train_one(data, out, variant, seed)
                manifest["training"].append(trained)
                atomic_json(manifest_path, manifest)
        for seed in SEEDS:
            pair = [r for r in manifest["training"] if r["seed"] == seed]
            assert pair[0]["initial_state_sha256"] == pair[1]["initial_state_sha256"]
        manifest["status"] = "evaluating"
        atomic_json(manifest_path, manifest)
        for trained in manifest["training"]:
            manifest["runs"].append(evaluate_one(data, out, trained))
            atomic_json(manifest_path, manifest)
        manifest["summary"] = summarize(manifest["runs"])
        manifest["core_sha256_after"] = core_hashes()
        assert manifest["core_sha256_after"] == before
        manifest["all_checkpoints_unchanged_during_evaluation"] = all(
            file_sha256(r["checkpoint"]) == r["checkpoint_sha256"] for r in manifest["training"])
        assert manifest["all_checkpoints_unchanged_during_evaluation"]
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
