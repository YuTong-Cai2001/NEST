"""Cross source compactness training with two fixed target scoring references.

This evaluation-only experiment reuses source-loss-selected checkpoints. It
does not train, fit preprocessing, change model switches, tune score signs, or
update manuscript/legacy result files. The no_proto row removes L_proto only;
the source detection head still uses its normal prototype.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SEEDS = (42, 43, 44)
VARIANTS = ("full", "no_proto")
SCORES = ("source_prototype_distance", "recency_neighborhood_disagreement")
SOURCE_FILES = (
    "anomaly_gfm_complete_model.py", "train_anomaly_gfm.py",
    "data_processor.py", "few_shot_finetune.py", "sparse_graph_utils.py",
    "model_adapter.py",
)
COMMON_TRAINING = (
    "num_epochs", "learning_rate", "batch_size", "sequence_length",
    "weight_decay", "start_year", "end_year", "prototype_mode",
    "alignment_mode", "lambda_self", "temporal_margin",
    "use_structural_temporal_descriptors", "model_config",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(str(values.dtype).encode())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def stable_percentile(values):
    """Use identical full-population ordinal ranking for both references."""
    values = np.asarray(values).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite scores; refusing to replace them silently")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def ranking_metrics(labels, values, mask):
    labels = np.asarray(labels)[mask].astype(np.int64)
    values = np.asarray(values)[mask]
    if len(np.unique(labels)) != 2:
        raise ValueError("Evaluation requires both valid binary classes")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite evaluation scores")
    return {
        "auc": float(roc_auc_score(labels, values)),
        "ap": float(average_precision_score(labels, values)),
    }


def score_metrics(labels, values, mask):
    ranked = ranking_metrics(labels, stable_percentile(values), mask)
    raw = ranking_metrics(labels, values, mask)
    evaluated = np.asarray(values)[mask]
    _, inverse, counts = np.unique(evaluated, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=np.asarray(labels)[mask] == 1,
                            minlength=len(counts))
    mixed = (positives > 0) & (positives < counts)
    return {
        **ranked,
        "raw_auc": raw["auc"], "raw_ap": raw["ap"],
        "rank_minus_raw_auc": ranked["auc"] - raw["auc"],
        "rank_minus_raw_ap": ranked["ap"] - raw["ap"],
        "unique_evaluated_raw_scores": int(len(counts)),
        "evaluated_nodes_in_tie_groups": int(counts[counts > 1].sum()),
        "mixed_class_tie_groups": int(mixed.sum()),
    }


def mean_std(values):
    return {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "values": [float(x) for x in values]}


def summarize(runs):
    by_key = {(run["variant"], run["seed"]): run for run in runs}
    cells = []
    for variant in VARIANTS:
        for score in SCORES:
            group = [by_key[(variant, seed)]["scores"][score] for seed in SEEDS]
            cells.append({
                "variant": variant, "score": score,
                **{key: mean_std([item[key] for item in group])
                   for key in ("auc", "ap", "raw_auc", "raw_ap")},
            })

    def value(variant, seed, score, metric):
        return by_key[(variant, seed)]["scores"][score][metric]

    paired = []
    for name, minuend, subtrahend in (
        ("local_minus_source_distance_full", ("full", SCORES[1]),
         ("full", SCORES[0])),
        ("local_minus_source_distance_no_proto", ("no_proto", SCORES[1]),
         ("no_proto", SCORES[0])),
        ("compactness_gain_local", ("full", SCORES[1]),
         ("no_proto", SCORES[1])),
        ("compactness_gain_source_distance", ("full", SCORES[0]),
         ("no_proto", SCORES[0])),
    ):
        paired.append({
            "contrast": name,
            **{metric: mean_std([
                value(minuend[0], seed, minuend[1], metric)
                - value(subtrahend[0], seed, subtrahend[1], metric)
                for seed in SEEDS
            ]) for metric in ("auc", "ap")},
        })
    return {"seeds": list(SEEDS), "ddof": 0, "cells": cells,
            "paired_contrasts": paired}


def input_fingerprint(data, labels):
    sequence = data["temporal_sequences"][0]
    graphs = data["temporal_adjacencies"][0]
    ids = json.dumps([str(x) for x in data["global_node_ids"]],
                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "node_order_sha256": hashlib.sha256(ids).hexdigest(),
        "labels_sha256": array_sha256(labels),
        "features_sha256": [array_sha256(x) for x in sequence],
        "graphs_sha256": [{"shape": list(g.shape), "nnz": int(g.nnz),
                           "indptr": array_sha256(g.indptr),
                           "indices": array_sha256(g.indices),
                           "data": array_sha256(g.data)} for g in graphs],
    }


def model_state_sha256(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def target_transform_signature(path):
    """Ignore saved RNG metadata; compare every fitted target transform."""
    with Path(path).open("rb") as handle:
        state = pickle.load(handle)
    keys = ("target_dim", "alignment_mode", "target_reducer",
            "target_scaler", "target_descriptor_scaler")
    return {key: hashlib.sha256(pickle.dumps(state.get(key))).hexdigest()
            for key in keys}


def self_test():
    labels = np.array([0, 1, -1, 0, 1])
    mask = np.isin(labels, (0, 1))
    scores = np.array([0.1, 0.8, 0.4, 0.2, 0.9])
    assert np.array_equal(stable_percentile(scores), [0, .75, .5, .25, 1])
    assert score_metrics(labels, scores, mask)["auc"] == 1
    assert score_metrics(labels, scores, mask)["ap"] == 1
    tied = score_metrics(np.array([1, 0, 1, 0]), np.ones(4), np.ones(4, bool))
    assert tied["raw_auc"] == .5
    assert tied["mixed_class_tie_groups"] == 1
    assert tied["evaluated_nodes_in_tie_groups"] == 4
    assert tied["auc"] != tied["raw_auc"]
    try:
        stable_percentile([0, np.nan])
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite scores must fail")
    assert stable_percentile([5]).tolist() == [0]
    assert np.array_equal(stable_percentile([2, 1, 1]), [1, 0, .5])
    print("SELF_TEST_OK: ranks, ties, label mask, score orientation, finite checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, default=WORKSPACE /
                        "outputs/spad_final_ablation_20261007/component_results.json")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE /
                        "outputs/spad_reference_scoring_20261015")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Verify saved results and run only missing checkpoints")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    import torch
    from anomaly_gfm_complete_model import AnomalyGFMModel
    from few_shot_finetune import FewShotFinetuner

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing an unplanned CPU run")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    records = {}
    for record in provenance["checks"]:
        name = record["run_name"]
        if name.startswith("spad_ablation_no_reg_seed"):
            records[("full", int(record["config"]["seed"]))] = record
        elif name.startswith("spad_final_ablation_20261007_no_proto_seed"):
            records[("no_proto", int(record["config"]["seed"]))] = record
    assert set(records) == {(v, s) for v in VARIANTS for s in SEEDS}
    reference = records[("full", 42)]
    reference_config = reference["config"]
    reference_model = reference["resolved_model_config"]
    for (variant, seed), record in records.items():
        config = record["config"]
        assert config["lambda_proto"] == (.3 if variant == "full" else 0)
        assert config.get("lambda_reg", 0) == 0
        assert config["seed"] == seed
        assert all(config[k] == reference_config[k] for k in COMMON_TRAINING)
        assert record["resolved_model_config"] == reference_model
        checkpoint_path = Path(record["checkpoint"])
        assert checkpoint_path.is_file(), checkpoint_path
        assert file_sha256(checkpoint_path) == record["checkpoint_sha256"]
        assert (checkpoint_path.parent.parent / "feature_aligner.pkl").is_file()
    expected = {(item["variant"], seed): {"auc": item["auc"][i],
                                         "ap": item["ap"][i]}
                for item in provenance["items"] if item["variant"] in VARIANTS
                for i, seed in enumerate(provenance["seeds"])}

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing experiment: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sources_before = {name: file_sha256(ROOT / name) for name in SOURCE_FILES}
    manifest = {
        "status": "running", "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training_performed": False,
        "seeds": list(SEEDS), "variants": list(VARIANTS), "scores": list(SCORES),
        "protocol": {
            "years": [2018, 2019, 2020, 2021, 2022],
            "normalization": "stable ordinal percentile over ALL target nodes",
            "orientation": "higher is more anomalous for both scores; no sign fitting",
            "evaluation": "final-year labels exactly 0 or 1; no support removal",
            "score_extraction": "two references from the SAME forward pass per checkpoint",
            "no_proto_scope": "lambda_proto=0; prototype remains in source detection head",
            "raw_score_metrics": "reported separately to expose tie-breaking effects",
            "source_selection": "existing best_source_loss checkpoints, not target selection",
            "reproduction_tolerance": 2e-5,
        },
        "source_sha256_before": sources_before,
        "evaluation_script_sha256": file_sha256(Path(__file__)),
        "provenance_sha256": file_sha256(args.provenance),
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "cuda_build": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0)},
        "runs": [],
    }
    manifest_path = out / "reference_scoring_results.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert previous["source_sha256_before"] == sources_before
        assert previous["provenance_sha256"] == file_sha256(args.provenance)
        assert previous["protocol"] == manifest["protocol"]
        for run in previous["runs"]:
            assert file_sha256(run["scores_path"]) == run["scores_sha256"]
            assert file_sha256(run["checkpoint"]) == run["checkpoint_sha256"]
        manifest = previous
        manifest["status"] = "running"
        manifest.pop("error", None)
        manifest.setdefault("resume_history", []).append({
            "script_sha256": file_sha256(Path(__file__)),
            "reason": "Reuse identical fitted target transforms after interrupted execution",
            "completed_runs_retained": len(manifest["runs"]),
        })

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                             allow_nan=False), encoding="utf-8")

    shared_input = manifest.get("shared_input_fingerprint")
    completed = {(run["variant"], run["seed"]) for run in manifest["runs"]}
    cached_data = cached_labels = cached_transform = None
    save()
    try:
        for variant in VARIANTS:
            for seed in SEEDS:
                if (variant, seed) in completed:
                    print(f"SKIP verified saved result {variant} seed={seed}", flush=True)
                    continue
                started = time.perf_counter()
                record = records[(variant, seed)]
                path = Path(record["checkpoint"])
                print(f"START {variant} seed={seed}: {path}", flush=True)
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                assert checkpoint["resolved_model_config"] == reference_model
                config = checkpoint["config"]
                assert config["seed"] == seed
                assert config["lambda_proto"] == (.3 if variant == "full" else 0)
                assert all(config[k] == reference_config[k] for k in COMMON_TRAINING)

                loader = FewShotFinetuner({
                    "model_path": str(path), "seed": seed,
                    "start_year": 2018, "end_year": 2022,
                    "figraph_path": str(ROOT / "data/FiGraph-main/data"),
                })
                loader.checkpoint = checkpoint
                loader.model_config = dict(checkpoint["resolved_model_config"])
                aligner = checkpoint.get("feature_aligner_path")
                if not aligner or not Path(aligner).is_file():
                    aligner = path.parent.parent / "feature_aligner.pkl"
                transform = target_transform_signature(aligner)
                if cached_data is not None and transform == cached_transform:
                    data, labels = cached_data, cached_labels
                    print("REUSE identical fitted target transforms and prepared inputs", flush=True)
                else:
                    data, labels = loader.load_test_data()
                    cached_data, cached_labels, cached_transform = data, labels, transform
                assert list(data["available_years"]) == [2018, 2019, 2020, 2021, 2022]
                assert data["num_nodes"] == 192468 and data["feature_dim"] == 24
                mask = np.isin(labels, (0, 1))
                assert int(mask.sum()) == 5132 and int((labels[mask] == 1).sum()) == 123
                inputs = input_fingerprint(data, labels)
                if shared_input is None:
                    shared_input = inputs
                    manifest["shared_input_fingerprint"] = inputs
                    np.savez_compressed(out / "target_population.npz",
                                        global_node_ids=np.array(data["global_node_ids"], dtype=str),
                                        labels=labels, evaluation_mask=mask)
                else:
                    assert inputs == shared_input, "Target inputs differ across checkpoints"

                model = AnomalyGFMModel(loader.model_config)
                state = checkpoint["model_state_dict"]
                if all(key.startswith("base_model.") for key in state):
                    state = {key[len("base_model."):]: tensor for key, tensor in state.items()}
                model.load_state_dict(state, strict=True)
                model.requires_grad_(False)
                model.to(loader.device).eval()
                state_before = model_state_sha256(model)
                torch.cuda.reset_peak_memory_stats()
                output = loader._full_graph_forward(model, data)
                raw = {
                    SCORES[0]: output["learned_prototype_distances"].detach().cpu().numpy().reshape(-1),
                    SCORES[1]: output["recency_neighbor_inconsistency"].detach().cpu().numpy().reshape(-1),
                }
                default_score = output["anomaly_scores"].detach().cpu().numpy().reshape(-1)
                assert all(len(values) == len(labels) for values in raw.values())
                assert state_before == model_state_sha256(model), "Evaluation changed model state"
                results = {name: score_metrics(labels, values, mask)
                           for name, values in raw.items()}
                default_metrics = ranking_metrics(labels, default_score, mask)
                reproduction = {
                    metric: {"expected": expected[(variant, seed)][metric],
                             "actual": default_metrics[metric],
                             "difference": default_metrics[metric] - expected[(variant, seed)][metric]}
                    for metric in ("auc", "ap")
                }
                assert all(abs(check["difference"]) <= 2e-5 for check in reproduction.values()), (
                    "Historical result not reproduced", reproduction)
                score_path = out / f"{variant}_seed{seed}_scores.npz"
                np.savez_compressed(score_path, **raw, default_model_score=default_score)
                run = {
                    "variant": variant, "seed": seed, "checkpoint": str(path),
                    "checkpoint_sha256": file_sha256(path),
                    "feature_aligner_sha256": file_sha256(aligner),
                    "target_transform_signature": transform,
                    "lambda_proto": config["lambda_proto"],
                    "lambda_self": config["lambda_self"],
                    "scores": results, "default_model_metrics": default_metrics,
                    "default_model_reproduction": reproduction,
                    "scores_path": str(score_path), "scores_sha256": file_sha256(score_path),
                    "modeled_nodes": len(labels), "evaluated_nodes": int(mask.sum()),
                    "anomalous_evaluated_nodes": int((labels[mask] == 1).sum()),
                    "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                manifest["runs"].append(run)
                save()
                print("RESULT " + json.dumps({"variant": variant, "seed": seed,
                                               "scores": results,
                                               "reproduction": reproduction}), flush=True)
                del output, model, checkpoint, state, loader, data, raw, labels
                gc.collect()
                torch.cuda.empty_cache()

        manifest["summary"] = summarize(manifest["runs"])
        manifest["source_sha256_after"] = {name: file_sha256(ROOT / name)
                                              for name in SOURCE_FILES}
        assert manifest["source_sha256_after"] == sources_before
        manifest["checkpoints_unchanged"] = all(
            file_sha256(record["checkpoint"]) == record["checkpoint_sha256"]
            for record in records.values())
        assert manifest["checkpoints_unchanged"]
        manifest["status"] = "complete"
        manifest["completed_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
        print("SUMMARY " + json.dumps(manifest["summary"]), flush=True)
        print(f"COMPLETE {manifest_path}", flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
