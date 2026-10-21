"""Generate paper experiment figures directly from reproduced CSV results.

All statistics are recomputed from per-seed rows with population standard
deviation (ddof=0).  The script writes vector PDF figures, PNG previews, a
long-form statistics audit CSV, and a SHA-256 manifest for reproducibility.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "figures" / "paper"
SEEDS = (42, 43, 44)

COLORS = {
    "primary": "#C44E52",
    "secondary": "#4C78A8",
    "adapter": "#E17C35",
    "neutral": "#7A8491",
    "grid": "#D9DEE5",
    "text": "#1F2933",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#68707A",
    "axes.linewidth": 0.7,
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def population_stats(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def add_audit_rows(audit, figure, items, metric_fields):
    for item_name, rows in items:
        seeds = sorted({int(row["seed"]) for row in rows})
        if tuple(seeds) != SEEDS:
            raise ValueError(
                f"{figure}/{item_name} has seeds {seeds}; expected {SEEDS}"
            )
        for metric_name, field in metric_fields:
            mean, std = population_stats(rows, field)
            audit.append({
                "figure": figure,
                "item": item_name,
                "metric": metric_name,
                "mean": mean,
                "population_std": std,
                "runs": len(rows),
                "seeds": "|".join(str(seed) for seed in seeds),
                "ddof": 0,
            })


def style_axis(axis, panel_title, x_label):
    axis.set_title(panel_title, loc="left", fontweight="bold")
    axis.set_xlabel(x_label)
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def padded_limits(means, stds, floor=None, pad_ratio=0.08):
    lower = min(mean - std for mean, std in zip(means, stds))
    upper = max(mean + std for mean, std in zip(means, stds))
    span = max(upper - lower, 1e-6)
    lower -= span * pad_ratio
    upper += span * pad_ratio
    if floor is not None:
        lower = max(floor, lower)
    return lower, upper


def save_figure(fig, stem):
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf_path, png_path


def plot_main_comparison(audit):
    source_paths = [
        WORKSPACE_DIR / "baselines" / f"final_comparison_seed{seed}.csv"
        for seed in SEEDS
    ]
    grouped = defaultdict(list)
    for path in source_paths:
        for row in read_csv(path):
            grouped[row["method"]].append(row)

    methods = sorted(
        grouped,
        key=lambda method: population_stats(grouped[method], "target_auc")[0],
    )
    auc_means, auc_stds, ap_means, ap_stds = [], [], [], []
    for method in methods:
        auc_mean, auc_std = population_stats(grouped[method], "target_auc")
        ap_mean, ap_std = population_stats(grouped[method], "target_ap")
        auc_means.append(auc_mean)
        auc_stds.append(auc_std)
        ap_means.append(ap_mean)
        ap_stds.append(ap_std)

    add_audit_rows(
        audit, "main_comparison", [(method, grouped[method]) for method in methods],
        [("ROC-AUC", "target_auc"), ("Average Precision", "target_ap")],
    )

    y = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 5.05), sharey=True)
    for axis, means, stds, title, label in (
        (axes[0], auc_means, auc_stds, "(a) ROC-AUC", "ROC-AUC"),
        (axes[1], ap_means, ap_stds, "(b) Average Precision", "Average Precision"),
    ):
        for index, method in enumerate(methods):
            is_spad = method == "SPAD"
            axis.errorbar(
                means[index], index, xerr=stds[index],
                fmt="o" if is_spad else "s",
                markersize=5.8 if is_spad else 4.2,
                color=COLORS["primary"] if is_spad else COLORS["secondary"],
                ecolor=COLORS["primary"] if is_spad else COLORS["neutral"],
                elinewidth=1.1, capsize=2.5, capthick=0.9, zorder=3,
            )
        style_axis(axis, title, label)
        axis.set_xlim(*padded_limits(means, stds, floor=0.0))
    axes[0].set_yticks(y, ["NEST" if method == "SPAD" else method for method in methods])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=COLORS["primary"], markeredgecolor=COLORS["primary"],
                   label="NEST"),
            Line2D([0], [0], marker="s", linestyle="none",
                   markerfacecolor=COLORS["secondary"], markeredgecolor=COLORS["secondary"],
                   label="Baselines"),
        ],
        loc="lower right", frameon=False,
    )
    fig.subplots_adjust(left=0.24, right=0.995, bottom=0.09, top=0.98, wspace=0.12)
    return source_paths, save_figure(fig, "fig_main_comparison")


def plot_component_ablation(audit, path=None, prefix="spad_ablation", full_key="no_reg"):
    # Optional inputs allow a final-objective rerun to regenerate only Figure 3.
    final_path = PROJECT_DIR / "results" / "component_ablation_final_20261007_table3_runs.csv"
    if path is None and final_path.exists():
        path, prefix, full_key = final_path, "spad_final_ablation_20261007", "full"
    path = Path(path) if path is not None else PROJECT_DIR / "results" / "component_ablation_runs.csv"
    rows = read_csv(path)
    key_map = {
        "spad_ablation_no_reg": "Full NEST",
        "spad_ablation_svd_alignment": "SVD coordinate alignment",
        "spad_ablation_no_descriptors": "w/o structural-temporal descriptors",
        "spad_ablation_no_residual": "w/o residual interaction",
        "spad_ablation_no_self": "w/o temporal consistency",
        "spad_ablation_no_proto": "w/o prototype compactness",
        "spad_ablation_dual": "Dual prototypes",
    }
    order = [
        "spad_ablation_no_reg",
        "spad_ablation_svd_alignment",
        "spad_ablation_no_descriptors",
        "spad_ablation_no_residual",
        "spad_ablation_no_self",
        "spad_ablation_no_proto",
        "spad_ablation_dual",
    ]
    key_map = {key.replace("spad_ablation", prefix).replace(f"{prefix}_no_reg", f"{prefix}_{full_key}"): label
               for key, label in key_map.items()}
    order = [key.replace("spad_ablation", prefix).replace(f"{prefix}_no_reg", f"{prefix}_{full_key}")
             for key in order]
    grouped = defaultdict(list)
    for row in rows:
        key = f"{prefix}_{row['variant']}" if "variant" in row else row["run_name"].rsplit("_seed", 1)[0]
        if key in key_map:
            grouped[key].append(row)
    items = [(key_map[key], grouped[key]) for key in order]
    add_audit_rows(
        audit, "component_ablation", items,
        [("ROC-AUC", "auc"), ("Average Precision", "ap")],
    )

    labels = [label for label, _ in items]
    auc = [population_stats(group, "auc") for _, group in items]
    ap = [population_stats(group, "ap") for _, group in items]
    y = np.arange(len(items))
    colors = [COLORS["primary"]] + [COLORS["secondary"]] * (len(items) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55), sharey=True)
    for axis, stats, title, x_label in (
        (axes[0], auc, "(a) ROC-AUC", "ROC-AUC"),
        (axes[1], ap, "(b) Average Precision", "Average Precision"),
    ):
        means = np.asarray([value[0] for value in stats])
        stds = np.asarray([value[1] for value in stats])
        axis.barh(
            y, means, height=0.58, color=colors, alpha=0.92,
            xerr=stds,
            error_kw={
                "ecolor": COLORS["neutral"], "elinewidth": 1.0,
                "capsize": 2.5, "capthick": 0.9,
            },
            zorder=3,
        )
        upper = float(np.max(means + stds))
        axis.set_xlim(0.0, upper * 1.15)
        value_offset = upper * 0.018
        for index, mean in enumerate(means):
            axis.text(
                means[index] + stds[index] + value_offset, index,
                f"{mean:.4f}",
                va="center", ha="left", fontsize=7.2,
                color=COLORS["text"],
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel(x_label)
        axis.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.31, right=0.995, bottom=0.14, top=0.95, wspace=0.15)
    return [path], save_figure(fig, "fig_component_ablation")


def plot_scoring_analysis(audit):
    path = WORKSPACE_DIR / "outputs" / "spad_ablation_results_20260928" / "scoring_ablation_runs.csv"
    rows = read_csv(path)
    labels = {
        "transported_prototype_distance": "Transported prototype distance",
        "latest_snapshot_local": "Latest-snapshot local inconsistency",
        "uniform_dynamic_local": "Uniform dynamic local inconsistency",
        "prototype_transport_residual": "Prototype transport residual",
        "recency_dynamic_local": "Recency-weighted dynamic local inconsistency",
    }
    order = [
        "recency_dynamic_local",
        "prototype_transport_residual",
        "uniform_dynamic_local",
        "latest_snapshot_local",
        "transported_prototype_distance",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy_key"]].append(row)
    items = [(labels[key], grouped[key]) for key in order]
    add_audit_rows(
        audit, "scoring_analysis", items,
        [("ROC-AUC", "roc_auc"), ("Average Precision", "average_precision")],
    )

    auc = [population_stats(group, "roc_auc") for _, group in items]
    ap = [population_stats(group, "average_precision") for _, group in items]
    short_labels = {
        "Recency-weighted dynamic local inconsistency": "Recency-weighted",
        "Prototype transport residual": "Prototype residual",
        "Uniform dynamic local inconsistency": "Uniform dynamic",
        "Latest-snapshot local inconsistency": "Latest snapshot",
        "Transported prototype distance": "Prototype distance",
    }
    annotation_offsets = {
        "Recency-weighted dynamic local inconsistency": (9, 9, "left", "bottom"),
        "Latest-snapshot local inconsistency": (-9, 9, "right", "bottom"),
        "Uniform dynamic local inconsistency": (-9, 9, "right", "bottom"),
        "Prototype transport residual": (9, -12, "left", "top"),
        "Transported prototype distance": (9, 8, "left", "bottom"),
    }

    auc_means = np.asarray([value[0] for value in auc])
    auc_stds = np.asarray([value[1] for value in auc])
    ap_means = np.asarray([value[0] for value in ap])
    ap_stds = np.asarray([value[1] for value in ap])
    fig, axis = plt.subplots(figsize=(6.35, 3.55))
    for index, (label, _) in enumerate(items):
        selected = label.startswith("Recency-weighted")
        axis.errorbar(
            auc_means[index], ap_means[index],
            xerr=auc_stds[index], yerr=ap_stds[index],
            fmt="o" if selected else "s",
            markersize=6.2 if selected else 4.8,
            color=COLORS["primary"] if selected else COLORS["secondary"],
            ecolor=COLORS["primary"] if selected else COLORS["neutral"],
            elinewidth=1.0, capsize=2.5, capthick=0.9, zorder=3,
        )
        dx, dy, horizontal, vertical = annotation_offsets[label]
        axis.annotate(
            short_labels[label],
            xy=(auc_means[index], ap_means[index]),
            xytext=(dx, dy), textcoords="offset points",
            ha=horizontal, va=vertical, fontsize=8,
            fontweight="bold" if selected else "normal",
            color=COLORS["primary"] if selected else COLORS["text"],
        )
    x_low, x_high = padded_limits(auc_means, auc_stds, floor=0.0, pad_ratio=0.16)
    y_low, y_high = padded_limits(ap_means, ap_stds, floor=0.0, pad_ratio=0.18)
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)
    axis.set_xlabel("ROC-AUC")
    axis.set_ylabel("Average Precision")
    axis.set_title("Scoring strategy trade-off", loc="left", fontweight="bold")
    axis.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.93, bottom=0.16, top=0.92)
    return [path], save_figure(fig, "fig_scoring_analysis")


def plot_few_shot(audit):
    path = PROJECT_DIR / "results" / "few_shot_adapter_trial.csv"
    rows = read_csv(path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["shot"])].append(row)
    shots = sorted(grouped)
    items = [(f"{shot}-shot", grouped[shot]) for shot in shots]
    add_audit_rows(
        audit, "few_shot_adaptation", items,
        [
            ("Zero-shot ROC-AUC", "zero_auc_query"),
            ("Adapter ROC-AUC", "auc"),
            ("Zero-shot AP", "zero_ap_query"),
            ("Adapter AP", "ap"),
        ],
    )

    x = np.asarray([0] + shots, dtype=float)
    reference_group = grouped[10] if 10 in grouped else grouped[shots[0]]
    zero_auc_at_zero = population_stats(reference_group, "zero_auc_query")
    zero_ap_at_zero = population_stats(reference_group, "zero_ap_query")
    zero_auc = [zero_auc_at_zero] + [
        population_stats(grouped[shot], "zero_auc_query") for shot in shots
    ]
    zero_ap = [zero_ap_at_zero] + [
        population_stats(grouped[shot], "zero_ap_query") for shot in shots
    ]
    adapter_auc = [
        population_stats(grouped[shot], "auc") for shot in shots
    ]
    adapter_ap = [
        population_stats(grouped[shot], "ap") for shot in shots
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    for axis, zero_stats, adapter_stats, title, y_label in (
        (axes[0], zero_auc, adapter_auc, "(a) ROC-AUC", "ROC-AUC"),
        (axes[1], zero_ap, adapter_ap, "(b) Average Precision", "Average Precision"),
    ):
        zero_means = np.asarray([item[0] for item in zero_stats])
        zero_stds = np.asarray([item[1] for item in zero_stats])
        adapter_means = np.asarray([item[0] for item in adapter_stats])
        adapter_stds = np.asarray([item[1] for item in adapter_stats])
        axis.errorbar(
            x, zero_means, yerr=zero_stds, marker="s", markersize=4.2,
            linestyle="--", linewidth=1.1, color=COLORS["neutral"],
            ecolor=COLORS["neutral"], capsize=2.5, label="Zero-shot reference",
        )
        axis.errorbar(
            np.asarray(shots), adapter_means, yerr=adapter_stds,
            marker="o", markersize=5.0, linestyle="-", linewidth=1.4,
            color=COLORS["adapter"], ecolor=COLORS["adapter"], capsize=2.5,
            label="Few-shot adapter",
        )
        axis.set_xticks(x, ["0"] + [str(shot) for shot in shots])
        axis.set_xlabel("Number of labeled target anomalies (shot)")
        axis.set_ylabel(y_label)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        all_means = list(zero_means) + list(adapter_means)
        all_stds = list(zero_stds) + list(adapter_stds)
        axis.set_ylim(*padded_limits(all_means, all_stds, floor=0.0, pad_ratio=0.14))
    axes[0].legend(loc="best", frameon=False)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.22, top=0.94, wspace=0.28)
    return [path], save_figure(fig, "fig_few_shot_adaptation")


def write_audit(audit):
    audit_path = OUTPUT_DIR / "paper_plot_statistics_population_std.csv"
    with audit_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    return audit_path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = []
    sources = []
    outputs = []
    for plotter in (
        plot_main_comparison,
        plot_component_ablation,
        plot_scoring_analysis,
        plot_few_shot,
    ):
        plot_sources, plot_outputs = plotter(audit)
        sources.extend(plot_sources)
        outputs.extend(plot_outputs)
    audit_path = write_audit(audit)
    outputs.append(audit_path)

    manifest_path = OUTPUT_DIR / "paper_figure_manifest.json"
    manifest = {
        "statistics": "mean and population standard deviation",
        "ddof": 0,
        "seeds": list(SEEDS),
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(set(sources))
        ],
        "outputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in outputs
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Paper figures written to {OUTPUT_DIR}")
    for path in outputs:
        print(path)
    print(manifest_path)


if __name__ == "__main__":
    main()
