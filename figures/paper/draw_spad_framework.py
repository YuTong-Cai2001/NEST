"""Draw the paper-facing SPAD framework as vector PDF/SVG plus PNG preview."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

OUT_DIR = Path(__file__).resolve().parent
STEM = OUT_DIR / "fig_spad_framework_draft"

NAVY = "#234F78"
INK = "#27313D"
MUTED = "#687582"
BLUE = "#DCECF7"
BLUE_2 = "#EDF5FA"
ORANGE = "#FBE7D7"
ORANGE_2 = "#FFF5ED"
GREEN = "#E4F2E4"
GREEN_2 = "#F1F8F1"
RED = "#D84B4B"
GRAY = "#EEF1F4"
GRAY_2 = "#F8F9FA"
MID = "#8B98A4"


def rounded(ax, x, y, w, h, fc="white", ec=NAVY, lw=1.4, radius=0.12,
            ls="solid", z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=8.2, weight="normal", color=INK,
          ha="center", va="center", z=5, linespacing=1.15, rotation=0):
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, zorder=z, linespacing=linespacing,
            rotation=rotation)


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.25, style="-|>",
          mutation=10, connection="arc3", ls="solid", z=3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=mutation, linewidth=lw, color=color,
        connectionstyle=connection, linestyle=ls, zorder=z,
        shrinkA=1.5, shrinkB=1.5,
    ))


def panel(ax, x, y, w, h, title, panel_id, fc):
    rounded(ax, x, y, w, h, fc=fc, ec="#AAB4BD", lw=1.1, radius=0.18)
    rounded(ax, x + 0.18, y + h - 0.42, 0.42, 0.28,
            fc=NAVY, ec=NAVY, lw=0.0, radius=0.08, z=4)
    label(ax, x + 0.39, y + h - 0.28, panel_id, size=9.0,
          weight="bold", color="white")
    label(ax, x + 0.72, y + h - 0.28, title, size=11.0,
          weight="bold", ha="left")


def badge(ax, x, y, text, fc, color=NAVY, width=None):
    if width is None:
        width = 0.48 + 0.055 * len(text)
    rounded(ax, x, y, width, 0.22, fc=fc, ec="none", lw=0,
            radius=0.06, z=5)
    label(ax, x + width / 2, y + 0.11, text, size=6.8,
          weight="bold", color=color, z=6)


def draw_graph(ax, cx, cy, scale=1.0, mode="source"):
    pts = [
        (-0.34, 0.03), (-0.13, 0.27), (0.14, 0.22),
        (0.34, -0.01), (0.10, -0.26), (-0.22, -0.23),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
             (5, 0), (1, 4), (0, 2), (2, 4)]
    for a, b in edges:
        x1, y1 = pts[a]
        x2, y2 = pts[b]
        ax.plot([cx + x1 * scale, cx + x2 * scale],
                [cy + y1 * scale, cy + y2 * scale],
                color="#657786", lw=0.7, zorder=3)
    for idx, (px, py) in enumerate(pts):
        if mode == "target":
            fc, ec = "#D8DDE2", "#A8B0B7"
        else:
            abnormal = idx == 3
            fc = "#F28B82" if abnormal else "#CFECCB"
            ec = RED if abnormal else "#68A95E"
        ax.add_patch(Circle((cx + px * scale, cy + py * scale),
                            0.066 * scale, facecolor=fc, edgecolor=ec,
                            lw=0.8, zorder=4))


def draw_sequence(ax, x, y, mode="source", compact=False, show_time=True):
    offsets = (0.0, 0.46, 0.92) if not compact else (0.0, 0.28, 0.56)
    scale = 0.74 if not compact else 0.50
    for i, off in enumerate(offsets):
        rounded(ax, x + off, y + 0.035 * i, 0.72 if not compact else 0.46,
                0.58 if not compact else 0.40, fc="white", ec="#C0C8CF",
                lw=0.7, radius=0.07, z=2 + i)
        draw_graph(ax, x + off + (0.36 if not compact else 0.23),
                   y + (0.29 if not compact else 0.20) + 0.035 * i,
                   scale=scale, mode=mode)
    if show_time:
        label(ax, x + offsets[-1] + (0.80 if not compact else 0.61),
              y + (0.30 if not compact else 0.23), "time", size=6.8,
              color=MUTED, ha="left")
        arrow(ax, x + offsets[-1] + (0.67 if not compact else 0.50),
              y + (0.30 if not compact else 0.23),
              x + offsets[-1] + (0.78 if not compact else 0.59),
              y + (0.30 if not compact else 0.23), color=MUTED,
              lw=0.8, mutation=7)


def draw_star(ax, cx, cy, r=0.18, fc="#BDE7B5", ec="#62A956"):
    import math
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        radius = r if i % 2 == 0 else r * 0.43
        points.append((cx + radius * math.cos(angle),
                       cy + radius * math.sin(angle)))
    ax.add_patch(Polygon(points, closed=True, facecolor=fc,
                         edgecolor=ec, lw=0.9, zorder=5))


def main():
    fig, ax = plt.subplots(figsize=(16, 9.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # (a) Domain construction
    panel(ax, 0.28, 7.20, 15.44, 2.14,
          "Domain-invariant dynamic graph construction", "a", GRAY_2)

    rounded(ax, 0.65, 8.17, 3.55, 0.55, fc="white", ec="#D28A52", lw=1.15)
    draw_sequence(ax, 0.78, 8.21, mode="source", compact=True, show_time=False)
    label(ax, 2.73, 8.45, "Labeled source dynamic sequence", size=8.0,
          weight="bold")

    rounded(ax, 0.65, 7.55, 3.55, 0.55, fc="white", ec="#6E98B8", lw=1.15)
    draw_sequence(ax, 0.78, 7.59, mode="target", compact=True, show_time=False)
    label(ax, 2.73, 7.83, "Unlabeled target dynamic sequence", size=8.0,
          weight="bold")

    rounded(ax, 4.75, 7.55, 3.55, 1.18, fc=BLUE_2, ec=NAVY, lw=1.25)
    label(ax, 6.525, 8.31, "Domain-wise robust normalization", size=8.7,
          weight="bold")
    label(ax, 6.525, 8.01, "+ distribution-statistic coordinates", size=8.0)
    label(ax, 6.525, 7.74, "Common attribute space without target labels",
          size=7.2, color=MUTED)
    arrow(ax, 4.20, 8.45, 4.72, 8.34)
    arrow(ax, 4.20, 7.83, 4.72, 7.94)

    rounded(ax, 8.75, 7.55, 2.85, 1.18, fc=ORANGE_2, ec="#C47A3D", lw=1.25)
    label(ax, 10.175, 8.31, "8-D structural-temporal", size=8.7, weight="bold")
    label(ax, 10.175, 8.02, "descriptor augmentation", size=8.7, weight="bold")
    label(ax, 10.175, 7.73, "degree context  |  change  |  edge churn",
          size=6.9, color=MUTED)
    arrow(ax, 8.30, 8.14, 8.72, 8.14)

    rounded(ax, 12.05, 7.55, 3.22, 1.18, fc="white", ec=NAVY, lw=1.35)
    label(ax, 13.66, 8.34, "Aligned dynamic sequences", size=8.8,
          weight="bold")
    label(ax, 13.66, 8.02, "source  +  target", size=8.1)
    label(ax, 13.66, 7.73, "shared semantic coordinates", size=7.0, color=MUTED)
    arrow(ax, 11.60, 8.14, 12.02, 8.14)

    # (b) Source learning
    panel(ax, 0.28, 0.28, 9.18, 6.64,
          "Source-domain SPAD representation learning", "b", ORANGE_2)
    badge(ax, 7.92, 6.33, "TRAINABLE", ORANGE, color="#A55322", width=1.15)

    rounded(ax, 0.66, 4.38, 1.15, 1.40, fc="white", ec="#D28A52", lw=1.2)
    draw_sequence(ax, 0.77, 4.75, mode="source", compact=True, show_time=False)
    label(ax, 1.235, 4.55, "aligned source", size=6.8, color=MUTED)

    rounded(ax, 2.12, 5.18, 1.48, 0.86, fc=BLUE, ec=NAVY, lw=1.3)
    label(ax, 2.86, 5.70, "Dynamic GNN", size=9.0, weight="bold")
    label(ax, 2.86, 5.43, "spatial evolution", size=7.0, color=MUTED)
    rounded(ax, 2.12, 3.98, 1.48, 0.86, fc=BLUE, ec=NAVY, lw=1.3)
    label(ax, 2.86, 4.50, "TCN", size=9.0, weight="bold")
    label(ax, 2.86, 4.23, "temporal context", size=7.0, color=MUTED)
    arrow(ax, 1.81, 5.10, 2.08, 5.62)
    arrow(ax, 1.81, 5.02, 2.08, 4.42)

    rounded(ax, 3.93, 5.18, 0.86, 0.86, fc="white", ec=NAVY, lw=1.1)
    label(ax, 4.36, 5.61, "Spatial", size=7.6, weight="bold")
    label(ax, 4.36, 5.36, "projector", size=7.0)
    rounded(ax, 3.93, 3.98, 0.86, 0.86, fc="white", ec=NAVY, lw=1.1)
    label(ax, 4.36, 4.41, "Temporal", size=7.6, weight="bold")
    label(ax, 4.36, 4.16, "projector", size=7.0)
    arrow(ax, 3.60, 5.61, 3.90, 5.61)
    arrow(ax, 3.60, 4.41, 3.90, 4.41)

    rounded(ax, 5.18, 4.34, 1.45, 1.30, fc=ORANGE, ec="#B96A32", lw=1.35)
    label(ax, 5.905, 5.23, "Residual interaction", size=8.3, weight="bold")
    label(ax, 5.905, 4.92, "temporal change", size=7.2)
    label(ax, 5.905, 4.66, "+ spatial-temporal residual", size=7.0)
    arrow(ax, 4.79, 5.61, 5.15, 5.22)
    arrow(ax, 4.79, 4.41, 5.15, 4.74)

    rounded(ax, 6.98, 4.34, 1.16, 1.30, fc="#F5DFC9", ec="#B96A32", lw=1.35)
    label(ax, 7.56, 5.18, "Hierarchical", size=8.1, weight="bold")
    label(ax, 7.56, 4.90, "fusion", size=8.1, weight="bold")
    label(ax, 7.56, 4.62, "all snapshots", size=6.9, color=MUTED)
    arrow(ax, 6.63, 4.99, 6.95, 4.99)

    rounded(ax, 8.42, 4.34, 0.72, 1.30, fc="white", ec=NAVY, lw=1.35)
    label(ax, 8.78, 5.20, "Node", size=7.6, weight="bold")
    label(ax, 8.78, 4.94, "trajectories", size=7.1)
    for i in range(4):
        ax.add_patch(Circle((8.58 + i * 0.13, 4.61 + (i % 2) * 0.08),
                            0.045, facecolor="#8AB6D6", edgecolor=NAVY,
                            lw=0.55, zorder=5))
    arrow(ax, 8.14, 4.99, 8.39, 4.99)

    label(ax, 0.72, 3.47, "Source objectives", size=8.3,
          weight="bold", ha="left", color="#8C4A20")
    ax.plot([0.72, 9.02], [3.29, 3.29], color="#DCC1AB", lw=0.9)

    rounded(ax, 0.82, 1.22, 2.22, 1.54, fc="white", ec="#B96A32", lw=1.05)
    label(ax, 1.93, 2.39, "Supervised detection", size=8.3, weight="bold")
    label(ax, 1.93, 2.03, r"$L_{\mathrm{det}}$", size=12.0,
          weight="bold", color="#9F4C1B")
    label(ax, 1.93, 1.60, "source-label BCE", size=7.0, color=MUTED)

    rounded(ax, 3.46, 1.22, 2.22, 1.54, fc="white", ec="#69A85F", lw=1.05)
    draw_star(ax, 4.02, 2.05, r=0.20)
    label(ax, 4.72, 2.39, "Single normal prototype", size=8.2, weight="bold")
    label(ax, 4.72, 2.03, r"$L_{\mathrm{proto}}$", size=10.2, weight="bold", color="#4F8E47")
    label(ax, 4.57, 1.60, "normal compactness", size=7.0, color=MUTED)

    rounded(ax, 6.10, 1.22, 2.72, 1.54, fc="white", ec=NAVY, lw=1.05)
    label(ax, 7.46, 2.39, "Temporal consistency", size=8.3, weight="bold")
    label(ax, 7.46, 2.03, r"$L_{\mathrm{self}}$", size=10.2, weight="bold", color=NAVY)
    label(ax, 7.46, 1.67, "normal smoothness + anomaly", size=6.9, color=MUTED)
    label(ax, 7.46, 1.47, "volatility separation", size=6.9, color=MUTED)

    arrow(ax, 8.78, 4.31, 8.78, 3.28, color="#6E7C88", lw=1.0,
          ls="dashed")
    arrow(ax, 8.78, 3.28, 7.46, 2.79, color="#6E7C88", lw=1.0,
          ls="dashed", connection="arc3,rad=0.05")
    arrow(ax, 8.78, 3.28, 4.57, 2.79, color="#6E7C88", lw=1.0,
          ls="dashed", connection="arc3,rad=0.04")
    arrow(ax, 8.78, 3.28, 1.93, 2.79, color="#6E7C88", lw=1.0,
          ls="dashed", connection="arc3,rad=0.03")
    rounded(ax, 2.68, 0.62, 4.35, 0.34, fc="#FFF0E3", ec="none", lw=0,
            radius=0.08)
    label(ax, 4.86, 0.79,
          r"$\mathcal{L}=\mathcal{L}_{\mathrm{det}}+\lambda_{\mathrm{proto}}\mathcal{L}_{\mathrm{proto}}+\lambda_{\mathrm{self}}\mathcal{L}_{\mathrm{self}}$",
          size=8.1, weight="bold", color="#85451E")

    # (c) Target inference
    panel(ax, 9.72, 0.28, 6.00, 6.64,
          "Label-efficient target inference", "c", BLUE_2)
    badge(ax, 14.13, 6.33, "FROZEN BACKBONE", BLUE, color=NAVY, width=1.35)

    rounded(ax, 10.05, 5.45, 1.30, 0.88, fc="white", ec="#6E98B8", lw=1.15)
    draw_sequence(ax, 10.14, 5.68, mode="target", compact=True, show_time=False)
    rounded(ax, 11.48, 5.45, 1.52, 0.88, fc=GRAY, ec=NAVY, lw=1.15, ls="dashed")
    label(ax, 12.24, 5.97, "Pretrained SPAD", size=8.2, weight="bold")
    label(ax, 12.24, 5.70, "GNN + TCN + fusion", size=6.9, color=MUTED)
    rounded(ax, 13.39, 5.45, 1.90, 0.88, fc="white", ec=NAVY, lw=1.15)
    label(ax, 14.34, 5.98, "Target node trajectories", size=8.0, weight="bold")
    label(ax, 14.34, 5.69, r"$z_i^1,\ldots,z_i^T$", size=8.0, color=NAVY)
    arrow(ax, 11.15, 5.89, 11.45, 5.89)
    arrow(ax, 13.00, 5.89, 13.36, 5.89)

    # Zero-shot branch
    rounded(ax, 10.03, 3.29, 5.38, 1.72, fc="#E6F1F9", ec="#6E98B8", lw=1.0)
    label(ax, 10.25, 4.80, "Zero-shot scoring", size=8.8, weight="bold",
          ha="left", color=NAVY)
    badge(ax, 13.96, 4.69, "NO TARGET LABELS", "#D4E8F5", color=NAVY, width=1.18)

    zs = [(10.22, "Dynamic\nneighbourhood\ninconsistency"),
          (11.48, "Recency\nweights\n1, ..., T"),
          (12.74, "Empirical\npercentile\ncalibration"),
          (14.03, "Zero-shot\nanomaly\nscore")]
    widths = [1.00, 1.00, 1.00, 1.10]
    for (x, txt), w in zip(zs, widths):
        rounded(ax, x, 3.57, w, 0.86, fc="white", ec=NAVY, lw=0.9,
                radius=0.09)
        label(ax, x + w / 2, 4.00, txt, size=6.9,
              weight="bold" if x == 14.03 else "normal")
    for x1, x2 in [(11.22, 11.45), (12.48, 12.71), (13.74, 14.00)]:
        arrow(ax, x1, 4.00, x2, 4.00, lw=0.9, mutation=8)
    label(ax, 12.72, 3.40,
          r"$r_i=\mathrm{ECDF}\!\left(\sum_t t\,\delta_i^t\,/\,\sum_t t\right)$",
          size=6.8, color=NAVY)
    arrow(ax, 14.34, 5.42, 12.72, 5.04, color="#6E7C88", lw=1.0,
          connection="arc3,rad=-0.10")

    # Few-shot branch
    rounded(ax, 10.03, 0.58, 5.38, 2.35, fc=GREEN_2, ec="#78AA73", lw=1.0)
    label(ax, 10.25, 2.72, "Optional parameter-efficient few-shot adaptation",
          size=8.5, weight="bold", ha="left", color="#477C42")
    badge(ax, 13.97, 2.61, "513 TRAINABLE PARAMS", GREEN, color="#477C42", width=1.22)

    rounded(ax, 10.22, 1.20, 0.96, 0.96, fc="white", ec="#78AA73", lw=0.9)
    label(ax, 10.70, 1.94, "Sparse target", size=6.9, weight="bold")
    label(ax, 10.70, 1.72, "support set", size=6.9, weight="bold")
    for i, fc in enumerate(["#CFECCB", "#F28B82", "#CFECCB", "#F28B82"]):
        ax.add_patch(Circle((10.47 + i * 0.16, 1.42), 0.055,
                            facecolor=fc,
                            edgecolor=RED if i % 2 else "#68A95E",
                            lw=0.65, zorder=5))

    rounded(ax, 11.47, 1.20, 1.05, 0.96, fc="#E6F1E5", ec="#5D9657", lw=1.05)
    label(ax, 11.995, 1.89, "Residual", size=7.2, weight="bold")
    label(ax, 11.995, 1.66, "adapter", size=7.2, weight="bold")
    label(ax, 11.995, 1.40, "8-D bottleneck", size=6.2, color=MUTED)

    rounded(ax, 12.82, 1.20, 1.08, 0.96, fc="white", ec="#5D9657", lw=0.95)
    draw_star(ax, 13.08, 1.76, r=0.12)
    draw_star(ax, 13.63, 1.76, r=0.12, fc="#F4B5B1", ec=RED)
    label(ax, 13.36, 1.40, "normal / anomaly", size=6.4, color=MUTED)

    rounded(ax, 14.20, 1.20, 0.96, 0.96, fc="#E6F1E5", ec="#5D9657", lw=1.05)
    label(ax, 14.68, 1.88, "Bounded", size=6.9, weight="bold")
    label(ax, 14.68, 1.66, "score correction", size=6.5, weight="bold")
    label(ax, 14.68, 1.40, r"$s_i^{\mathrm{few}}$", size=7.3,
          color="#477C42", weight="bold")

    arrow(ax, 11.18, 1.68, 11.44, 1.68, color="#5D9657", lw=0.95, mutation=8)
    arrow(ax, 12.52, 1.68, 12.79, 1.68, color="#5D9657", lw=0.95, mutation=8)
    arrow(ax, 13.90, 1.68, 14.17, 1.68, color="#5D9657", lw=0.95, mutation=8)

    # Stable-majority prior and base-score paths.
    label(ax, 13.37, 2.36, "stable-majority target normal prior", size=6.5,
          color="#477C42")
    arrow(ax, 14.28, 5.42, 13.45, 2.18, color="#5D9657", lw=0.9,
          ls="dashed", connection="arc3,rad=0.10")
    arrow(ax, 14.58, 3.55, 14.73, 2.18, color="#5D9657", lw=0.9,
          ls="dashed", connection="arc3,rad=0.03")
    label(ax, 14.97, 2.39, "base zero-shot score", size=6.1,
          color=MUTED)

    # Cross-panel links.
    arrow(ax, 13.40, 7.53, 1.23, 6.95, color="#B96A32", lw=1.15,
          connection="arc3,rad=0.02")
    label(ax, 7.10, 7.05, "source", size=7.1, color="#A55322")
    arrow(ax, 13.92, 7.53, 10.60, 6.95, color="#6E98B8", lw=1.15,
          connection="arc3,rad=-0.02")
    label(ax, 12.24, 7.05, "target", size=7.1, color=NAVY)

    label(ax, 0.32, 0.08,
          "Solid modules are optimized on the labeled source domain; the target backbone remains frozen.",
          size=6.8, color=MUTED, ha="left")

    fig.subplots_adjust(left=0.012, right=0.988, top=0.988, bottom=0.028)
    for ext, kwargs in {
        ".pdf": {"bbox_inches": "tight", "pad_inches": 0.04},
        ".svg": {"bbox_inches": "tight", "pad_inches": 0.04},
        ".png": {"bbox_inches": "tight", "pad_inches": 0.04, "dpi": 300},
    }.items():
        fig.savefig(f"{STEM}{ext}", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
