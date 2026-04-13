"""Plot precision, recall, and F1 across epochs — one figure per pest + overall.

Each figure is a 1x3 grid (Precision | Recall | F1) with a shared legend.
Uses split="test" rows from 1_epoch_evaluation.csv.

Output (analysis/output/):
    fig_ep_prc_tot.png        -- pest="all"
    fig_ep_prc_mouse.png      -- pest="mouse"
    fig_ep_prc_rat.png        -- pest="rat"
    fig_ep_prc_cockroach.png  -- pest="cockroach"
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH  = SCRIPT_DIR / "data" / "1_epoch_evaluation.csv"
OUT_DIR    = SCRIPT_DIR / "output"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df = df[df["split"] == "test"].copy()
for col in ("precision", "recall", "f1"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.sort_values(["run_id", "epoch"])

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "serif",
    "mathtext.fontset":  "dejavuserif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "axes.prop_cycle":   plt.cycler(color=["#0072B2", "#D55E00", "#009E73"]),
})

METRICS = [
    ("precision", "Precision"),
    ("recall",    "Recall"),
    ("f1",        "F1 Score"),
]

FIGURES = [
    ("all",       "Overall",   "fig_ep_prc_tot"),
    ("mouse",     "Mouse",     "fig_ep_prc_mouse"),
    ("rat",       "Rat",       "fig_ep_prc_rat"),
    ("cockroach", "Cockroach", "fig_ep_prc_cockroach"),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)

for pest, title, fname in FIGURES:
    subset = df[df["pest"] == pest]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    fig.suptitle(f"{title} — Precision, Recall & F1 vs. Epoch (test split)",
                 fontsize=13, y=1.02)

    STRATEGY_ORDER = ["partial_4stages", "partial_2stages", "head_only"]
    STRATEGY_LABELS = {
        "partial_4stages": "Partial freeze (4 stages)",
        "partial_2stages": "Partial freeze (2 stages)",
        "head_only":       "Head only",
    }

    handles, labels = [], []

    for ax, (metric, metric_label) in zip(axes, METRICS):
        runs_sorted = sorted(
            subset.groupby("run_id"),
            key=lambda x: STRATEGY_ORDER.index(x[1]["freeze_strategy"].iloc[0])
        )
        for run_id, group in runs_sorted:
            strategy = group["freeze_strategy"].iloc[0]
            label = STRATEGY_LABELS.get(strategy, strategy)
            line, = ax.plot(
                group["epoch"], group[metric],
                linewidth=1.6, marker="none", zorder=3,
            )
            if ax is axes[0]:
                handles.append(line)
                labels.append(label)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(False)
        ax.set_aspect("auto")
        ax.set_box_aspect(1)

    fig.legend(handles, labels, fontsize=9, frameon=False,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout()
    out_path = OUT_DIR / f"{fname}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
