"""Plot train loss across iterations — one line per freeze strategy.

X-axis : global_step (iteration index across all epochs)
Y-axis : train_loss
Legend : strategy label + batch size

Output (analysis/output/):
    fig_ite_loss.png
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
DATA_PATH  = SCRIPT_DIR / "data" / "1_02_ite_evaluation.csv"
OUT_DIR    = SCRIPT_DIR / "output"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
df = df.sort_values(["run_id", "global_step"])

# ---------------------------------------------------------------------------
# Style  (matches 2_epoch_analysis.py)
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

STRATEGY_ORDER = ["partial_4stages", "partial_2stages", "head_only"]
STRATEGY_LABELS = {
    "partial_4stages": "Partial freeze (4 stages)",
    "partial_2stages": "Partial freeze (2 stages)",
    "head_only":       "Head only",
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 4.3))
fig.suptitle("Train Loss vs. Iteration", fontsize=13, y=1.02)

handles, labels = [], []

runs_sorted = sorted(
    df.groupby("run_id"),
    key=lambda x: (
        STRATEGY_ORDER.index(x[1]["freeze_strategy"].iloc[0])
        if x[1]["freeze_strategy"].iloc[0] in STRATEGY_ORDER
        else len(STRATEGY_ORDER)
    ),
)

for run_id, group in runs_sorted:
    strategy   = group["freeze_strategy"].iloc[0]
    batch_size = int(group["batch_size"].iloc[0])
    strat_label = STRATEGY_LABELS.get(strategy, strategy)
    label = f"{strat_label}  (batch {batch_size})"

    line, = ax.plot(
        group["global_step"], group["train_loss"],
        linewidth=1.2, alpha=0.85, zorder=3,
    )
    handles.append(line)
    labels.append(label)

ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Train Loss", fontsize=11)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.grid(False)

fig.legend(handles, labels, fontsize=9, frameon=False,
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14))

plt.tight_layout()
out_path = OUT_DIR / "fig_ite_loss.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")
