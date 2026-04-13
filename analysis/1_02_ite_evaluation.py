"""Build analysis/data/1_02_ite_evaluation.csv from *_iterations.jsonl files.

One row per (run_id, epoch, global_step, batch_idx).

Columns
-------
Identifiers : run_id, timestamp, epoch, global_step, batch_idx
Metrics     : train_loss, lr_backbone, lr_head
Config      : learning_rate, batch_size, weight_decay, freeze_backbone,
              partial_freeze, freeze_strategy, augment, epochs_requested,
              model_base, device_type, torch_version
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution  (mirrors 1_01_epoch_evaluation.py)
# ---------------------------------------------------------------------------

HPC_EVAL_DIR  = Path("/cwork/ad641/pest_detection_model/evaluations")
HPC_MODEL_DIR = Path("/cwork/ad641/pest_detection_model")

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent
LOCAL_EVAL_DIR = PROJECT_ROOT / "evaluation"


def resolve_eval_dir() -> Path:
    if HPC_EVAL_DIR.exists():
        return HPC_EVAL_DIR
    if LOCAL_EVAL_DIR.exists():
        return LOCAL_EVAL_DIR
    raise FileNotFoundError(
        f"Cannot find evaluation directory. Tried:\n  {HPC_EVAL_DIR}\n  {LOCAL_EVAL_DIR}"
    )


def resolve_model_dir(run_report: dict) -> Path | None:
    model_repo = run_report.get("model", {}).get("model_repo_dir")
    if model_repo and Path(model_repo).exists():
        return Path(model_repo)
    if HPC_MODEL_DIR.exists():
        return HPC_MODEL_DIR
    return None

# ---------------------------------------------------------------------------
# Config loading  (mirrors 1_01_epoch_evaluation.py)
# ---------------------------------------------------------------------------

def load_run_report(eval_dir: Path, run_id: str, timestamp: str) -> dict:
    path = eval_dir / f"{timestamp}_{run_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def extract_config_fields(run_report: dict) -> dict:
    hp         = run_report.get("hyperparams", {})
    model_meta = run_report.get("model", {})
    env        = run_report.get("environment", {})
    return {
        "learning_rate":    hp.get("learning_rate"),
        "batch_size":       hp.get("batch_size"),
        "weight_decay":     hp.get("weight_decay"),
        "freeze_backbone":  hp.get("freeze_backbone"),
        "partial_freeze":   hp.get("partial_freeze"),
        "freeze_strategy":  model_meta.get("strategy"),
        "augment":          hp.get("augment"),
        "epochs_requested": hp.get("epochs_requested"),
        "model_base":       model_meta.get("base_checkpoint"),
        "device_type":      env.get("device_type"),
        "torch_version":    env.get("torch_version"),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    eval_dir   = resolve_eval_dir()
    iter_files = sorted(eval_dir.glob("*_iterations.jsonl"))

    if not iter_files:
        print(f"No *_iterations.jsonl files found in {eval_dir}", file=sys.stderr)
        sys.exit(1)

    all_rows: list[dict] = []

    for iter_path in iter_files:
        stem  = iter_path.stem                        # e.g. 20260407_044650_run_d6ec9003_iterations
        base  = stem.replace("_iterations", "")
        parts = base.split("_")
        timestamp = f"{parts[0]}_{parts[1]}"
        run_id    = "_".join(parts[2:])

        foe_files = list(eval_dir.glob(f"{timestamp}_{run_id}_foe*.jsonl"))
        if not foe_files:
            print(f"Skipping {run_id} ({timestamp}) — no foe*.jsonl companion found")
            continue

        print(f"Processing {run_id} ({timestamp}) ...")

        run_report    = load_run_report(eval_dir, run_id, timestamp)
        config_fields = extract_config_fields(run_report)

        with open(iter_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row_json = json.loads(line)

                lr_groups = row_json.get("lr_groups", [])
                row = {
                    "run_id":       run_id,
                    "timestamp":    timestamp,
                    "epoch":        row_json.get("epoch"),
                    "global_step":  row_json.get("global_step"),
                    "batch_idx":    row_json.get("batch_idx"),
                    "train_loss":   row_json.get("train_loss"),
                    "lr_backbone":  lr_groups[0] if len(lr_groups) > 0 else None,
                    "lr_head":      lr_groups[1] if len(lr_groups) > 1 else None,
                }
                row.update(config_fields)
                all_rows.append(row)

    df = pd.DataFrame(all_rows)

    out_dir  = SCRIPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "1_02_ite_evaluation.csv"
    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df):,} rows x {len(df.columns)} columns → {out_path}")
    print(f"Runs: {df['run_id'].nunique()}  |  Epochs: {sorted(df['epoch'].unique())}  |  Steps: {df['global_step'].max()}")


if __name__ == "__main__":
    main()
