"""Build analysis/data/1_epoch_evaluation.csv from *_foe1_0.jsonl evaluation files.

One row per (run_id, epoch, split, pest).
pest values: "all" (aggregate), "mouse", "rat", "cockroach".

Metric coverage
---------------
pest="all"  -> coco_metrics + aggregate project_metrics + any_pest frame_presence_metrics
pest=class  -> per_class project_metrics + per_class frame_presence_metrics (coco cols NaN)

Config coverage (from run report .json + runs/{run_id}/last/config.json)
------------------------------------------------------------------------
Hyperparams  : eval_threshold, score_threshold, learning_rate, batch_size, weight_decay,
               freeze_backbone, partial_freeze, augment, foe, epochs_requested
Model        : model_base, freeze_strategy
Architecture : activation_function, activation_dropout, attention_dropout, dropout,
               num_queries, d_model, encoder_layers, decoder_layers,
               encoder_attention_heads, decoder_attention_heads, auxiliary_loss, backbone
Environment  : device_type, torch_version
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# On HPC the artefacts live under /cwork; locally they live next to this file.
HPC_EVAL_DIR = Path("/cwork/ad641/pest_detection_model/evaluations")
HPC_MODEL_DIR = Path("/cwork/ad641/pest_detection_model")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
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
    """Return the model repo root from the run report or fall back to HPC default."""
    model_repo = run_report.get("model", {}).get("model_repo_dir")
    if model_repo and Path(model_repo).exists():
        return Path(model_repo)
    if HPC_MODEL_DIR.exists():
        return HPC_MODEL_DIR
    return None

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_run_report(eval_dir: Path, run_id: str, timestamp: str) -> dict:
    """Load the companion run-report JSON for a given run."""
    path = eval_dir / f"{timestamp}_{run_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def load_arch_config(run_report: dict, run_id: str) -> dict:
    """Load the HuggingFace config.json from the run's last/ checkpoint directory."""
    # Prefer the path recorded in the run report (works across machines if mounted)
    last_dir = run_report.get("model", {}).get("last_dir")
    if last_dir:
        cfg_path = Path(last_dir) / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f)

    # Fall back to model_repo_dir / runs / run_id / last / config.json
    model_dir = resolve_model_dir(run_report)
    if model_dir:
        cfg_path = model_dir / "runs" / run_id / "last" / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f)
    return {}

def extract_config_fields(run_report: dict, arch: dict) -> dict:
    hp = run_report.get("hyperparams", {})
    model_meta = run_report.get("model", {})
    env = run_report.get("environment", {})

    return {
        # Eval / thresholding
        "eval_threshold":        hp.get("eval_threshold"),
        "postprocess_threshold": hp.get("postprocess_threshold"),   # may be absent
        # Optimiser
        "learning_rate":         hp.get("learning_rate"),
        "batch_size":            hp.get("batch_size"),
        "weight_decay":          hp.get("weight_decay"),
        # Freeze strategy
        "freeze_backbone":       hp.get("freeze_backbone"),
        "partial_freeze":        hp.get("partial_freeze"),         # number of stages frozen
        "freeze_strategy":       model_meta.get("strategy"),
        # Data / training setup
        "augment":               hp.get("augment"),
        "foe":                   hp.get("foe"),
        "epochs_requested":      hp.get("epochs_requested"),
        # Model identity
        "model_base":            model_meta.get("base_checkpoint"),
        # Architecture (from HF config.json)
        "backbone":              arch.get("backbone"),
        "activation_function":   arch.get("activation_function"),
        "activation_dropout":    arch.get("activation_dropout"),
        "attention_dropout":     arch.get("attention_dropout"),
        "dropout":               arch.get("dropout"),
        "num_queries":           arch.get("num_queries"),
        "d_model":               arch.get("d_model"),
        "encoder_layers":        arch.get("encoder_layers"),
        "decoder_layers":        arch.get("decoder_layers"),
        "encoder_attention_heads": arch.get("encoder_attention_heads"),
        "decoder_attention_heads": arch.get("decoder_attention_heads"),
        "auxiliary_loss":        arch.get("auxiliary_loss"),
        # Environment
        "device_type":           env.get("device_type"),
        "torch_version":         env.get("torch_version"),
    }

# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

CLASSES = ("mouse", "rat", "cockroach")

def row_base(run_id: str, timestamp: str, epoch: int, global_step: int, split: str, pest: str) -> dict:
    return {
        "run_id":      run_id,
        "timestamp":   timestamp,
        "epoch":       epoch,
        "global_step": global_step,
        "split":       split,
        "pest":        pest,
    }

def extract_coco(coco: dict) -> dict:
    return {
        "coco_mAP_0.5_0.95": coco.get("mAP_0.5_0.95"),
        "coco_mAP_0.5":      coco.get("mAP_0.5"),
        "coco_mAP_0.75":     coco.get("mAP_0.75"),
        "coco_AP_small":     coco.get("AP_small"),
        "coco_AR_max1":      coco.get("AR_max1"),
        "coco_AR_max10":     coco.get("AR_max10"),
        "coco_AR_max100":    coco.get("AR_max100"),
        "coco_AR_small":     coco.get("AR_small"),
    }

COCO_NULL = {k: None for k in extract_coco({}).keys()}

def extract_proj_agg(pm: dict) -> dict:
    return {
        "detection_rate":      pm.get("detection_rate"),
        "false_positive_rate": pm.get("false_positive_rate"),
        "precision":           pm.get("precision"),
        "recall":              pm.get("recall"),
        "f1":                  pm.get("f1"),
        "tp":                  pm.get("tp"),
        "fp":                  pm.get("fp"),
        "fn":                  pm.get("fn"),
        "fp_frames":           pm.get("fp_frames"),
        "tn_frames":           pm.get("tn_frames"),
        "detection_rate_pass": pm.get("detection_rate_pass"),
        "fpr_pass":            pm.get("fpr_pass"),
    }

def extract_proj_class(pc: dict) -> dict:
    return {
        "detection_rate":      None,
        "false_positive_rate": None,
        "precision":           pc.get("precision"),
        "recall":              pc.get("recall"),
        "f1":                  pc.get("f1"),
        "tp":                  pc.get("tp"),
        "fp":                  pc.get("fp"),
        "fn":                  pc.get("fn"),
        "fp_frames":           None,
        "tn_frames":           None,
        "detection_rate_pass": None,
        "fpr_pass":            None,
    }

def extract_presence(fp: dict) -> dict:
    return {
        "fp_threshold":  fp.get("threshold"),
        "fp_num_examples": fp.get("num_examples"),
        "fp_positive_examples": fp.get("positive_examples"),
        "fp_negative_examples": fp.get("negative_examples"),
        "fp_precision":  fp.get("precision"),
        "fp_recall":     fp.get("recall"),
        "fp_f1":         fp.get("f1"),
        "fp_auroc":      fp.get("auroc"),
        "fp_aupr":       fp.get("aupr"),
    }

PRESENCE_NULL = {k: None for k in extract_presence({}).keys()}

def build_rows_for_split(
    run_id: str, timestamp: str, epoch: int, global_step: int,
    split: str, eval_split: dict, config_fields: dict
) -> list[dict]:
    rows = []
    coco  = eval_split.get("coco_metrics", {})
    pm    = eval_split.get("project_metrics", {})
    fpm   = eval_split.get("frame_presence_metrics", {})
    pc    = pm.get("per_class", {})
    fpc   = fpm.get("per_class", {})

    # ---- pest = "all" -------------------------------------------------------
    r = row_base(run_id, timestamp, epoch, global_step, split, "all")
    r.update(extract_coco(coco))
    r.update(extract_proj_agg(pm))
    r.update(extract_presence(fpm.get("any_pest", {})))
    r.update(config_fields)
    rows.append(r)

    # ---- pest = per class ---------------------------------------------------
    for cls in CLASSES:
        r = row_base(run_id, timestamp, epoch, global_step, split, cls)
        r.update(COCO_NULL)
        r.update(extract_proj_class(pc.get(cls, {})))
        r.update(extract_presence(fpc.get(cls, {})))
        r.update(config_fields)
        rows.append(r)

    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    eval_dir = resolve_eval_dir()
    foe_files = sorted(eval_dir.glob("*_foe1_0.jsonl"))

    if not foe_files:
        print(f"No *_foe1_0.jsonl files found in {eval_dir}", file=sys.stderr)
        sys.exit(1)

    all_rows: list[dict] = []

    for foe_path in foe_files:
        # Derive run_id and timestamp from filename
        # Format: {timestamp}_{run_id}_foe1_0.jsonl  e.g. 20260407_044650_run_d6ec9003_foe1_0.jsonl
        stem = foe_path.stem  # strip .jsonl
        # strip trailing _foe1_0
        base = stem.replace("_foe1_0", "")
        # split into timestamp (first two parts: date_time) and run_id (rest)
        parts = base.split("_")
        timestamp = f"{parts[0]}_{parts[1]}"       # 20260407_044650
        run_id    = "_".join(parts[2:])             # run_d6ec9003

        print(f"Processing {run_id} ({timestamp}) ...")

        run_report  = load_run_report(eval_dir, run_id, timestamp)
        arch_config = load_arch_config(run_report, run_id)
        config_fields = extract_config_fields(run_report, arch_config)

        with open(foe_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row_json = json.loads(line)

                epoch       = row_json.get("epoch")
                global_step = row_json.get("global_step")
                evaluation  = row_json.get("evaluation", {})

                for split, eval_split in evaluation.items():
                    rows = build_rows_for_split(
                        run_id, timestamp, epoch, global_step,
                        split, eval_split, config_fields
                    )
                    all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Reorder: identifiers first, config last
    id_cols = ["run_id", "timestamp", "epoch", "global_step", "split", "pest"]
    metric_cols = [c for c in df.columns if c not in id_cols and c not in config_fields]
    config_cols = list(config_fields.keys())
    # keep only cols that actually exist
    ordered = [c for c in id_cols + metric_cols + config_cols if c in df.columns]
    df = df[ordered]

    out_dir  = SCRIPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "1_epoch_evaluation.csv"
    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df):,} rows x {len(df.columns)} columns → {out_path}")
    print(f"Runs: {df['run_id'].nunique()}  |  Epochs: {sorted(df['epoch'].unique())}  |  Splits: {sorted(df['split'].unique())}  |  Pests: {sorted(df['pest'].unique())}")

if __name__ == "__main__":
    main()
