"""Helpers for evaluation artifact naming and report persistence."""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "run"


def make_run_artifacts(experiment_name: str, evaluation_dir: str | None = None, foe: float = 1.0) -> dict:
    """Create stable artifact paths for one workflow run.

    Produces three files:
        <stem>.json                  -- run metadata, hyperparams, training summary, final eval
        <stem>_foe<foe>.jsonl        -- mid-training eval on 30 random images per split at each foe fraction
        <stem>_iterations.jsonl      -- per-batch loss log
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    target_dir = Path(evaluation_dir) if evaluation_dir else PROJECT_ROOT / "evaluation"
    target_dir.mkdir(parents=True, exist_ok=True)
    foe_str = str(foe).replace(".", "_")
    stem = f"{timestamp}_{run_id}"

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "evaluation_dir": str(target_dir),
        "run_report_path": str(target_dir / f"{stem}.json"),
        "foe_path": str(target_dir / f"{stem}_foe{foe_str}.jsonl"),
        "iterations_path": str(target_dir / f"{stem}_iterations.jsonl"),
    }


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def update_json(path: str | Path, updates: dict) -> None:
    """Load an existing JSON file, merge updates into it, and save."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def make_model_repo_layout(model_repo_dir: str | Path, run_id: str) -> dict:
    """Create the model artifact layout inside a local HF model repo clone."""
    root = Path(model_repo_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs_root = root / "runs"
    run_root = runs_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    best_dir = run_root / "best"
    last_dir = run_root / "last"

    return {
        "root": root,
        "runs_root": runs_root,
        "run_root": run_root,
        "best_dir": best_dir,
        "last_dir": last_dir,
        "best_state_path": root / "best_mdl_state.jsonl",
        "last_state_path": root / "last_mdl_state.jsonl",
    }


def save_model_bundle(
    target_dir: str | Path,
    model,
    processor,
    training_state: dict | None = None,
) -> Path:
    """Persist a model, processor, and optional trainer state to one directory."""
    target_dir = Path(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(target_dir)
    processor.save_pretrained(target_dir)
    if training_state is not None:
        torch.save(training_state, target_dir / "trainer_state.pt")
    return target_dir


def try_git_commit(project_root: str | Path | None = None) -> str | None:
    project_root = Path(project_root) if project_root else PROJECT_ROOT
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def build_environment_metadata(device: torch.device, num_workers: int) -> dict:
    cuda_version = getattr(torch.version, "cuda", None)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "device_type": getattr(device, "type", str(device)),
        "num_workers": num_workers,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_version,
    }
