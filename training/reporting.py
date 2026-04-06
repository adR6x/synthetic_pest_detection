"""Helpers for evaluation artifact naming and report persistence."""

from __future__ import annotations

import json
import platform
import re
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


def make_run_artifacts(experiment_name: str, evaluation_dir: str | None = None) -> dict:
    """Create stable artifact paths for one workflow run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    slug = slugify(experiment_name)
    target_dir = Path(evaluation_dir) if evaluation_dir else PROJECT_ROOT / "evaluation"
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamp}_{slug}_{run_id}"

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "evaluation_dir": str(target_dir),
        "report_path": str(target_dir / f"{stem}.json"),
        "iteration_log_path": str(target_dir / f"{stem}_iterations.jsonl"),
    }


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


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
