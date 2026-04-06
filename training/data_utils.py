"""Dataset path resolution helpers for local and HF-downloaded COCO repos."""

from __future__ import annotations

import json
from pathlib import Path


_SPLIT_CANDIDATES = {
    "train": {
        "image_dirs": ["train"],
        "annotation_files": ["train.json"],
    },
    "val": {
        "image_dirs": ["val", "validation"],
        "annotation_files": ["val.json", "validation.json"],
    },
    "validation": {
        "image_dirs": ["val", "validation"],
        "annotation_files": ["val.json", "validation.json"],
    },
    "test": {
        "image_dirs": ["test"],
        "annotation_files": ["test.json"],
    },
}


def normalize_split_name(split: str) -> str:
    split = split.lower().strip()
    if split == "validation":
        return "val"
    return split


def resolve_split_paths(data_root: str | Path, split: str) -> dict:
    """Resolve image and annotation paths for a split.

    Supports both `val` and `validation` naming conventions so local datasets and
    Hub-downloaded datasets can share the same workflow.
    """
    data_root = Path(data_root)
    split_key = split.lower().strip()
    if split_key not in _SPLIT_CANDIDATES:
        raise ValueError(f"Unsupported split: {split}")

    image_dir = None
    annotation_path = None

    for name in _SPLIT_CANDIDATES[split_key]["image_dirs"]:
        candidate = data_root / "images" / name
        if candidate.exists():
            image_dir = candidate
            break

    for name in _SPLIT_CANDIDATES[split_key]["annotation_files"]:
        candidate = data_root / "annotations" / name
        if candidate.exists():
            annotation_path = candidate
            break

    if image_dir is None or annotation_path is None:
        image_candidates = [
            str(data_root / "images" / name)
            for name in _SPLIT_CANDIDATES[split_key]["image_dirs"]
        ]
        ann_candidates = [
            str(data_root / "annotations" / name)
            for name in _SPLIT_CANDIDATES[split_key]["annotation_files"]
        ]
        raise FileNotFoundError(
            "Could not resolve split paths for "
            f"'{split}'. Tried image dirs {image_candidates} and annotations {ann_candidates}."
        )

    return {
        "requested_split": split,
        "canonical_split": normalize_split_name(split),
        "image_dir": image_dir,
        "annotation_path": annotation_path,
        "resolved_image_split": image_dir.name,
        "resolved_annotation_file": annotation_path.name,
    }


def _load_annotation_summary(annotation_path: Path) -> dict:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    return {
        "num_images": len(data.get("images", [])),
        "num_annotations": len(data.get("annotations", [])),
        "categories": categories,
        "category_names": [c.get("name", str(c.get("id"))) for c in categories],
    }


def collect_dataset_metadata(data_root: str | Path) -> dict:
    """Collect lightweight dataset metadata for reporting."""
    data_root = Path(data_root)
    metadata = {
        "root": str(data_root),
        "available_splits": [],
        "splits": {},
        "class_names": [],
    }
    category_names = None

    for split in ("train", "val", "test"):
        try:
            resolved = resolve_split_paths(data_root, split)
        except FileNotFoundError:
            continue

        summary = _load_annotation_summary(resolved["annotation_path"])
        metadata["available_splits"].append(split)
        metadata["splits"][split] = {
            "image_dir": str(resolved["image_dir"]),
            "annotation_path": str(resolved["annotation_path"]),
            "resolved_image_split": resolved["resolved_image_split"],
            "resolved_annotation_file": resolved["resolved_annotation_file"],
            "num_images": summary["num_images"],
            "num_annotations": summary["num_annotations"],
        }
        if category_names is None:
            category_names = summary["category_names"]

    metadata["class_names"] = category_names or []
    return metadata


def infer_hf_revision_from_cache_path(data_root: str | Path) -> str | None:
    """Best-effort extraction of the HF snapshot revision from a cache path."""
    parts = Path(data_root).parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]
