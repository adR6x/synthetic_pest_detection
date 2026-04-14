"""Standalone CLI for the real video generator — no Flask required.

Replicates exactly what the web app's Real Generator tab does:
  - reads curated kitchen images + train/test split
  - calls generate_video() for each kitchen
  - writes frames to pest_detection_dataset/images/{split}/{job_id}/
  - appends to pest_detection_dataset/annotations/{split}.json
  - appends to pest_detection_dataset/generated_state.json

Usage
-----
    python -m generator.generate                            # 10 videos, 24s @ 10fps, auto workers
    python -m generator.generate --count 50
    python -m generator.generate --count 100 --workers 8
    python -m generator.generate --count 50 --workers 2 --fps 10
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Inject the mmcv stub so Metric3D can load without a real mmcv installation.
_STUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mmcv_stub")
if _STUB not in sys.path:
    sys.path.insert(0, _STUB)

# Ensure timm is available — Metric3D's ConvNeXt backbone requires it.
try:
    import timm  # noqa: F401
except ImportError:
    import subprocess
    print("Installing missing dependency: timm ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm", "--quiet"])

from generator.pipeline import generate_video
from generator.depth_estimator import preload_models

# ---------------------------------------------------------------------------
# Paths — mirrors app/main.py logic exactly
# ---------------------------------------------------------------------------
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KITCHEN_ROOT  = os.path.join(_PROJECT_ROOT, "generator", "kitchen_img")
CURATED_IMG_DIR       = os.path.join(_KITCHEN_ROOT, "curated_img")
TRAIN_TEST_SPLIT_PATH = os.path.join(_KITCHEN_ROOT, "tain_split.csv")

if os.getcwd().startswith("/hpc"):
    HF_DATASET_DIR = "/cwork/ad641/pest_detection_dataset"
else:
    HF_DATASET_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "..", "pest_detection_dataset"))

GENERATED_STATE_PATH   = os.path.join(HF_DATASET_DIR, "generated_state.json")
REAL_TRAIN_FRAME_STRIDE = 5

_HF_DEFAULT_CATEGORIES = [
    {"id": 1, "name": "mouse",     "supercategory": "pest"},
    {"id": 2, "name": "rat",       "supercategory": "pest"},
    {"id": 3, "name": "cockroach", "supercategory": "pest"},
]

_real_state_lock = threading.Lock()
_HF_SPLIT_LOCKS  = {
    "train": threading.Lock(),
    "val":   threading.Lock(),
    "test":  threading.Lock(),
}
_TEMP_LABELS_DIR = tempfile.mkdtemp(prefix="pest_gen_labels_")

# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

_progress_lock = threading.Lock()
_progress = {
    "done": 0,
    "total": 0,
    "generated": 0,
    "failures": 0,
    "start_time": 0.0,
}


def _bar(done, total, width=30):
    filled = int(width * done / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_progress(row=None, error=None):
    with _progress_lock:
        p = _progress
        elapsed   = time.time() - p["start_time"]
        rate      = p["done"] / elapsed if elapsed > 0 else 0
        remaining = (p["total"] - p["done"]) / rate if rate > 0 else 0
        bar       = _bar(p["done"], p["total"])

        # Build status line
        status = (
            f"\r{bar} {p['done']}/{p['total']}  "
            f"ok={p['generated']} fail={p['failures']}  "
            f"elapsed={elapsed:.0f}s  eta={remaining:.0f}s"
        )
        sys.stdout.write(status)
        sys.stdout.flush()

        # Print detail for completed/failed jobs on a new line above the bar
        if row or error:
            sys.stdout.write("\n")
            if row:
                pests = (f"mouse={row['mouse_count']} "
                         f"rat={row['rat_count']} "
                         f"roach={row['cockroach_count']}")
                print(f"  ✓ {row['video_id']}  split={row['split']}  "
                      f"{pests}  ({row['time_taken_to_generate_seconds']}s)")
            if error:
                print(f"  ✗ {error}")
            sys.stdout.write(status)
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kitchen_image_id(filename):
    return os.path.splitext(filename)[0]


def _load_train_test_split():
    if not os.path.exists(TRAIN_TEST_SPLIT_PATH):
        raise FileNotFoundError(
            f"Missing split file: {TRAIN_TEST_SPLIT_PATH}\n"
            "Generate it with: python generator/kitchen_img/test_train_split.py"
        )
    split_map = {}
    with open(TRAIN_TEST_SPLIT_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        def _norm(kid):
            # strip extension so kitchen_0001.jpg and kitchen_0001 both match
            return os.path.splitext(kid)[0] if kid else kid

        if fieldnames == ["id", "train", "val", "test"]:
            for row in reader:
                kid = _norm((row.get("id") or "").strip())
                if not kid:
                    continue
                if row.get("train", "0").strip() == "1":
                    split_map[kid] = "train"
                elif row.get("val", "0").strip() == "1":
                    split_map[kid] = "val"
                elif row.get("test", "0").strip() == "1":
                    split_map[kid] = "test"
        elif fieldnames == ["id", "train"]:
            for row in reader:
                kid  = _norm((row.get("id") or "").strip())
                flag = (row.get("train") or "").strip()
                if kid:
                    split_map[kid] = "train" if flag == "1" else "test"
        else:
            raise ValueError(
                f"{TRAIN_TEST_SPLIT_PATH} must have columns [id, train, val, test] "
                f"or legacy [id, train]. Got: {fieldnames}"
            )
    return split_map


def _real_worker_count(target_count):
    """Cap workers at allocated SLURM cores (if on HPC), else min(count, cpu_count, 4)."""
    slurm_cores = (
        int(os.environ.get("SLURM_CPUS_PER_TASK") or
            os.environ.get("SLURM_JOB_CPUS_PER_NODE") or 0)
    )
    cpu = slurm_cores if slurm_cores > 0 else max(1, os.cpu_count() or 1)
    return max(1, min(target_count, cpu))


def _output_roots_for_split(split_name):
    return {
        "frames_root": os.path.join(HF_DATASET_DIR, "images", split_name),
        "labels_root": os.path.join(_TEMP_LABELS_DIR, split_name, "labels"),
        "videos_root": os.path.join(_TEMP_LABELS_DIR, split_name, "videos"),
    }


def _load_generated_state():
    if not os.path.exists(GENERATED_STATE_PATH):
        return {"generated_videos": []}
    try:
        with open(GENERATED_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"generated_videos": []}
    if isinstance(data, list):
        return {"generated_videos": data}
    return {"generated_videos": data.get("generated_videos", [])}


def _save_generated_state(state):
    os.makedirs(os.path.dirname(GENERATED_STATE_PATH), exist_ok=True)
    payload = {
        "generated_videos": list(state.get("generated_videos", [])),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(GENERATED_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_generated_state_rows(rows):
    if not rows:
        return
    with _real_state_lock:
        state = _load_generated_state()
        state["generated_videos"].extend(rows)
        _save_generated_state(state)


def _append_to_hf_dataset(row):
    if not os.path.isdir(HF_DATASET_DIR):
        return
    job_id   = row["job_id"]
    split    = row["split"]
    ann_lock = _HF_SPLIT_LOCKS.get(split)
    if ann_lock is None:
        return

    hf_frames_dir = os.path.join(HF_DATASET_DIR, "images", split, job_id)
    if not os.path.isdir(hf_frames_dir):
        return
    frame_fnames = sorted(
        f for f in os.listdir(hf_frames_dir)
        if re.match(r"^frame_\d+\.(png|jpg|jpeg|webp)$", f, re.IGNORECASE)
    )
    if not frame_fnames:
        return

    per_job_ann_path = os.path.join(
        _output_roots_for_split(split)["labels_root"], job_id, "annotations.json"
    )
    per_job_coco = {"images": [], "annotations": [], "categories": []}
    if os.path.exists(per_job_ann_path):
        with open(per_job_ann_path, "r", encoding="utf-8") as f:
            per_job_coco = json.load(f)

    local_fname_to_id = {img["file_name"]: img["id"] for img in per_job_coco.get("images", [])}
    anns_by_local_id  = {}
    for ann in per_job_coco.get("annotations", []):
        anns_by_local_id.setdefault(ann["image_id"], []).append(ann)

    hf_ann_path = os.path.join(HF_DATASET_DIR, "annotations", f"{split}.json")
    with ann_lock:
        if os.path.exists(hf_ann_path):
            with open(hf_ann_path, "r", encoding="utf-8") as f:
                hf_coco = json.load(f)
        else:
            os.makedirs(os.path.dirname(hf_ann_path), exist_ok=True)
            hf_coco = {"images": [], "annotations": [], "categories": list(_HF_DEFAULT_CATEGORIES)}

        img_id = max((img["id"] for img in hf_coco.get("images", [])), default=0) + 1
        ann_id = max((a["id"]  for a  in hf_coco.get("annotations", [])), default=0) + 1

        for fname in frame_fnames:
            local_img_id = local_fname_to_id.get(fname)
            hf_coco["images"].append({
                "id":        img_id,
                "file_name": f"{job_id}/{fname}",
                "width":     640,
                "height":    480,
            })
            for ann in anns_by_local_id.get(local_img_id, []):
                hf_coco["annotations"].append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": ann["category_id"],
                    "bbox":        ann["bbox"],
                    "area":        ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                    "iscrowd":     ann.get("iscrowd", 0),
                })
                ann_id += 1
            img_id += 1

        if not hf_coco.get("categories"):
            hf_coco["categories"] = per_job_coco.get("categories") or list(_HF_DEFAULT_CATEGORIES)

        with open(hf_ann_path, "w", encoding="utf-8") as f:
            json.dump(hf_coco, f, indent=2)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _run_single(curated_filename, split_map, num_frames, fps, assemble_video):
    image_path = os.path.join(CURATED_IMG_DIR, curated_filename)
    kitchen_id = _kitchen_image_id(curated_filename)
    split_name = split_map.get(kitchen_id)
    if split_name not in {"train", "val", "test"}:
        raise ValueError(
            f"Kitchen {kitchen_id!r} not in split file. Regenerate the split file first."
        )

    roots          = _output_roots_for_split(split_name)
    length_seconds = num_frames / fps

    t0 = time.time()
    result = generate_video(
        image_path,
        frames_root=roots["frames_root"],
        labels_root=roots["labels_root"],
        videos_root=roots["videos_root"],
        num_frames=num_frames,
        fps=fps,
        assemble_video=assemble_video,
        frame_format="png",
        save_scene_previews=False,
        save_mask_previews=False,
        save_movement_masks=False,
        keep_only_frame_outputs=True,
        save_every_n=REAL_TRAIN_FRAME_STRIDE,
        keep_full_annotations=True,
    )
    elapsed     = round(time.time() - t0, 2)
    pest_counts = result.get("pest_counts") or {}
    video_id    = result.get("video_id") or result.get("job_id")

    return {
        "job_id":                         video_id,
        "video_id":                       video_id,
        "kitchen_id":                     kitchen_id,
        "split":                          split_name,
        "length_of_video_seconds":        round(length_seconds, 2),
        "fps":                            fps,
        "mouse_count":                    int(pest_counts.get("mouse", 0)),
        "rat_count":                      int(pest_counts.get("rat", 0)),
        "cockroach_count":                int(pest_counts.get("cockroach", 0)),
        "date_time_generated":            datetime.now().isoformat(timespec="seconds"),
        "time_taken_to_generate_seconds": elapsed,
        "pest_size_multiplier":           float(result.get("pest_size_multiplier", 1.0)),
        "pest_generation_metadata":       result.get("pest_generation_metadata") or [],
        "frames_dir":                     f"images/{split_name}/{video_id}",
        "labels_dir":                     f"annotations/{split_name}.json",
        "video_path":                     result.get("video_path"),
    }


def run_generation(count, fps, assemble_video, workers_override=None):
    all_curated = sorted(
        f for f in os.listdir(CURATED_IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ) if os.path.isdir(CURATED_IMG_DIR) else []

    if not all_curated:
        raise RuntimeError(f"No curated images found in {CURATED_IMG_DIR}")

    split_map  = _load_train_test_split()
    length_s   = 24.0                          # fixed — matches web app
    num_frames = max(1, round(length_s * fps))
    workers    = workers_override if workers_override else _real_worker_count(count)
    selected   = random.choices(all_curated, k=count)

    print(f"Dataset dir  : {HF_DATASET_DIR}")
    print(f"Curated imgs : {len(all_curated)}")
    print(f"Videos       : {count}  ({length_s}s @ {fps}fps = {num_frames} frames each)")
    print(f"Workers      : {workers}  (min(count={count}, cpus_allocated={os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_JOB_CPUS_PER_NODE') or os.cpu_count()}))")
    print(f"Assemble MP4 : False (matches web app default)")
    print()

    os.makedirs(HF_DATASET_DIR, exist_ok=True)

    print("Preloading Metric3D into GPU memory...")
    preload_models()
    print("Model ready.\n")

    _progress.update({
        "done": 0, "total": count,
        "generated": 0, "failures": 0,
        "start_time": time.time(),
    })
    _print_progress()

    rows = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_run_single, fn, split_map, num_frames, fps, assemble_video): fn
            for fn in selected
        }
        for future in as_completed(future_map):
            fn = future_map[future]
            with _progress_lock:
                _progress["done"] += 1
            try:
                row = future.result()
                with _progress_lock:
                    _progress["generated"] += 1
                rows.append(row)
                _append_generated_state_rows([row])
                _append_to_hf_dataset(row)
                _print_progress(row=row)
            except Exception as e:
                with _progress_lock:
                    _progress["failures"] += 1
                _print_progress(error=f"{fn}: {e}")

    print(f"\n\nDone. {_progress['generated']}/{count} generated, "
          f"{_progress['failures']} failed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic pest videos (no web app needed)"
    )
    parser.add_argument("--count",   type=int,  default=10,
                        help="Number of videos to generate (default: 10)")
    parser.add_argument("--fps",     type=int,  default=10,
                        help="Frames per second (default: 10, length fixed at 24s)")
    parser.add_argument("--workers", type=int,  default=None,
                        help="Parallel workers (default: min(count, cpu_count, 4))")
    args = parser.parse_args()

    run_generation(
        count=args.count,
        fps=args.fps,
        assemble_video=False,
        workers_override=args.workers,
    )


if __name__ == "__main__":
    main()
