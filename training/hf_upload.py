"""Incrementally upload new generator outputs to a Hugging Face dataset repo.

The real generator writes:
    outputs/train/frames/{job_id}/frame_NNNN.png
    outputs/train/labels/{job_id}/annotations.json
    outputs/test/frames/{job_id}/frame_NNNN.png
    outputs/test/labels/{job_id}/annotations.json
    outputs/generated_state.json

The HF dataset format (preserved exactly):
    images/train/{job_id}/frame_NNNN.png
    images/val/{job_id}/frame_NNNN.png
    images/test/{job_id}/frame_NNNN.png
    annotations/train.json     (COCO, file_name="{job_id}/frame_NNNN.png")
    annotations/val.json
    annotations/test.json
    generated_state.json

Only jobs not already present in the HF repo are uploaded.  The three split
annotation JSONs are downloaded from HF, updated in-place, and re-uploaded
together with the new frame images and the merged generated_state.json.

Usage:
    # upload with your repo (log in first: huggingface-cli login)
    python -m training.hf_upload --repo_id adR6x/pest_detection_dataset

    # carve 10% of new train jobs into the val split
    python -m training.hf_upload --repo_id adR6x/pest_detection_dataset --val_frac 0.1

    # preview what would be uploaded without touching HF
    python -m training.hf_upload --repo_id adR6x/pest_detection_dataset --dry_run
"""

import argparse
import json
import os
import re
import shutil
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_NAME_RE = re.compile(r"^frame_(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)

# The generator only produces train and test output splits.
# val is optionally carved from train at upload time.
_GENERATOR_SPLITS = ("train", "test")

# Fallback COCO categories matching the generator (compositing.py)
_DEFAULT_CATEGORIES = [
    {"id": 1, "name": "mouse",      "supercategory": "pest"},
    {"id": 2, "name": "rat",        "supercategory": "pest"},
    {"id": 3, "name": "cockroach",  "supercategory": "pest"},
]


# ---------------------------------------------------------------------------
# Local output scanning
# ---------------------------------------------------------------------------

def _collect_local_jobs(output_dir):
    """Scan outputs/ and return {generator_split: {job_id: job_info}}.

    job_info keys:
        frames_dir  – absolute path to the job's frames directory
        ann_path    – absolute path to the job's annotations.json
        frames      – sorted list of frame filenames present on disk
    """
    result = {}
    for split in _GENERATOR_SPLITS:
        frames_root = os.path.join(output_dir, split, "frames")
        labels_root = os.path.join(output_dir, split, "labels")
        if not os.path.isdir(frames_root):
            continue
        jobs = {}
        for job_id in sorted(os.listdir(frames_root)):
            frames_dir = os.path.join(frames_root, job_id)
            ann_path   = os.path.join(labels_root, job_id, "annotations.json")
            if not os.path.isdir(frames_dir) or not os.path.exists(ann_path):
                continue
            frames = sorted(f for f in os.listdir(frames_dir) if FRAME_NAME_RE.match(f))
            if frames:
                jobs[job_id] = {
                    "frames_dir": frames_dir,
                    "ann_path":   ann_path,
                    "frames":     frames,
                }
        if jobs:
            result[split] = jobs
    return result


# ---------------------------------------------------------------------------
# HF metadata helpers
# ---------------------------------------------------------------------------

def _load_hf_json(api, repo_id, repo_filename):
    """Download a JSON file from the HF dataset repo.  Returns None on failure."""
    try:
        local = api.hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_filename,
        )
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _empty_coco():
    return {"images": [], "annotations": [], "categories": list(_DEFAULT_CATEGORIES)}


def _extract_job_ids(coco):
    """Return set of job_ids referenced by file_name entries ("{job_id}/frame_N.png")."""
    job_ids = set()
    for img in coco.get("images", []):
        fname = img.get("file_name", "")
        parts = fname.split("/")
        if len(parts) >= 2:
            job_ids.add(parts[0])
    return job_ids


def _max_ids(coco):
    """Return (max_image_id, max_ann_id) in a COCO dict (0 if empty)."""
    max_img = max((img["id"] for img in coco.get("images", [])), default=0)
    max_ann = max((a["id"]  for a  in coco.get("annotations", [])), default=0)
    return max_img, max_ann


# ---------------------------------------------------------------------------
# COCO entry building
# ---------------------------------------------------------------------------

def _build_coco_entries(job_id, job_info, start_img_id, start_ann_id):
    """Build new COCO image + annotation entries for a single job.

    HF file_name convention:  "{job_id}/frame_NNNN.png"
    The per-job annotations.json uses bare "frame_NNNN.png" names; this
    function maps those local IDs to new globally-unique HF IDs.

    Returns:
        new_images, new_anns, categories, next_img_id, next_ann_id
    """
    with open(job_info["ann_path"], "r", encoding="utf-8") as f:
        per_job = json.load(f)

    categories = per_job.get("categories") or list(_DEFAULT_CATEGORIES)

    # Build lookup: local bare filename → local image id
    local_fname_to_id = {
        img["file_name"]: img["id"]
        for img in per_job.get("images", [])
    }
    # Build lookup: local image id → list of annotations
    anns_by_local_id = {}
    for ann in per_job.get("annotations", []):
        anns_by_local_id.setdefault(ann["image_id"], []).append(ann)

    new_images = []
    new_anns   = []
    img_id     = start_img_id
    ann_id     = start_ann_id

    for fname in job_info["frames"]:
        hf_fname     = f"{job_id}/{fname}"
        local_img_id = local_fname_to_id.get(fname)

        new_images.append({
            "id":        img_id,
            "file_name": hf_fname,
            "width":     640,
            "height":    480,
        })

        for ann in anns_by_local_id.get(local_img_id, []):
            new_anns.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": ann["category_id"],
                "bbox":        ann["bbox"],
                "area":        ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd":     ann.get("iscrowd", 0),
            })
            ann_id += 1

        img_id += 1

    return new_images, new_anns, categories, img_id, ann_id


# ---------------------------------------------------------------------------
# Main upload logic
# ---------------------------------------------------------------------------

def upload_incremental(
    output_dir,
    repo_id,
    val_frac=0.0,
    private=False,
    commit_message="Add new generated pest detection data",
    dry_run=False,
):
    """Upload only new jobs to the HF dataset, preserving its existing format.

    Args:
        output_dir:     Local outputs/ directory produced by the real generator.
        repo_id:        HF repo id, e.g. "adR6x/pest_detection_dataset".
        val_frac:       Fraction of new train jobs to assign to the val split
                        (0 = all new train jobs go to train).
        private:        Create a private repo if it does not yet exist.
        commit_message: Git commit message for the HF push.
        dry_run:        Print plan without uploading anything.
    """
    import random
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"Repository: https://huggingface.co/datasets/{repo_id}")

    # --- 1. Scan local generator outputs ---
    local_jobs = _collect_local_jobs(output_dir)
    if not local_jobs:
        print("No generator outputs found in", output_dir)
        print("Run the real generator first, then re-run this script.")
        return

    local_total = sum(len(j) for j in local_jobs.values())
    print(f"Found {local_total} local job(s) across splits: "
          + ", ".join(f"{s}={len(j)}" for s, j in local_jobs.items()))

    # --- 2. Download existing HF COCO annotations to find already-uploaded jobs ---
    print("Fetching existing HF annotation metadata...")
    hf_coco = {}
    for split in ("train", "val", "test"):
        data = _load_hf_json(api, repo_id, f"annotations/{split}.json")
        hf_coco[split] = data if data is not None else _empty_coco()

    hf_generated_state = _load_hf_json(api, repo_id, "generated_state.json")
    if hf_generated_state is None:
        hf_generated_state = {"generated_videos": []}
    elif isinstance(hf_generated_state, list):
        hf_generated_state = {"generated_videos": hf_generated_state}

    # All job_ids already in HF (across all three splits)
    hf_existing_job_ids = set()
    for coco in hf_coco.values():
        hf_existing_job_ids.update(_extract_job_ids(coco))
    print(f"HF dataset already contains {len(hf_existing_job_ids)} job(s).")

    # --- 3. Determine new jobs per HF split ---
    new_jobs_by_hf_split = {"train": {}, "val": {}, "test": {}}

    for gen_split, jobs in local_jobs.items():
        new_jobs = {jid: info for jid, info in jobs.items() if jid not in hf_existing_job_ids}
        if not new_jobs:
            continue

        if gen_split == "train" and val_frac > 0.0:
            # Deterministically carve a val fraction from new train jobs
            sorted_ids = sorted(new_jobs)
            rng = random.Random(42)
            rng.shuffle(sorted_ids)
            n_val = max(1, int(len(sorted_ids) * val_frac))
            for jid in sorted_ids[:n_val]:
                new_jobs_by_hf_split["val"][jid] = new_jobs[jid]
            for jid in sorted_ids[n_val:]:
                new_jobs_by_hf_split["train"][jid] = new_jobs[jid]
        elif gen_split == "test":
            new_jobs_by_hf_split["test"].update(new_jobs)
        else:
            # gen_split == "train" with no val carving
            new_jobs_by_hf_split["train"].update(new_jobs)

    total_new = sum(len(j) for j in new_jobs_by_hf_split.values())
    if total_new == 0:
        print("All local jobs are already in the HF dataset. Nothing to upload.")
        return

    print(
        f"New jobs to upload: "
        + ", ".join(
            f"{s}={len(j)}"
            for s, j in new_jobs_by_hf_split.items()
            if j
        )
    )

    if dry_run:
        for hf_split, jobs in new_jobs_by_hf_split.items():
            if jobs:
                print(f"  [{hf_split}] " + ", ".join(sorted(jobs)))
        print("[dry-run] No files uploaded.")
        return

    # --- 4. Build updated COCOs + stage files for upload ---
    with tempfile.TemporaryDirectory(prefix="hf_upload_") as staging_root:
        hf_staging = os.path.join(staging_root, "hf_root")
        os.makedirs(hf_staging, exist_ok=True)

        for hf_split, new_jobs in new_jobs_by_hf_split.items():
            if not new_jobs:
                continue

            existing_coco = hf_coco[hf_split]
            # Ensure categories are populated
            if not existing_coco.get("categories"):
                existing_coco["categories"] = list(_DEFAULT_CATEGORIES)

            img_id, ann_id = _max_ids(existing_coco)
            img_id += 1
            ann_id += 1

            all_new_images = []
            all_new_anns   = []
            categories     = existing_coco["categories"]

            for job_id, job_info in sorted(new_jobs.items()):
                imgs, anns, job_cats, img_id, ann_id = _build_coco_entries(
                    job_id, job_info, img_id, ann_id
                )
                all_new_images.extend(imgs)
                all_new_anns.extend(anns)
                if not categories and job_cats:
                    categories = job_cats

                # Stage each frame image under images/{hf_split}/{job_id}/
                for fname in job_info["frames"]:
                    src = os.path.join(job_info["frames_dir"], fname)
                    dst = os.path.join(
                        hf_staging, "images", hf_split, job_id, fname
                    )
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)

            # Write merged annotation JSON
            merged_coco = {
                "images":      existing_coco.get("images", []) + all_new_images,
                "annotations": existing_coco.get("annotations", []) + all_new_anns,
                "categories":  categories,
            }
            ann_out = os.path.join(hf_staging, "annotations", f"{hf_split}.json")
            os.makedirs(os.path.dirname(ann_out), exist_ok=True)
            with open(ann_out, "w", encoding="utf-8") as f:
                json.dump(merged_coco, f, indent=2)

            print(
                f"  {hf_split}: +{len(all_new_images)} frames, "
                f"+{len(all_new_anns)} annotations"
            )

        # --- 5. Merge + stage generated_state.json ---
        local_state_path = os.path.join(output_dir, "generated_state.json")
        if os.path.exists(local_state_path):
            with open(local_state_path, "r", encoding="utf-8") as f:
                local_state = json.load(f)
            if isinstance(local_state, list):
                local_state = {"generated_videos": local_state}

            hf_state_ids = {
                row.get("job_id") or row.get("video_id")
                for row in hf_generated_state.get("generated_videos", [])
            }
            new_rows = [
                row for row in local_state.get("generated_videos", [])
                if (row.get("job_id") or row.get("video_id")) not in hf_state_ids
            ]
            merged_state = {
                "generated_videos": (
                    hf_generated_state.get("generated_videos", []) + new_rows
                ),
            }
            if "updated_at" in local_state:
                merged_state["updated_at"] = local_state["updated_at"]

            state_out = os.path.join(hf_staging, "generated_state.json")
            with open(state_out, "w", encoding="utf-8") as f:
                json.dump(merged_state, f, indent=2)
            print(f"  generated_state.json: +{len(new_rows)} new entries")

        # --- 6. Upload staging directory to HF ---
        print(f"\nUploading to {repo_id} ...")
        api.upload_folder(
            folder_path=hf_staging,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

    print(f"\nDone. View dataset at: https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Incrementally upload new generator outputs to a HuggingFace dataset"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_ROOT, "outputs"),
        help="Project outputs/ directory (default: outputs/)",
    )
    parser.add_argument(
        "--repo_id",
        default="adR6x/pest_detection_dataset",
        help="HuggingFace dataset repo  (default: adR6x/pest_detection_dataset)",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.0,
        help=(
            "Fraction of new train jobs to assign to the val split "
            "(default: 0, all new train jobs go to train)"
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository if it does not exist",
    )
    parser.add_argument(
        "--commit_message",
        default="Add new generated pest detection data",
        help="Commit message for the HF push",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be uploaded without touching HF",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        raise SystemExit(
            f"outputs directory not found: {args.output_dir}\n"
            "Run the real generator first."
        )

    upload_incremental(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        val_frac=args.val_frac,
        private=args.private,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
