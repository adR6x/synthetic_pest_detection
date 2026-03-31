"""Convert generator output into a DETR-compatible COCO dataset.

The generator writes:
    outputs/frames/{job_id}/frame_XXXX.png   (one PNG per frame)
    outputs/labels/{job_id}/annotations.json  (one COCO file per job)

DETR training expects:
    data_dir/images/train/*.png
    data_dir/images/val/*.png
    data_dir/images/test/*.png
    data_dir/annotations/train.json
    data_dir/annotations/val.json
    data_dir/annotations/test.json

Split strategy: split by JOB (video), not by frame.
  Each video is 300 nearly-identical frames of the same background.
  Splitting at the frame level would leak the background into val/test,
  making metrics look good while the model learns nothing generalisable.
  Splitting by job means val and test contain entirely unseen kitchens.

Usage:
    python -m training.prepare_dataset
    python -m training.prepare_dataset --frames_root outputs/frames \\
                                       --labels_root outputs/labels \\
                                       --output_dir  outputs/dataset \\
                                       --val_frac 0.1 --test_frac 0.1 \\
                                       --every_n 10
"""

import argparse
import json
import os
import random
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_jobs(frames_root, labels_root, skip_empty=False, every_n=1):
    """Return list of jobs, each job being a list of (frame_path, anns).

    Splitting is done at the job level to prevent data leakage between
    frames from the same video appearing in both train and val/test.
    """
    jobs = []
    categories = None

    for job_id in sorted(os.listdir(frames_root)):
        job_frames = os.path.join(frames_root, job_id)
        ann_path   = os.path.join(labels_root, job_id, "annotations.json")

        if not os.path.isdir(job_frames) or not os.path.exists(ann_path):
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        if categories is None:
            categories = coco.get("categories", [])

        fname_to_id   = {img["file_name"]: img["id"] for img in coco.get("images", [])}
        anns_by_image = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        job_frames_list = []
        frame_fnames = sorted(
            f for f in os.listdir(job_frames)
            if f.endswith(".png") and f.startswith("frame_")
        )

        for i, fname in enumerate(frame_fnames):
            if i % every_n != 0:          # subsample frames to reduce redundancy
                continue
            image_id = fname_to_id.get(fname)
            if image_id is None:
                continue
            anns = anns_by_image.get(image_id, [])
            if skip_empty and not anns:
                continue
            job_frames_list.append((os.path.join(job_frames, fname), anns))

        if job_frames_list:
            jobs.append((job_id, job_frames_list))

    return jobs, categories or []


def write_split(job_frames_flat, split_dir, ann_path, categories,
                global_image_id_start, global_ann_id_start):
    """Copy frames and write a COCO JSON for one split."""
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ann_path), exist_ok=True)

    coco_images = []
    coco_annotations = []
    img_id = global_image_id_start
    ann_id = global_ann_id_start

    for frame_path, anns in job_frames_flat:
        fname    = os.path.basename(frame_path)
        dst_name = f"{img_id:06d}_{fname}"
        shutil.copy2(frame_path, os.path.join(split_dir, dst_name))

        coco_images.append({
            "id": img_id, "file_name": dst_name, "width": 640, "height": 480,
        })
        for ann in anns:
            coco_annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
            })
            ann_id += 1
        img_id += 1

    with open(ann_path, "w") as f:
        json.dump({"images": coco_images, "annotations": coco_annotations,
                   "categories": categories}, f, indent=2)

    print(f"  {os.path.basename(split_dir):5s}: "
          f"{len(coco_images):4d} frames, {len(coco_annotations):5d} annotations")
    return img_id, ann_id


def main():
    parser = argparse.ArgumentParser(description="Prepare DETR dataset from generator output")
    parser.add_argument("--frames_root", default=os.path.join(PROJECT_ROOT, "outputs", "frames"))
    parser.add_argument("--labels_root", default=os.path.join(PROJECT_ROOT, "outputs", "labels"))
    parser.add_argument("--output_dir",  default=os.path.join(PROJECT_ROOT, "outputs", "dataset"))
    parser.add_argument("--val_frac",    type=float, default=0.1)
    parser.add_argument("--test_frac",   type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--skip_empty",  action="store_true",
                        help="Exclude frames with no pest annotations")
    parser.add_argument("--every_n",     type=int, default=10,
                        help="Use every Nth frame per video (default 10). "
                             "Consecutive frames are nearly identical so subsampling "
                             "reduces redundancy without losing coverage.")
    args = parser.parse_args()

    print(f"Scanning {args.frames_root} (every_n={args.every_n}) ...")
    jobs, categories = collect_jobs(
        args.frames_root, args.labels_root,
        skip_empty=args.skip_empty, every_n=args.every_n,
    )

    if not jobs:
        print("No jobs found. Generate some videos via the web app first.")
        return

    # Shuffle at the JOB level — keeps all frames from the same video together
    rng = random.Random(args.seed)
    rng.shuffle(jobs)

    n_jobs  = len(jobs)
    n_test  = max(1, int(n_jobs * args.test_frac))
    n_val   = max(1, int(n_jobs * args.val_frac))
    n_train = n_jobs - n_val - n_test

    if n_train < 1:
        print(f"Not enough jobs ({n_jobs}) to split. Generate more videos.")
        return

    job_splits = {
        "train": jobs[:n_train],
        "val":   jobs[n_train:n_train + n_val],
        "test":  jobs[n_train + n_val:],
    }

    total_frames = sum(len(f) for _, f in jobs)
    print(f"\n{n_jobs} jobs ({total_frames} frames after subsampling)")
    print(f"Split: train={n_train} jobs, val={n_val} jobs, test={n_test} jobs")
    print(f"Writing to: {args.output_dir}\n")

    img_id = 1
    ann_id = 1
    for split, split_jobs in job_splits.items():
        flat_frames = [frame for _, job_frames in split_jobs for frame in job_frames]
        img_id, ann_id = write_split(
            flat_frames,
            split_dir=os.path.join(args.output_dir, "images", split),
            ann_path=os.path.join(args.output_dir, "annotations", f"{split}.json"),
            categories=categories,
            global_image_id_start=img_id,
            global_ann_id_start=ann_id,
        )

    print(f"\nDone. Dataset at: {args.output_dir}")
    print("\nTo train DETR:")
    print(f"  python -m training.train --data_dir {args.output_dir} --freeze_backbone")


if __name__ == "__main__":
    main()
