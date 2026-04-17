"""YOLOv8s training entry point.

Mirrors the Colab notebook experiment, writing outputs to the cluster model directory.
Labels are loaded on-the-fly from COCO JSON — no .txt label files are written to disk.

Usage:
    python -m training.train_yolo
    python -m training.train_yolo --data_dir /cwork/ad641/pest_detection_dataset

Arguments:
    --data_dir      COCO dataset root with images/ and annotations/ (default: cluster path)
    --project       Root output directory (default: /cwork/ad641/pest_detection_model/results)
    --name          Run subdirectory name (default: yolo8smine)
    --epochs        Number of epochs (default: 50)
    --resume        Resume from last checkpoint if it exists (default: True)
"""

import argparse
import json
import os
from pathlib import Path

_ON_HPC = os.getcwd().startswith("/hpc")
_DEFAULT_DATA_DIR = (
    "/cwork/ad641/pest_detection_dataset" if _ON_HPC else "./pest_detection_dataset"
)
_DEFAULT_PROJECT = (
    "/cwork/ad641/pest_detection_model/results" if _ON_HPC else "./pest_detection_model/results"
)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8s pest detection model")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR)
    parser.add_argument("--project", default=_DEFAULT_PROJECT)
    parser.add_argument("--name", default="yolo8smine")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Write data.yaml if missing (fast — just class names + paths, no label conversion)
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        with open(data_dir / "annotations" / "train.json") as f:
            categories = json.load(f)["categories"]
        from training.dataset import write_yolo_yaml
        write_yolo_yaml(data_dir, categories)

    # Build COCO category id → YOLO 0-based class index
    with open(data_dir / "annotations" / "train.json") as f:
        categories = json.load(f)["categories"]
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    cat_id_to_yolo = {c["id"]: i for i, c in enumerate(sorted_cats)}

    ann_files = {
        "train": str(data_dir / "annotations" / "train.json"),
        "val":   str(data_dir / "annotations" / "val.json"),
    }

    from training.dataset import make_coco_trainer
    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(yaml_path),
        trainer=make_coco_trainer(ann_files, cat_id_to_yolo),
        resume=args.resume,
        epochs=args.epochs,
        imgsz=640,
        batch=-1,
        patience=15,
        save_period=5,
        device="0",
        workers=4,
        project=args.project,
        name=args.name,
        pretrained=True,
        seed=42,
    )


if __name__ == "__main__":
    main()
