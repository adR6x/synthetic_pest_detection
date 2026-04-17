"""Evaluate a fine-tuned detection model (DETR or YOLOv8) on a dataset split.

Results are saved to:
  <model_repo_dir>/test_evaluation/<run_name>/

For DETR: results.json + YOLO-style plots (confusion matrix, PR/P/R/F1 curves, results.csv)
For YOLO: native ultralytics output (same plots produced automatically)

Usage examples:
  # DETR
  python -m training.evaluate --model_path /cwork/.../runs/run_a54317ae/best

  # YOLO
  python -m training.evaluate --model_path /cwork/.../results/yolo8smine/weights/best.pt

  # Explicit split / threshold
  python -m training.evaluate --model_path ... --split test --threshold 0.5
"""

import csv
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO

from training.config import IOU_THRESHOLD, get_device
from training.data_utils import collect_dataset_metadata, resolve_split_paths
from training.metrics import (
    compute_coco_metrics,
    compute_frame_presence_metrics,
    compute_project_metrics,
    load_model_from_path,
    run_detection_on_dataset,
)
from training.reporting import (
    build_environment_metadata,
    save_json,
    try_git_commit,
    utc_now_iso,
)

_ON_HPC = os.getcwd().startswith("/hpc")
_DEFAULT_MODEL_REPO = (
    "/cwork/ad641/pest_detection_model" if _ON_HPC else "./pest_detection_model"
)
_DEFAULT_DATA_DIR = (
    "/cwork/ad641/pest_detection_dataset" if _ON_HPC else "./pest_detection_dataset"
)


# ---------------------------------------------------------------------------
# Plot helpers (DETR → ultralytics format conversion + rendering)
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _iou_xyxy(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0.0


def _generate_detr_plots(coco_gt, predictions, threshold, save_dir,
                          iou_threshold=IOU_THRESHOLD, img_ids=None):
    """Generate YOLO-style plots for DETR using ultralytics rendering utilities."""
    from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class

    save_dir = Path(save_dir)
    cats = sorted(coco_gt.dataset["categories"], key=lambda c: c["id"])
    id2idx = {c["id"]: i for i, c in enumerate(cats)}
    nc = len(cats)
    names = {i: c["name"] for i, c in enumerate(cats)}
    img_ids = img_ids or coco_gt.getImgIds()

    gt_by_img = {img_id: coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
                 for img_id in img_ids}

    # --- Confusion matrix ---
    print("  Building confusion matrix ...")
    names_list = list(names.values())
    for _kwargs in [
        {"nc": nc, "conf": threshold, "iou_thres": iou_threshold},
        {"nc": nc, "conf": threshold, "iou_threshold": iou_threshold},
        {"nc": nc, "conf": threshold},
        {"names": names_list, "conf": threshold, "iou_thres": iou_threshold},
        {"names": names_list, "conf": threshold},
        {"names": names_list},
    ]:
        try:
            cm = ConfusionMatrix(**_kwargs)
            break
        except TypeError:
            continue
    else:
        cm = ConfusionMatrix(names_list)

    pred_by_img = defaultdict(list)
    for p in predictions:
        if p["score"] >= threshold:
            pred_by_img[p["image_id"]].append(p)

    for img_id in img_ids:
        gt_anns = gt_by_img[img_id]
        gt_boxes = (np.array([_xywh_to_xyxy(a["bbox"]) for a in gt_anns], dtype=np.float32)
                    if gt_anns else np.zeros((0, 4), dtype=np.float32))
        gt_cls   = (np.array([id2idx.get(a["category_id"], 0) for a in gt_anns], dtype=np.int32)
                    if gt_anns else np.zeros(0, dtype=np.int32))
        preds = pred_by_img[img_id]
        det = (np.array([[*_xywh_to_xyxy(p["bbox"]), p["score"],
                          id2idx.get(p["category_id"], 0)] for p in preds], dtype=np.float32)
               if preds else np.zeros((0, 6), dtype=np.float32))
        try:
            # older ultralytics: process_batch(det_array, gt_boxes, gt_cls)
            cm.process_batch(det, gt_boxes, gt_cls)
        except (IndexError, TypeError):
            # newer ultralytics: process_batch({"cls": ..., "bboxes": ..., "conf": ...}, gt_boxes, gt_cls)
            det_dict = {
                "cls":   torch.tensor(det[:, 5]) if len(det) else torch.zeros(0),
                "bboxes": torch.tensor(det[:, :4]) if len(det) else torch.zeros((0, 4)),
                "conf":  torch.tensor(det[:, 4]) if len(det) else torch.zeros(0),
            }
            cm.process_batch(det_dict, torch.tensor(gt_boxes), torch.tensor(gt_cls))

    for normalize in (False, True):
        try:
            cm.plot(normalize=normalize, save_dir=str(save_dir), names=list(names.values()))
        except TypeError:
            try:
                cm.plot(normalize=normalize, save_dir=str(save_dir))
            except Exception:
                pass

    # --- PR / P / R / F1 curves ---
    print("  Computing PR curve data ...")
    gt_boxes_by_img_cls: dict = defaultdict(lambda: defaultdict(list))
    match_used:          dict = defaultdict(lambda: defaultdict(list))
    target_cls_list = []

    for img_id in img_ids:
        for ann in gt_by_img[img_id]:
            cls_idx = id2idx.get(ann["category_id"], -1)
            if cls_idx >= 0:
                gt_boxes_by_img_cls[img_id][cls_idx].append(_xywh_to_xyxy(ann["bbox"]))
                match_used[img_id][cls_idx].append(False)
                target_cls_list.append(cls_idx)

    tp_list, conf_list, pred_cls_list = [], [], []
    for pred in sorted(predictions, key=lambda x: -x["score"]):
        cls_idx = id2idx.get(pred["category_id"], -1)
        if cls_idx < 0:
            continue
        img_id   = pred["image_id"]
        pred_box = _xywh_to_xyxy(pred["bbox"])
        gt_list  = gt_boxes_by_img_cls[img_id][cls_idx]
        flags    = match_used[img_id][cls_idx]

        best_iou, best_gi = 0.0, -1
        for gi, gt_box in enumerate(gt_list):
            if not flags[gi]:
                iou = _iou_xyxy(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi

        is_tp = best_iou >= iou_threshold and best_gi >= 0
        if is_tp:
            flags[best_gi] = True
        tp_list.append(is_tp)
        conf_list.append(pred["score"])
        pred_cls_list.append(cls_idx)

    if tp_list:
        ap_per_class(
            tp=np.array(tp_list, dtype=bool),
            conf=np.array(conf_list, dtype=np.float32),
            pred_cls=np.array(pred_cls_list, dtype=np.int32),
            target_cls=np.array(target_cls_list, dtype=np.int32),
            plot=True,
            save_dir=save_dir,
            names=names,
        )

    print(f"  Plots saved to {save_dir}")


def _save_results_csv(split_report, save_path):
    pm     = split_report.get("project_metrics", {})
    coco   = split_report.get("coco_metrics", {})
    fp_any = split_report.get("frame_presence_metrics", {}).get("any_pest", {})
    row = {
        "split":     split_report.get("split", ""),
        "precision": pm.get("precision", ""),
        "recall":    pm.get("recall", ""),
        "f1":        pm.get("f1", ""),
        "mAP50":     coco.get("mAP_0.5", ""),
        "mAP50-95":  coco.get("mAP_0.5_0.95", ""),
        "auroc":     fp_any.get("auroc", ""),
        "aupr":      fp_any.get("aupr", ""),
        "tp":        pm.get("tp", ""),
        "fp":        pm.get("fp", ""),
        "fn":        pm.get("fn", ""),
    }
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Model type detection / run name
# ---------------------------------------------------------------------------

def _is_yolo(model_path: str) -> bool:
    return str(model_path).endswith(".pt")


def _run_name_from_path(model_path: Path, model_type: str) -> str:
    if model_type == "yolo":
        return model_path.parent.parent.name   # .../results/yolo8smine/weights/best.pt
    return model_path.parent.name              # .../runs/run_a54317ae/best


# ---------------------------------------------------------------------------
# YOLO evaluation
# ---------------------------------------------------------------------------

def evaluate_yolo(model_path: str, data_dir: str, split: str, output_dir: Path, threshold: float):
    from ultralytics import YOLO
    from training.dataset import make_coco_validator, write_yolo_yaml

    data_dir = Path(data_dir)
    yaml_path = data_dir / "data.yaml"

    if not yaml_path.exists():
        with open(data_dir / "annotations" / "train.json") as f:
            categories = json.load(f)["categories"]
        write_yolo_yaml(data_dir, categories)

    with open(data_dir / "annotations" / "train.json") as f:
        categories = json.load(f)["categories"]
    cat_id_to_yolo = {c["id"]: i for i, c in enumerate(sorted(categories, key=lambda c: c["id"]))}

    ann_files = {
        "train": str(data_dir / "annotations" / "train.json"),
        "val":   str(data_dir / "annotations" / "val.json"),
        "test":  str(data_dir / "annotations" / "test.json"),
    }

    ValidatorClass = make_coco_validator(ann_files, cat_id_to_yolo)

    class _CocoYOLO(YOLO):
        @property
        def task_map(self):
            tm = super().task_map
            return {**tm, "detect": {**tm.get("detect", {}), "validator": ValidatorClass}}

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _CocoYOLO(model_path).val(
            data=str(yaml_path),
            split=split,
            project=str(output_dir.parent),
            name=output_dir.name,
            conf=threshold,
            plots=True,
            exist_ok=True,
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# DETR evaluation
# ---------------------------------------------------------------------------

def evaluate_detr(model_path: str, data_dir: str, split: str, output_dir: Path, threshold: float, device, n_images: int = None):
    resolved = resolve_split_paths(data_dir, split)
    coco_gt = COCO(str(resolved["annotation_path"]))
    model, processor = load_model_from_path(model_path, device)

    postprocess_threshold = max(0.05, threshold * 0.5)
    img_ids = coco_gt.getImgIds()
    if n_images is not None and n_images < len(img_ids):
        import random
        img_ids = random.sample(img_ids, n_images)
        print(f"  Sampling {n_images} images from {split} split ...")
    print(f"  Running inference on {len(img_ids)} images ...")
    predictions = run_detection_on_dataset(
        model, processor, coco_gt, str(resolved["image_dir"]),
        device, threshold=postprocess_threshold, img_ids=img_ids,
    )
    thresholded = [p for p in predictions if p["score"] >= threshold]

    split_report = {
        "split": split,
        "resolved_split": resolved["resolved_image_split"],
        "annotation_file": resolved["resolved_annotation_file"],
        "image_dir": str(resolved["image_dir"]),
        "annotation_path": str(resolved["annotation_path"]),
        "num_images": len(img_ids),
        "num_annotations": len(coco_gt.getAnnIds(imgIds=img_ids)),
        "raw_prediction_count": len(predictions),
        "thresholded_prediction_count": len(thresholded),
        "score_threshold": threshold,
        "postprocess_threshold": postprocess_threshold,
        "coco_metrics": compute_coco_metrics(coco_gt, predictions, img_ids=img_ids),
        "project_metrics": compute_project_metrics(coco_gt, thresholded, img_ids=img_ids),
        "frame_presence_metrics": compute_frame_presence_metrics(coco_gt, predictions, threshold, img_ids=img_ids),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Generating plots ...")
    _generate_detr_plots(coco_gt, predictions, threshold, output_dir,
                         iou_threshold=IOU_THRESHOLD, img_ids=img_ids)
    _save_results_csv(split_report, output_dir / "results.csv")

    report = {
        "schema_version": 1,
        "status": "success",
        "workflow": "evaluate",
        "created_at_utc": utc_now_iso(),
        "model_path": str(model_path),
        "split": split,
        "dataset": collect_dataset_metadata(data_dir),
        "environment": build_environment_metadata(device, num_workers=0),
        "code": {"git_commit": try_git_commit()},
        "evaluation": split_report,
    }
    output_json = output_dir / "results.json"
    save_json(output_json, report)

    del model, predictions, thresholded
    gc.collect()
    torch.cuda.empty_cache()

    return report, output_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DETR or YOLOv8 pest detection model")
    parser.add_argument("--model_path", required=True,
                        help="HF weights dir (DETR) or .pt file (YOLO)")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "validation", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model_repo_dir", default=_DEFAULT_MODEL_REPO)
    parser.add_argument("--model_type", choices=["detr", "yolo"], default=None,
                        help="Auto-detected from path if omitted (.pt → yolo, dir → detr)")
    parser.add_argument("--n_images", type=int, default=None,
                        help="Evaluate on a random sample of N images (default: all)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_type = args.model_type or ("yolo" if _is_yolo(args.model_path) else "detr")
    run_name   = _run_name_from_path(model_path, model_type)
    output_dir = Path(args.model_repo_dir) / "test_evaluation" / run_name

    print(f"Model type : {model_type}")
    print(f"Model path : {model_path}")
    print(f"Split      : {args.split}")
    print(f"Output dir : {output_dir}")

    if model_type == "yolo":
        evaluate_yolo(str(model_path), args.data_dir, args.split, output_dir, args.threshold)
        print(f"\nResults saved to: {output_dir}")
    else:
        device = get_device()
        print(f"Device     : {device}\n")
        report, output_json = evaluate_detr(
            str(model_path), args.data_dir, args.split, output_dir, args.threshold, device,
            n_images=args.n_images,
        )

        ev = report["evaluation"]
        print("\n" + "=" * 60)
        print("  COCO Standard Metrics")
        print("=" * 60)
        for key, value in ev["coco_metrics"].items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print(f"  Project Metrics (threshold={args.threshold:.2f})")
        print("=" * 60)
        pm = ev["project_metrics"]
        print(f"  Detection Rate : {pm['detection_rate']:.1%}")
        print(f"  False Pos Rate : {pm['false_positive_rate']:.1%}")
        print(f"  Precision      : {pm['precision']:.1%}")
        print(f"  F1             : {pm['f1']:.1%}")

        print("\n" + "=" * 60)
        print("  Frame Presence Metrics")
        print("=" * 60)
        ap = ev["frame_presence_metrics"]["any_pest"]
        print(f"  AUROC     : {ap['auroc']}")
        print(f"  AUPR      : {ap['aupr']}")
        print(f"  Precision : {ap['precision']}")
        print(f"  Recall    : {ap['recall']}")
        print(f"  F1        : {ap['f1']}")

        print(f"\nReport saved to: {output_json}")


if __name__ == "__main__":
    main()
