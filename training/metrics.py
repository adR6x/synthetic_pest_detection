"""Reusable evaluation helpers for DETR detection workflows."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor

from training.config import DETECTION_RATE_THRESHOLD, FPR_THRESHOLD, IOU_THRESHOLD
from training.data_utils import resolve_split_paths


def _round_or_none(value, digits: int = 4):
    if value is None:
        return None
    return round(float(value), digits)


def load_model_from_path(model_path: str, device: torch.device):
    processor = DetrImageProcessor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor


def _auto_batch_size(device: torch.device, bytes_per_image: int = 400 * 1024 * 1024) -> int:
    """Estimate a safe batch size from free GPU memory. Falls back to 1 on CPU/MPS."""
    if device.type != "cuda":
        return 1
    free, _ = torch.cuda.mem_get_info(device.index if device.index is not None else 0)
    batch = max(1, int(free * 0.8 / bytes_per_image))
    return min(batch, 32)


@torch.no_grad()
def run_detection_on_dataset(model, processor, coco_gt, img_dir, device, threshold=0.3,
                              img_ids=None, batch_size: int | None = None):
    """Run the model on all images in a COCO dataset. Returns a predictions list."""
    img_ids = img_ids if img_ids is not None else coco_gt.getImgIds()
    if batch_size is None:
        batch_size = _auto_batch_size(device)

    predictions = []
    img_info_map = {info["id"]: info for info in coco_gt.loadImgs(img_ids)}

    def _resolve_path(file_name):
        p = os.path.join(img_dir, file_name)
        if not os.path.exists(p):
            p = os.path.join(img_dir, os.path.basename(file_name))
        return p if os.path.exists(p) else None

    for batch_start in tqdm(range(0, len(img_ids), batch_size), desc="Running detection"):
        batch_ids = img_ids[batch_start: batch_start + batch_size]

        images, valid_ids = [], []
        for img_id in batch_ids:
            path = _resolve_path(img_info_map[img_id]["file_name"])
            if path:
                images.append(Image.open(path).convert("RGB"))
                valid_ids.append(img_id)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        target_sizes = torch.tensor([img.size[::-1] for img in images]).to(device)
        batch_results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold,
        )

        for img_id, results in zip(valid_ids, batch_results):
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.cpu().numpy()
                predictions.append({
                    "image_id": img_id,
                    "category_id": label.item(),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": round(score.item(), 6),
                })

    return predictions


def compute_coco_metrics(coco_gt, predictions, img_ids=None):
    if not predictions:
        return {}

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "mAP_0.5_0.95": round(coco_eval.stats[0], 4),
        "mAP_0.5": round(coco_eval.stats[1], 4),
        "mAP_0.75": round(coco_eval.stats[2], 4),
        "AP_small": round(coco_eval.stats[3], 4),
        "AR_max1": round(coco_eval.stats[6], 4),
        "AR_max10": round(coco_eval.stats[7], 4),
        "AR_max100": round(coco_eval.stats[8], 4),
        "AR_small": round(coco_eval.stats[9], 4),
    }


def compute_project_metrics(coco_gt, predictions, iou_threshold=IOU_THRESHOLD, img_ids=None):
    """Compute detection-rate-style metrics using class-aware IoU matching."""
    img_ids = img_ids if img_ids is not None else coco_gt.getImgIds()
    gt_by_img = defaultdict(list)
    pred_by_img = defaultdict(list)

    for img_id in img_ids:
        gt_by_img[img_id] = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
    for pred in predictions:
        pred_by_img[pred["image_id"]].append(pred)

    tp_total = 0
    fn_total = 0
    fp_total = 0
    fp_frames = 0
    tn_frames = 0
    per_class_tp = defaultdict(int)
    per_class_fn = defaultdict(int)
    per_class_fp = defaultdict(int)

    for img_id in img_ids:
        gt_anns = gt_by_img[img_id]
        preds = sorted(pred_by_img[img_id], key=lambda x: -x["score"])
        gt_matched = [False] * len(gt_anns)
        gt_cats = [ann["category_id"] for ann in gt_anns]

        for pred in preds:
            px, py, pw, ph = pred["bbox"]
            pred_cat = pred["category_id"]
            best_iou = 0.0
            best_idx = -1

            for gi, gt_ann in enumerate(gt_anns):
                if gt_matched[gi] or gt_ann["category_id"] != pred_cat:
                    continue

                gx, gy, gw, gh = gt_ann["bbox"]
                x1 = max(px, gx)
                y1 = max(py, gy)
                x2 = min(px + pw, gx + gw)
                y2 = min(py + ph, gy + gh)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                union = pw * ph + gw * gh - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi

            if best_iou >= iou_threshold and best_idx >= 0:
                gt_matched[best_idx] = True
                tp_total += 1
                per_class_tp[pred_cat] += 1
            else:
                fp_total += 1
                per_class_fp[pred_cat] += 1

        for gi, matched in enumerate(gt_matched):
            if not matched:
                fn_total += 1
                per_class_fn[gt_cats[gi]] += 1

        if not gt_anns:
            if preds:
                fp_frames += 1
            else:
                tn_frames += 1

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp_frames / (fp_frames + tn_frames) if (fp_frames + tn_frames) > 0 else 0.0

    cat_names = {c["id"]: c["name"] for c in coco_gt.dataset.get("categories", [])}
    per_class = {}
    all_cat_ids = set(per_class_tp) | set(per_class_fn) | set(per_class_fp)
    for cat_id in all_cat_ids:
        tp = per_class_tp[cat_id]
        fn = per_class_fn[cat_id]
        fp = per_class_fp[cat_id]
        cat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        cat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cat_f1 = (
            2 * cat_precision * cat_recall / (cat_precision + cat_recall)
            if (cat_precision + cat_recall) > 0
            else 0.0
        )
        per_class[cat_names.get(cat_id, str(cat_id))] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(cat_precision, 4),
            "recall": round(cat_recall, 4),
            "f1": round(cat_f1, 4),
        }

    return {
        "detection_rate": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "fp_frames": fp_frames,
        "tn_frames": tn_frames,
        "per_class": per_class,
        "detection_rate_pass": recall >= DETECTION_RATE_THRESHOLD,
        "fpr_pass": fpr < FPR_THRESHOLD,
    }


def _compute_binary_metrics(y_true, y_score, threshold):
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    metrics = {
        "threshold": threshold,
        "num_examples": len(y_true),
        "positive_examples": int(sum(y_true)),
        "negative_examples": int(len(y_true) - sum(y_true)),
        "precision": _round_or_none(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _round_or_none(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _round_or_none(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(set(y_true)) > 1:
        metrics["auroc"] = _round_or_none(roc_auc_score(y_true, y_score))
        metrics["aupr"] = _round_or_none(average_precision_score(y_true, y_score))
    else:
        metrics["auroc"] = None
        metrics["aupr"] = None

    return metrics


def compute_frame_presence_metrics(coco_gt, predictions, threshold, img_ids=None):
    """Compute frame-level presence metrics using max detection score per frame."""
    img_ids = img_ids if img_ids is not None else coco_gt.getImgIds()
    gt_by_img = defaultdict(list)
    pred_by_img = defaultdict(list)

    for img_id in img_ids:
        gt_by_img[img_id] = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
    for pred in predictions:
        pred_by_img[pred["image_id"]].append(pred)

    y_true = []
    y_score = []
    per_class_true = defaultdict(list)
    per_class_score = defaultdict(list)
    cat_names = {c["id"]: c["name"] for c in coco_gt.dataset.get("categories", [])}

    for img_id in img_ids:
        gt_anns = gt_by_img[img_id]
        preds = pred_by_img[img_id]
        y_true.append(1 if gt_anns else 0)
        y_score.append(max((pred["score"] for pred in preds), default=0.0))

        gt_classes = {ann["category_id"] for ann in gt_anns}
        for cat_id, cat_name in cat_names.items():
            per_class_true[cat_name].append(1 if cat_id in gt_classes else 0)
            per_class_score[cat_name].append(
                max(
                    (pred["score"] for pred in preds if pred["category_id"] == cat_id),
                    default=0.0,
                )
            )

    per_class = {
        name: _compute_binary_metrics(per_class_true[name], per_class_score[name], threshold)
        for name in sorted(per_class_true)
    }

    return {
        "any_pest": _compute_binary_metrics(y_true, y_score, threshold),
        "per_class": per_class,
    }


def evaluate_model_on_split(
    model,
    processor,
    data_root: str | Path,
    split: str,
    device: torch.device,
    threshold: float = 0.5,
    postprocess_threshold: float | None = None,
    sample_n: int | None = None,
):
    import random as _random
    resolved = resolve_split_paths(data_root, split)
    annotation_path = resolved["annotation_path"]
    image_dir = resolved["image_dir"]

    coco_gt = COCO(str(annotation_path))
    if postprocess_threshold is None:
        postprocess_threshold = max(0.05, threshold * 0.5)

    all_img_ids = coco_gt.getImgIds()
    if sample_n is not None and sample_n < len(all_img_ids):
        img_ids = _random.sample(all_img_ids, sample_n)
    else:
        img_ids = all_img_ids

    predictions = run_detection_on_dataset(
        model,
        processor,
        coco_gt,
        str(image_dir),
        device,
        threshold=postprocess_threshold,
        img_ids=img_ids,
    )
    thresholded_predictions = [p for p in predictions if p["score"] >= threshold]

    return {
        "split": split,
        "resolved_split": resolved["resolved_image_split"],
        "annotation_file": resolved["resolved_annotation_file"],
        "image_dir": str(image_dir),
        "annotation_path": str(annotation_path),
        "num_images": len(img_ids),
        "num_annotations": len(coco_gt.getAnnIds(imgIds=img_ids)),
        "sampled": sample_n is not None,
        "raw_prediction_count": len(predictions),
        "thresholded_prediction_count": len(thresholded_predictions),
        "score_threshold": threshold,
        "postprocess_threshold": postprocess_threshold,
        "coco_metrics": compute_coco_metrics(coco_gt, predictions, img_ids=img_ids),
        "project_metrics": compute_project_metrics(coco_gt, thresholded_predictions, img_ids=img_ids),
        "frame_presence_metrics": compute_frame_presence_metrics(coco_gt, predictions, threshold, img_ids=img_ids),
    }
