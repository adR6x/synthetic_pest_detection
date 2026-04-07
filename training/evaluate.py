"""Evaluate a fine-tuned DETR model against ground-truth COCO annotations."""

import argparse
import os
from pathlib import Path

import torch

from training.config import get_device
from training.data_utils import collect_dataset_metadata
from training.metrics import evaluate_model_on_split, load_model_from_path
from training.reporting import (
    build_environment_metadata,
    make_run_artifacts,
    save_json,
    try_git_commit,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DETR pest detection")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "validation", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--evaluation_dir", default=None)
    _default_model_repo = (
        "/cwork/ad641/pest_detection_model" if os.getcwd().startswith("/hpc")
        else "./pest_detection_model"
    )
    parser.add_argument("--model_repo_dir", default=_default_model_repo)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    if args.evaluation_dir is None:
        args.evaluation_dir = str(Path(args.model_repo_dir) / "evaluations")

    artifacts = make_run_artifacts(
        experiment_name=f"evaluate_{Path(args.model_path).name}_{args.split}",
        evaluation_dir=args.evaluation_dir,
    )

    model, processor = load_model_from_path(args.model_path, device)
    split_report = evaluate_model_on_split(
        model,
        processor,
        args.data_dir,
        args.split,
        device,
        threshold=args.threshold,
    )

    print("\n" + "=" * 60)
    print("  COCO Standard Metrics")
    print("=" * 60)
    for key, value in split_report["coco_metrics"].items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print(f"  Project Metrics (threshold={args.threshold:.2f})")
    print("=" * 60)
    project_metrics = split_report["project_metrics"]
    print(f"  Detection Rate: {project_metrics['detection_rate']:.1%}")
    print(f"  False Positive Rate: {project_metrics['false_positive_rate']:.1%}")
    print(f"  Precision: {project_metrics['precision']:.1%}")
    print(f"  F1: {project_metrics['f1']:.1%}")

    print("\n" + "=" * 60)
    print("  Frame Presence Metrics")
    print("=" * 60)
    any_pest = split_report["frame_presence_metrics"]["any_pest"]
    print(f"  AUROC: {any_pest['auroc']}")
    print(f"  AUPR: {any_pest['aupr']}")
    print(f"  Precision: {any_pest['precision']}")
    print(f"  Recall: {any_pest['recall']}")
    print(f"  F1: {any_pest['f1']}")

    report = {
        "schema_version": 1,
        "status": "success",
        "workflow": "evaluate",
        "created_at_utc": utc_now_iso(),
        "run_id": artifacts["run_id"],
        "model_path": args.model_path,
        "dataset": collect_dataset_metadata(args.data_dir),
        "environment": build_environment_metadata(device, num_workers=0),
        "code": {"git_commit": try_git_commit()},
        "evaluation": split_report,
    }

    output_json = args.output_json or artifacts["report_path"]
    save_json(output_json, report)
    print(f"\nReport saved to: {output_json}")


if __name__ == "__main__":
    main()
