"""Training entry point for DETR object detection.

Usage (local):
    python -m training.train --data_dir outputs/dataset --epochs 20 --freeze_backbone

Usage (Hugging Face):
    python -m training.train --hf_dataset your-username/pest-detection-dataset --freeze_backbone
    # Downloads the dataset from HF Hub into a local cache, then trains normally.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.config import (
    get_device,
    DETR_BATCH_SIZE, DETR_NUM_EPOCHS, DETR_WEIGHT_DECAY, DETR_MODEL_NAME,
)
from training.data_utils import collect_dataset_metadata, infer_hf_revision_from_cache_path, resolve_split_paths
from training.dataset import CocoDetectionDETR, detr_collate_fn
from training.metrics import evaluate_model_on_split, load_model_from_path
from training.model import create_detr_model, apply_freeze_strategy
from training.reporting import (
    append_jsonl,
    build_environment_metadata,
    make_run_artifacts,
    save_json,
    try_git_commit,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# DETR training helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(model, dataloader, optimizer, device, epoch, iteration_log_path, global_step):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar, start=1):
        pv = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in batch["labels"]]
        outputs = model(pixel_values=pv, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        global_step += 1
        lr_groups = [group["lr"] for group in optimizer.param_groups]
        append_jsonl(iteration_log_path, {
            "global_step": global_step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "train_loss": round(loss.item(), 6),
            "running_train_loss": round(total_loss / n, 6),
            "lr_groups": [round(lr, 12) for lr in lr_groups],
        })
        pbar.set_postfix({"loss": f"{total_loss/n:.4f}"})
    return total_loss / max(n, 1), global_step


@torch.no_grad()
def _evaluate(model, dataloader, device, split_name="Val"):
    model.eval()
    total_loss = 0.0
    n = 0
    pbar = tqdm(dataloader, desc=f"[{split_name}]")
    for batch in pbar:
        pv = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in batch["labels"]]
        outputs = model(pixel_values=pv, labels=labels)
        total_loss += outputs.loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{total_loss/n:.4f}"})
    return total_loss / max(n, 1)


def _save_results_csv(csv_path, row):
    fieldnames = ["experiment_name", "strategy", "epochs_trained",
                  "best_val_loss", "final_train_loss", "lr"]
    exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train pest detection model")
    parser.add_argument("--model_name", default=DETR_MODEL_NAME)
    parser.add_argument("--data_dir", default=None,
                        help="Path to COCO dataset with images/ and annotations/ dirs")
    parser.add_argument("--hf_dataset", default=None,
                        help="Hugging Face dataset repo (e.g. username/pest-detection-dataset). "
                             "Downloads to a local cache and sets --data_dir automatically.")
    parser.add_argument("--hf_revision", default=None,
                        help="Optional HF dataset revision/tag/commit to download.")
    parser.add_argument("--batch_size", type=int, default=DETR_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DETR_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", default="./detr_finetuned")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--partial_freeze", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--experiment_name", default="experiment")
    parser.add_argument("--results_csv", default="results_detection.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_threshold", type=float, default=0.5)
    parser.add_argument("--final_eval_split", default="test",
                        choices=["train", "val", "validation", "test"])
    parser.add_argument("--evaluation_dir", default=None)
    args = parser.parse_args()

    artifacts = make_run_artifacts(args.experiment_name, args.evaluation_dir)

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    report_context = {
        "schema_version": 1,
        "run_id": artifacts["run_id"],
        "created_at_utc": utc_now_iso(),
        "workflow": "train_and_eval",
        "experiment_name": args.experiment_name,
        "artifacts": {
            "evaluation_report": artifacts["report_path"],
            "iteration_metrics_file": artifacts["iteration_log_path"],
        },
    }

    try:
        if args.hf_dataset is not None:
            from huggingface_hub import snapshot_download
            print(f"Downloading dataset from HF Hub: {args.hf_dataset}")
            args.data_dir = snapshot_download(
                repo_id=args.hf_dataset,
                repo_type="dataset",
                revision=args.hf_revision,
            )
            print(f"Dataset cached at: {args.data_dir}")

        if args.data_dir is None:
            parser.error("--data_dir or --hf_dataset is required for DETR training")

        if args.lr is None:
            if args.freeze_backbone:
                args.lr = 1e-4
            elif args.partial_freeze > 0:
                args.lr = 5e-5
            else:
                args.lr = 1e-5

        device = get_device()
        print(f"Device: {device}")

        data_root = Path(args.data_dir)
        train_paths = resolve_split_paths(data_root, "train")
        val_paths = resolve_split_paths(data_root, "val")
        dataset_metadata = collect_dataset_metadata(data_root)

        print(f"Loading model: {args.model_name}")
        model, processor = create_detr_model(args.model_name)
        apply_freeze_strategy(model, args.freeze_backbone, args.partial_freeze)
        model.to(device)

        train_ds = CocoDetectionDETR(
            str(train_paths["image_dir"]),
            str(train_paths["annotation_path"]),
            processor,
            augment=args.augment,
        )
        val_ds = CocoDetectionDETR(
            str(val_paths["image_dir"]),
            str(val_paths["annotation_path"]),
            processor,
            augment=False,
        )
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=detr_collate_fn,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=detr_collate_fn,
            pin_memory=pin_memory,
        )

        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters()
                        if "backbone" not in n and p.requires_grad], "lr": args.lr},
            {"params": [p for n, p in model.named_parameters()
                        if "backbone" in n and p.requires_grad],     "lr": args.lr * 0.1},
        ], weight_decay=DETR_WEIGHT_DECAY)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        strategy = ("head_only" if args.freeze_backbone
                    else f"partial_{args.partial_freeze}stages" if args.partial_freeze
                    else "full_finetune")

        best_val_loss = float("inf")
        train_loss = 0.0
        final_val_loss = None
        epoch_history = []
        global_step = 0
        train_seconds_total = 0.0
        print(f"\nStarting DETR training: {args.experiment_name}")
        print(f"Strategy={strategy} | LR={args.lr} | Epochs={args.epochs}")
        print("-" * 60)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, global_step = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                artifacts["iteration_log_path"],
                global_step,
            )
            val_loss = _evaluate(model, val_loader, device)
            scheduler.step()
            epoch_seconds = time.time() - t0
            train_seconds_total += epoch_seconds
            final_val_loss = val_loss
            lr_groups = [group["lr"] for group in optimizer.param_groups]
            epoch_history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "epoch_time_sec": round(epoch_seconds, 2),
                "lr_groups": [round(lr, 12) for lr in lr_groups],
                "global_step": global_step,
            })
            print(f"Epoch {epoch:3d}: Train={train_loss:.4f}  Val={val_loss:.4f}  "
                  f"({epoch_seconds:.0f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(args.output_dir, exist_ok=True)
                model.save_pretrained(args.output_dir)
                processor.save_pretrained(args.output_dir)
                print(f"  -> Best val loss {val_loss:.4f} — model saved to {args.output_dir}")

        print("-" * 60)
        print(f"Done. Best val loss: {best_val_loss:.4f}")

        _save_results_csv(args.results_csv, {
            "experiment_name": args.experiment_name,
            "strategy": strategy,
            "epochs_trained": args.epochs,
            "best_val_loss": round(best_val_loss, 4),
            "final_train_loss": round(train_loss, 4),
            "lr": args.lr,
        })

        best_model, best_processor = load_model_from_path(args.output_dir, device)
        eval_t0 = time.time()
        final_evaluation = evaluate_model_on_split(
            best_model,
            best_processor,
            data_root,
            args.final_eval_split,
            device,
            threshold=args.eval_threshold,
        )
        eval_seconds_total = time.time() - eval_t0

        report = {
            **report_context,
            "status": "success",
            "model": {
                "base_checkpoint": args.model_name,
                "output_dir": args.output_dir,
                "strategy": strategy,
            },
            "dataset": {
                **dataset_metadata,
                "source_type": "hf" if args.hf_dataset else "local",
                "repo_id_or_path": args.hf_dataset or str(data_root),
                "requested_hf_revision": args.hf_revision,
                "resolved_hf_revision": infer_hf_revision_from_cache_path(data_root),
            },
            "training": {
                "epochs_requested": args.epochs,
                "epochs_completed": len(epoch_history),
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": DETR_WEIGHT_DECAY,
                "freeze_backbone": args.freeze_backbone,
                "partial_freeze": args.partial_freeze,
                "augment": args.augment,
                "seed": args.seed,
                "best_val_loss": round(best_val_loss, 6),
                "final_train_loss": round(train_loss, 6),
                "final_val_loss": round(final_val_loss, 6) if final_val_loss is not None else None,
                "epoch_history": epoch_history,
            },
            "timing": {
                "train_seconds_total": round(train_seconds_total, 2),
                "eval_seconds_total": round(eval_seconds_total, 2),
                "seconds_per_epoch": round(train_seconds_total / max(len(epoch_history), 1), 2),
            },
            "evaluation": final_evaluation,
            "environment": build_environment_metadata(device, args.num_workers),
            "code": {"git_commit": try_git_commit()},
        }
        save_json(artifacts["report_path"], report)
        print(f"Evaluation report saved to: {artifacts['report_path']}")
        print(f"Iteration metrics saved to: {artifacts['iteration_log_path']}")
    except Exception as exc:
        failure_report = {
            **report_context,
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "code": {"git_commit": try_git_commit()},
        }
        save_json(artifacts["report_path"], failure_report)
        raise


if __name__ == "__main__":
    main()
