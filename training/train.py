"""Training entry point for DETR object detection.

Usage (local):
    python -m training.train --data_dir outputs/dataset --epochs 20 --freeze_backbone

Usage (Hugging Face):
    python -m training.train --hf_dataset your-username/pest-detection-dataset --freeze_backbone
    # Downloads the dataset from HF Hub into a local cache, then trains normally.

Tips:
    - On HPC, set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before running to reduce
      CUDA memory fragmentation. This makes PyTorch use smaller growable memory segments instead
      of one large contiguous block, which helps when reserved-but-unallocated memory is large.
      Tradeoff: slightly slower allocations, negligible for single training runs.
      Example: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m training.train ...
    - Default batch size is 4. DETR is memory-heavy; going above ~8 risks OOM on a 16GB GPU.

Arguments:
    Dataset:
        --data_dir          Path to local COCO dataset (default: /cwork/ad641/pest_detection_dataset on HPC)
        --hf_dataset        HF repo instead of local (e.g. adR6x/pest_detection_dataset)
        --hf_revision       Specific HF revision/tag/commit to download

    Model:
        --model_name        Base checkpoint (default: facebook/detr-resnet-50)

    Training:
        --epochs            Number of epochs (default: 20)
        --batch_size        Batch size (default: 4)
        --lr                Learning rate (auto-set based on freeze strategy if not given)
        --freeze_backbone   Freeze entire backbone, train head only
        --partial_freeze N  Unfreeze last N ResNet stages (1-4)
        --augment           Enable color jitter / blur augmentation
        --seed              Random seed (default: 42)
        --num_workers       Dataloader workers (default: 2)

    Evaluation:
        --eval_threshold    Confidence threshold for final eval (default: 0.5)
        --final_eval_split  Split to evaluate after training: train/val/test (default: test)
        --foe               Fraction of epoch at which to run mid-training eval on 300 random
                            images per split. 1.0 = once per epoch, 0.5 = twice per epoch.
                            (default: 1.0)

    Output:
        --output_dir        Where to save best model (default: /cwork/ad641/pest_detection_model/best on HPC)
        --model_repo_dir    Root of model repo (default: /cwork/ad641/pest_detection_model on HPC)
        --evaluation_dir    Where to save eval reports (default: <model_repo_dir>/evaluations)
        --name              Label for this run (default: "experiment")
        --results_csv       CSV file to append summary row (default: results_detection.csv)

Functions:
    _train_one_epoch    -- runs one training epoch, logs per-batch loss to JSONL, fires foe evals
    _evaluate           -- computes validation loss over a dataloader (no grad)
    _save_results_csv   -- appends a summary row to the results CSV file
    _trainer_state_payload      -- builds the optimizer/scheduler state dict for checkpointing
    _build_model_state_record   -- builds a metadata record for best/last JSONL logs
    main                -- CLI entry point: parses args, runs training loop, final eval, reporting
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
    make_model_repo_layout,
    save_model_bundle,
    save_json,
    update_json,
    try_git_commit,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# DETR training helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(model, dataloader, optimizer, device, epoch, iterations_path, global_step,
                     foe_eval_fn=None, foe_interval=None):
    model.train()
    total_loss = 0.0
    n = 0
    total_batches = len(dataloader)
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
        append_jsonl(iterations_path, {
            "global_step": global_step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "train_loss": round(loss.item(), 6),
            "lr_groups": [round(lr, 12) for lr in lr_groups],
        })
        pbar.set_postfix({"loss": f"{total_loss/n:.4f}"})

        if foe_eval_fn and foe_interval and batch_idx % foe_interval == 0:
            foe_eval_fn(
                epoch=epoch,
                batch_idx=batch_idx,
                global_step=global_step,
                epoch_fraction=round(batch_idx / total_batches, 4),
            )
            model.train()

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


def _trainer_state_payload(args, epoch, global_step, best_val_loss, optimizer, scheduler):
    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_args": vars(args),
    }


def _build_model_state_record(
    *,
    record_type,
    model_dir,
    model_repo_dir,
    run_id,
    experiment_name,
    args,
    epoch,
    global_step,
    train_loss,
    val_loss,
    best_val_loss_so_far,
    device,
    git_commit,
    dataset_source,
    dataset_revision,
    train_split,
    val_split,
    run_report_path,
    iterations_path,
):
    model_dir = Path(model_dir)
    model_repo_dir = Path(model_repo_dir)
    return {
        "timestamp_utc": utc_now_iso(),
        "run_id": run_id,
        "experiment_name": experiment_name,
        "record_type": record_type,
        "epoch": epoch,
        "global_step": global_step,
        "model_dir": str(model_dir.resolve()),
        "model_rel_dir": str(model_dir.relative_to(model_repo_dir)),
        "model_repo_dir": str(model_repo_dir.resolve()),
        "run_report_path": str(Path(run_report_path).resolve()),
        "iterations_path": str(Path(iterations_path).resolve()),
        "source_model_name": args.model_name,
        "dataset_source": dataset_source,
        "dataset_revision": dataset_revision,
        "train_split": train_split,
        "val_split": val_split,
        "eval_split": args.final_eval_split,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": DETR_WEIGHT_DECAY,
        "freeze_backbone": args.freeze_backbone,
        "partial_freeze": args.partial_freeze,
        "augment": args.augment,
        "seed": args.seed,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
        "best_val_loss_so_far": round(best_val_loss_so_far, 6),
        "eval_threshold": args.eval_threshold,
        "primary_metric_name": "val_loss",
        "primary_metric_value": round(val_loss, 6),
        "device": str(device),
        "git_commit": git_commit,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train pest detection model")
    parser.add_argument("--model_name", default=DETR_MODEL_NAME)
    _default_data_dir = (
        "/cwork/ad641/pest_detection_dataset" if os.getcwd().startswith("/hpc") else None
    )
    parser.add_argument("--data_dir", default=_default_data_dir,
                        help="Path to COCO dataset with images/ and annotations/ dirs")
    parser.add_argument("--hf_dataset", default=None,
                        help="Hugging Face dataset repo (e.g. username/pest-detection-dataset). "
                             "Downloads to a local cache and sets --data_dir automatically.")
    parser.add_argument("--hf_revision", default=None,
                        help="Optional HF dataset revision/tag/commit to download.")
    parser.add_argument("--batch_size", type=int, default=DETR_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DETR_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=None)
    _on_hpc = os.getcwd().startswith("/hpc")
    _default_model_repo = "/cwork/ad641/pest_detection_model" if _on_hpc else "./pest_detection_model"
    _default_output_dir = "/cwork/ad641/pest_detection_model/best" if _on_hpc else "./detr_finetuned"

    parser.add_argument("--output_dir", default=_default_output_dir)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--partial_freeze", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--name", dest="experiment_name", default="experiment")
    parser.add_argument("--results_csv", default="results_detection.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_threshold", type=float, default=0.5)
    parser.add_argument("--final_eval_split", default="test",
                        choices=["train", "val", "validation", "test"])
    parser.add_argument("--foe", type=float, default=1.0,
                        help="Fraction of epoch at which to run mid-training eval on 300 random "
                             "images per split. 1.0=once per epoch, 0.5=twice per epoch.")
    parser.add_argument("--evaluation_dir", default=None)
    parser.add_argument("--model_repo_dir", default=_default_model_repo,
                        help="Local clone of the HF model repo where best/last are saved.")
    args = parser.parse_args()

    if args.evaluation_dir is None:
        args.evaluation_dir = str(Path(args.model_repo_dir) / "evaluations")

    artifacts = make_run_artifacts(args.experiment_name, args.evaluation_dir, foe=args.foe)

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Written immediately at startup — always exists even if training crashes
    initial_report = {
        "schema_version": 1,
        "run_id": artifacts["run_id"],
        "created_at_utc": artifacts["timestamp"],
        "status": "running",
        "experiment_name": args.experiment_name,
        "artifacts": {
            "run_report": artifacts["run_report_path"],
            "foe_evals": artifacts["foe_path"],
            "iterations": artifacts["iterations_path"],
        },
    }
    save_json(artifacts["run_report_path"], initial_report)

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
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Device: {device}")
        git_commit = try_git_commit()

        data_root = Path(args.data_dir)
        train_paths = resolve_split_paths(data_root, "train")
        val_paths = resolve_split_paths(data_root, "val")
        dataset_metadata = collect_dataset_metadata(data_root)
        model_repo_layout = make_model_repo_layout(args.model_repo_dir, artifacts["run_id"])
        model_repo_root = model_repo_layout["root"]
        dataset_source = args.hf_dataset or str(data_root)
        dataset_revision = args.hf_revision or infer_hf_revision_from_cache_path(data_root)

        strategy = ("head_only" if args.freeze_backbone
                    else f"partial_{args.partial_freeze}stages" if args.partial_freeze
                    else "full_finetune")

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

        # Update report with full context now that everything is initialized
        update_json(artifacts["run_report_path"], {
            "model": {
                "base_checkpoint": args.model_name,
                "output_dir": args.output_dir,
                "strategy": strategy,
                "model_repo_dir": str(model_repo_root),
                "run_model_dir": str(model_repo_layout["run_root"]),
                "runs_root": str(model_repo_layout["runs_root"]),
                "best_dir": str(model_repo_layout["best_dir"]),
                "last_dir": str(model_repo_layout["last_dir"]),
                "best_state_path": str(model_repo_layout["best_state_path"]),
                "last_state_path": str(model_repo_layout["last_state_path"]),
            },
            "dataset": {
                **dataset_metadata,
                "source_type": "hf" if args.hf_dataset else "local",
                "repo_id_or_path": args.hf_dataset or str(data_root),
                "requested_hf_revision": args.hf_revision,
                "resolved_hf_revision": dataset_revision,
            },
            "hyperparams": {
                "batch_size": args.batch_size,
                "epochs_requested": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": DETR_WEIGHT_DECAY,
                "freeze_backbone": args.freeze_backbone,
                "partial_freeze": args.partial_freeze,
                "augment": args.augment,
                "seed": args.seed,
                "foe": args.foe,
                "eval_threshold": args.eval_threshold,
                "final_eval_split": args.final_eval_split,
            },
            "environment": build_environment_metadata(device, args.num_workers),
            "code": {"git_commit": git_commit},
        })

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

        # foe: how many batches between mid-training evals
        foe_interval = max(1, int(len(train_loader) * args.foe))

        def _foe_eval_fn(epoch, batch_idx, global_step, epoch_fraction):
            model.eval()
            results = {}
            for split in ["train", "val", "test"]:
                try:
                    results[split] = evaluate_model_on_split(
                        model, processor, data_root, split, device,
                        threshold=args.eval_threshold, sample_n=300,
                    )
                except FileNotFoundError:
                    pass
            append_jsonl(artifacts["foe_path"], {
                "timestamp_utc": utc_now_iso(),
                "epoch": epoch,
                "batch_idx": batch_idx,
                "global_step": global_step,
                "epoch_fraction": epoch_fraction,
                "evaluation": results,
            })

        best_val_loss = float("inf")
        train_loss = 0.0
        final_val_loss = None
        epoch_history = []
        global_step = 0
        train_seconds_total = 0.0
        if device.type == "cuda":
            gpu_id = device.index
            gpu_name = torch.cuda.get_device_name(gpu_id)
            print(f"GPU {gpu_id}: {gpu_name}")
        print(f"\nStarting DETR training: {args.experiment_name}")
        print(f"Strategy={strategy} | LR={args.lr} | Epochs={args.epochs} | FOE={args.foe}")
        print("-" * 60)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, global_step = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                artifacts["iterations_path"],
                global_step,
                foe_eval_fn=_foe_eval_fn,
                foe_interval=foe_interval,
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

            trainer_state = _trainer_state_payload(
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=min(best_val_loss, val_loss),
                optimizer=optimizer,
                scheduler=scheduler,
            )
            last_dir = save_model_bundle(
                model_repo_layout["last_dir"],
                model,
                processor,
                trainer_state,
            )
            append_jsonl(
                model_repo_layout["last_state_path"],
                _build_model_state_record(
                    record_type="last",
                    model_dir=last_dir,
                    model_repo_dir=model_repo_root,
                    run_id=artifacts["run_id"],
                    experiment_name=args.experiment_name,
                    args=args,
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss_so_far=min(best_val_loss, val_loss),
                    device=device,
                    git_commit=git_commit,
                    dataset_source=dataset_source,
                    dataset_revision=dataset_revision,
                    train_split=train_paths["resolved_image_split"],
                    val_split=val_paths["resolved_image_split"],
                    run_report_path=artifacts["run_report_path"],
                    iterations_path=artifacts["iterations_path"],
                ),
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = save_model_bundle(
                    model_repo_layout["best_dir"],
                    model,
                    processor,
                    trainer_state,
                )
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                    if output_dir.resolve() != best_dir.resolve():
                        save_model_bundle(output_dir, model, processor, trainer_state)
                append_jsonl(
                    model_repo_layout["best_state_path"],
                    _build_model_state_record(
                        record_type="best",
                        model_dir=best_dir,
                        model_repo_dir=model_repo_root,
                        run_id=artifacts["run_id"],
                        experiment_name=args.experiment_name,
                        args=args,
                        epoch=epoch,
                        global_step=global_step,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        best_val_loss_so_far=best_val_loss,
                        device=device,
                        git_commit=git_commit,
                        dataset_source=dataset_source,
                        dataset_revision=dataset_revision,
                        train_split=train_paths["resolved_image_split"],
                        val_split=val_paths["resolved_image_split"],
                        run_report_path=artifacts["run_report_path"],
                        iterations_path=artifacts["iterations_path"],
                    ),
                )
                print(f"  -> Best val loss {val_loss:.4f} — model saved to {best_dir}")

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

        best_model, best_processor = load_model_from_path(str(model_repo_layout["best_dir"]), device)
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

        update_json(artifacts["run_report_path"], {
            "status": "success",
            "training": {
                "epochs_completed": len(epoch_history),
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
            "final_evaluation": final_evaluation,
        })
        print(f"Run report saved to:         {artifacts['run_report_path']}")
        print(f"Mid-training evals saved to: {artifacts['foe_path']}")
        print(f"Iteration log saved to:      {artifacts['iterations_path']}")
        print(f"Model artifacts saved under: {model_repo_root}")
    except Exception as exc:
        update_json(artifacts["run_report_path"], {
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        })
        raise


if __name__ == "__main__":
    main()
