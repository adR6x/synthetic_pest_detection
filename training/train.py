"""Training loop skeleton for pest classification with ViT."""

import os
import sys

import torch
from torch.utils.data import DataLoader

from training.config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, NUM_CLASSES
from training.dataset import PestDetectionDataset
from training.model import create_model

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_ROOT = os.path.join(PROJECT_ROOT, "outputs", "frames")
LABELS_ROOT = os.path.join(PROJECT_ROOT, "outputs", "labels")


def collate_fn(batch):
    """Custom collate that stacks pixel_values and labels, keeps bboxes as list."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    bboxes = [item["bboxes"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels, "bboxes": bboxes}


def train():
    dataset = PestDetectionDataset(FRAMES_ROOT, LABELS_ROOT)

    if len(dataset) == 0:
        print("No training data found in outputs/frames/ and outputs/labels/.")
        print("Generate data first: python -m app.main, then upload an image.")
        sys.exit(0)

    print(f"Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch}/{NUM_EPOCHS} — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    train()
