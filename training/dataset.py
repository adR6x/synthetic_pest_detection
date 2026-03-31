"""Dataset classes for both ViT classification and DETR object detection."""

import json
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CocoDetection

from training.config import LABEL_MAP, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# ViT dataset — reads per-frame PNG + JSON from outputs/frames/ and labels/
# ---------------------------------------------------------------------------

class PestDetectionDataset(Dataset):
    """Reads rendered frames and JSON labels from outputs/.

    Each sample returns:
        pixel_values: Tensor [3, 224, 224] normalized with ImageNet stats.
        label: Integer class label (primary pest type in the frame).
        bboxes: List of bounding boxes (stored for future detection use).
    """

    def __init__(self, frames_root, labels_root):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        if not os.path.exists(frames_root):
            return

        for job_id in sorted(os.listdir(frames_root)):
            job_frames = os.path.join(frames_root, job_id)
            job_labels = os.path.join(labels_root, job_id)
            if not os.path.isdir(job_frames):
                continue

            for fname in sorted(os.listdir(job_frames)):
                if not fname.endswith(".png"):
                    continue
                frame_path = os.path.join(job_frames, fname)
                label_name = fname.replace(".png", ".json")
                label_path = os.path.join(job_labels, label_name)

                if os.path.exists(label_path):
                    self.samples.append((frame_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, label_path = self.samples[idx]

        image = Image.open(frame_path).convert("RGB")
        pixel_values = self.transform(image)

        with open(label_path) as f:
            label_data = json.load(f)

        annotations = label_data.get("annotations", [])
        bboxes = [a["bbox"] for a in annotations]

        if annotations:
            primary_type = annotations[0]["pest_type"]
        else:
            primary_type = "background"
        label = LABEL_MAP.get(primary_type, 0)

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(label, dtype=torch.long),
            "bboxes": bboxes,
        }


# ---------------------------------------------------------------------------
# DETR dataset — reads COCO-format annotations produced by the generator
# ---------------------------------------------------------------------------

def build_augmentation():
    """Augmentation transforms to reduce synthetic-to-real domain gap."""
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    ])


class CocoDetectionDETR(CocoDetection):
    """Wraps torchvision CocoDetection to return DETR-compatible batches.

    Expected dataset layout (produced by the generator pipeline):
        data_dir/
        ├── images/
        │   ├── train/  *.jpg
        │   ├── val/    *.jpg
        │   └── test/   *.jpg
        └── annotations/
            ├── train.json
            ├── val.json
            └── test.json
    """

    def __init__(self, img_folder, ann_file, processor, augment=False):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.augment_transform = build_augmentation() if augment else None

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)

        if self.augment_transform is not None:
            img = self.augment_transform(img)

        image_id = self.ids[idx]
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))

        annotations = []
        for ann in ann_info:
            if ann.get("iscrowd", 0):
                continue
            annotations.append({
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": 0,
            })

        target_dict = {"image_id": image_id, "annotations": annotations}

        encoding = self.processor(
            images=img,
            annotations=[target_dict],
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]
        return pixel_values, labels


def detr_collate_fn(batch):
    """Collate function for CocoDetectionDETR batches."""
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}
