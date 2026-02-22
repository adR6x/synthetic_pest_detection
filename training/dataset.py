"""PyTorch Dataset for pest detection frames and labels."""

import json
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from training.config import LABEL_MAP, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


class PestDetectionDataset(Dataset):
    """Reads rendered frames and JSON labels from outputs/.

    Each sample returns:
        pixel_values: Tensor [3, 224, 224] normalized with ImageNet stats.
        label: Integer class label (primary pest type in the frame).
        bbox: List of bounding boxes (stored for future detection use).
    """

    def __init__(self, frames_root, labels_root):
        """
        Args:
            frames_root: Path to outputs/frames/ (contains job_id subdirs).
            labels_root: Path to outputs/labels/ (contains job_id subdirs).
        """
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

        # Primary pest type = first annotation, or "background" if none
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
