"""Dataset utilities for DETR and YOLOv8 object detection."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CocoDetection

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
        │   ├── train/  *.png
        │   ├── val/    *.png
        │   └── test/   *.png
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


# ---------------------------------------------------------------------------
# YOLOv8 on-the-fly COCO loading — no .txt label files written to disk
# ---------------------------------------------------------------------------

def _build_coco_label_index(img_path, ann_file, cat_id_to_yolo):
    """Load COCO JSON and return {abs_img_path: {shape, cls, bboxes}}.

    Bboxes are stored as normalized xywh (center) matching YOLO format.
    """
    img_path = Path(img_path).resolve()
    print(f"Loading annotations from {ann_file} ...")
    with open(ann_file) as f:
        coco = json.load(f)

    img_meta = {img["id"]: img for img in coco["images"]}

    anns_by_image: dict = defaultdict(list)
    for ann in coco["annotations"]:
        img = img_meta[ann["image_id"]]
        W, H = img["width"], img["height"]
        x, y, w, h = ann["bbox"]
        cls = cat_id_to_yolo[ann["category_id"]]
        anns_by_image[ann["image_id"]].append([cls, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H])

    index = {}
    for img_id, img in img_meta.items():
        abs_path = str(img_path / img["file_name"])
        rows = anns_by_image.get(img_id, [])
        if rows:
            arr = np.array(rows, dtype=np.float32)
            cls_arr = arr[:, :1]
            bbox_arr = arr[:, 1:]
        else:
            cls_arr = np.zeros((0, 1), dtype=np.float32)
            bbox_arr = np.zeros((0, 4), dtype=np.float32)
        index[abs_path] = {
            "shape": (img["height"], img["width"]),
            "cls": cls_arr,
            "bboxes": bbox_arr,
        }

    return index


def write_yolo_yaml(data_dir, categories):
    """Write data.yaml for Ultralytics. Returns the yaml path."""
    data_dir = Path(data_dir)
    names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
    yaml_path = data_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {data_dir}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    print(f"data.yaml written to: {yaml_path}")
    return yaml_path


class CocoYOLODataset:
    """Wraps YOLODataset to load labels from COCO JSON on the fly.

    Constructed lazily so the import of ultralytics is deferred until training.
    """

    def __new__(cls, *args, ann_file, cat_id_to_yolo, **kwargs):
        from ultralytics.data.dataset import YOLODataset

        class _CocoYOLODataset(YOLODataset):
            def __init__(self, *a, ann_file, cat_id_to_yolo, **kw):
                img_path = a[0] if a else kw.get("img_path")
                self._label_index = _build_coco_label_index(img_path, ann_file, cat_id_to_yolo)
                super().__init__(*a, **kw)

            def get_labels(self):
                labels = []
                for im_file in self.im_files:
                    entry = self._label_index.get(str(Path(im_file).resolve()))
                    if entry is None:
                        entry = {
                            "shape": (640, 640),
                            "cls": np.zeros((0, 1), dtype=np.float32),
                            "bboxes": np.zeros((0, 4), dtype=np.float32),
                        }
                    labels.append({
                        "im_file": im_file,
                        "shape": entry["shape"],
                        "cls": entry["cls"],
                        "bboxes": entry["bboxes"],
                        "segments": [],
                        "keypoints": None,
                        "normalized": True,
                        "bbox_format": "xywh",
                    })
                return labels

        return _CocoYOLODataset(*args, ann_file=ann_file, cat_id_to_yolo=cat_id_to_yolo, **kwargs)


def make_coco_trainer(ann_files, cat_id_to_yolo):
    """Return a DetectionTrainer subclass that loads labels from COCO JSON on the fly.

    ann_files: {'train': '/path/train.json', 'val': '/path/val.json'}
    cat_id_to_yolo: {coco_cat_id: yolo_class_index}
    """
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import colorstr

    class _CocoDetectionTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            m = self.model.module if (self.model and hasattr(self.model, "module")) else self.model
            gs = max(int(m.stride.max() if m else 0), 32)
            split = "val" if "val" in str(img_path) else "train"
            return CocoYOLODataset(
                img_path=str(img_path),
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=mode == "val",
                cache=self.args.cache if mode == "train" else None,
                single_cls=self.args.single_cls or False,
                stride=int(gs),
                pad=0.0 if mode == "train" else 0.5,
                prefix=colorstr(f"{mode}: "),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction if mode == "train" else 1.0,
                ann_file=ann_files[split],
                cat_id_to_yolo=cat_id_to_yolo,
            )

    return _CocoDetectionTrainer
