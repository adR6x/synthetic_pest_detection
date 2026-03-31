"""Training hyperparameters and label mapping for both ViT and DETR approaches."""

import torch

# ---------------------------------------------------------------------------
# Shared label definitions
# ---------------------------------------------------------------------------

# ViT classification: string → index
LABEL_MAP = {
    "background": 0,
    "mouse": 1,
    "rat": 2,
    "cockroach": 3,
}
NUM_CLASSES = len(LABEL_MAP)
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# DETR object detection: COCO category ID → name
# Must match compositing.py _CATEGORY_MAP: mouse=1, rat=2, cockroach=3
DETR_ID_TO_LABEL = {1: "mouse", 2: "rat", 3: "cockroach"}
DETR_LABEL_TO_ID = {v: k for k, v in DETR_ID_TO_LABEL.items()}
DETR_NUM_CLASSES = len(DETR_ID_TO_LABEL)

# ---------------------------------------------------------------------------
# ViT model settings
# ---------------------------------------------------------------------------

MODEL_NAME = "google/vit-base-patch16-224"
INPUT_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01

# ---------------------------------------------------------------------------
# DETR model settings
# ---------------------------------------------------------------------------

DETR_MODEL_NAME = "facebook/detr-resnet-50"
DETR_BATCH_SIZE = 4
DETR_NUM_EPOCHS = 20
DETR_WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------------------
# Evaluation thresholds
# ---------------------------------------------------------------------------

DETECTION_RATE_THRESHOLD = 0.80   # target recall ≥ 80%
FPR_THRESHOLD = 0.05              # target false-positive rate < 5%
IOU_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

BBOX_COLORS = {
    "mouse":      (0, 200, 0),
    "cockroach":  (0, 100, 255),
    "rat":        (255, 100, 0),
}

# ---------------------------------------------------------------------------
# Device helper (shared by all scripts)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
