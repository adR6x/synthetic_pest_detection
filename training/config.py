"""Training hyperparameters and label mapping."""

# Label mapping
LABEL_MAP = {
    "background": 0,
    "mouse": 1,
    "rat": 2,
    "cockroach": 3,
}
NUM_CLASSES = len(LABEL_MAP)
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Model
MODEL_NAME = "google/vit-base-patch16-224"
INPUT_SIZE = 224

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
