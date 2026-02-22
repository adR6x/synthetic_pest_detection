"""ViT model wrapper using HuggingFace transformers."""

from transformers import ViTForImageClassification

from training.config import MODEL_NAME, NUM_CLASSES, ID_TO_LABEL, LABEL_MAP


def create_model():
    """Create a ViT model for pest classification.

    Returns:
        ViTForImageClassification with NUM_CLASSES outputs.
    """
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
        ignore_mismatched_sizes=True,
    )
    return model
