"""Generator configuration — pure Python, no bpy dependency."""

import os


# Render settings
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
FPS = 10
NUM_FRAMES = 100

# Output directories (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# Sprite directories — place RGBA PNG files here to override procedural fallback.
# Resolution order: sprites/{pest_type}/*.png → procedural PIL ellipse.
SPRITES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites")

# Pest parameters
PEST_TYPES = ["mouse", "rat", "cockroach"]

PEST_PARAMS = {
    "mouse": {
        "body_scale": (0.12, 0.06, 0.05),
        "head_scale": (0.04, 0.035, 0.035),
        "head_offset": (0.13, 0.0, 0.01),
        "color": (0.45, 0.35, 0.25, 1.0),
        "speed": 0.04,
        "min_nz": 0.5,
    },
    "rat": {
        "body_scale": (0.18, 0.09, 0.07),
        "head_scale": (0.06, 0.05, 0.05),
        "head_offset": (0.19, 0.0, 0.015),
        "color": (0.35, 0.25, 0.18, 1.0),
        "speed": 0.03,
        "min_nz": 0.8,
    },
    "cockroach": {
        "body_scale": (0.08, 0.04, 0.015),
        "head_scale": None,  # no separate head
        "head_offset": None,
        "color": (0.25, 0.12, 0.05, 1.0),
        "speed": 0.06,
        "min_nz": 0.1,
    },
}

# Real physical body lengths (metres) used for depth-based scale calculation.
# Body length = longest horizontal dimension of the animal.
PEST_REAL_SIZES_M = {
    "mouse": 0.08,       # ~8 cm body (excluding tail)
    "rat": 0.20,         # ~20 cm body (excluding tail)
    "cockroach": 0.04,   # ~4 cm body
}

# The sprite's local axis that points toward the head (forward direction).
# Used to orient the pest along its movement vector each frame.
# Procedural sprites have the head drawn on the right (+X), so forward_axis = "X".
# Valid values: "X", "-X", "Y", "-Y"
PEST_FORWARD_AXIS = {
    "mouse": "X",
    "rat": "X",
    "cockroach": "X",
}

# Number of pests to place per scene
MIN_PESTS = 1
MAX_PESTS = 3

# Kitchen plane dimensions (world units)
PLANE_WIDTH = 2.0
PLANE_HEIGHT = 1.5
