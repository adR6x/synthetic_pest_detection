"""Generator configuration — pure Python, no bpy dependency."""

import os
import shutil
import sys


def _find_blender():
    """Locate the Blender executable.

    Resolution order:
    1. BLENDER_PATH environment variable
    2. 'blender' on the system PATH
    3. Common Windows installation directories (Windows only)
    Falls back to 'blender' so the error message from subprocess is clear.
    """
    env_path = os.environ.get("BLENDER_PATH")
    if env_path:
        return env_path

    which = shutil.which("blender")
    if which:
        return which

    if sys.platform == "win32":
        program_files = [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        ]
        for pf in program_files:
            blender_root = os.path.join(pf, "Blender Foundation")
            if os.path.isdir(blender_root):
                # Pick the highest-versioned folder
                versions = sorted(os.listdir(blender_root), reverse=True)
                for version_dir in versions:
                    candidate = os.path.join(blender_root, version_dir, "blender.exe")
                    if os.path.isfile(candidate):
                        return candidate

    return "blender"  # fallback — subprocess will raise a clear FileNotFoundError


BLENDER_PATH = _find_blender()

# Render settings
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
FPS = 2
NUM_FRAMES = 10

# Depth estimation model (runs in system Python, not Blender)
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# Output directories (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# Pest parameters
PEST_TYPES = ["mouse", "rat", "cockroach"]

PEST_PARAMS = {
    "mouse": {
        "body_scale": (0.12, 0.06, 0.05),
        "head_scale": (0.04, 0.035, 0.035),
        "head_offset": (0.13, 0.0, 0.01),
        "color": (0.45, 0.35, 0.25, 1.0),
        "speed": 0.08,
        "min_nz": 0.5,
    },
    "rat": {
        "body_scale": (0.18, 0.09, 0.07),
        "head_scale": (0.06, 0.05, 0.05),
        "head_offset": (0.19, 0.0, 0.015),
        "color": (0.35, 0.25, 0.18, 1.0),
        "speed": 0.06,
        "min_nz": 0.8,
    },
    "cockroach": {
        "body_scale": (0.08, 0.04, 0.015),
        "head_scale": None,  # no separate head
        "head_offset": None,
        "color": (0.25, 0.12, 0.05, 1.0),
        "speed": 0.12,
        "min_nz": 0.1,
    },
}

# Number of pests to place per scene
MIN_PESTS = 1
MAX_PESTS = 3

# Kitchen plane dimensions (world units)
PLANE_WIDTH = 2.0
PLANE_HEIGHT = 1.5
