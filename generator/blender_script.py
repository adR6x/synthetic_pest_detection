"""Blender entry point — runs INSIDE Blender via:
    blender --background --python blender_script.py -- <config_json>

Coordinates scene setup, pest creation, animation, rendering, and labeling.
"""

import json
import os
import random
import sys

# Add this script's directory to sys.path so we can import sibling modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bpy

from scene_setup import (
    clear_scene,
    configure_render,
    setup_background_plane,
    setup_camera,
    setup_lighting,
)
from pest_models import create_pest
from pest_animation import animate_pest
from labeler import generate_frame_label


def main():
    # Parse config from command-line args (everything after "--")
    argv = sys.argv
    separator_idx = argv.index("--") if "--" in argv else -1
    if separator_idx == -1 or separator_idx + 1 >= len(argv):
        print("ERROR: No config JSON provided after '--'")
        sys.exit(1)

    config = json.loads(argv[separator_idx + 1])

    image_path = config["image_path"]
    job_id = config["job_id"]
    frames_dir = config["frames_dir"]
    labels_dir = config["labels_dir"]
    width = config.get("width", 640)
    height = config.get("height", 480)
    num_frames = config.get("num_frames", 10)
    plane_width = config.get("plane_width", 2.0)
    plane_height = config.get("plane_height", 1.5)
    pest_configs = config.get("pests", [])

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 1. Scene setup
    clear_scene()
    setup_background_plane(image_path, plane_width, plane_height)
    camera = setup_camera(plane_width, plane_height)
    setup_lighting()
    configure_render(width, height)

    # 2. Create pests
    pests = []  # list of (pest_type, pest_object)
    for i, pest_cfg in enumerate(pest_configs):
        pest_type = pest_cfg["type"]
        params = pest_cfg["params"]
        pest_obj = create_pest(pest_type, params, i)
        pests.append((pest_type, pest_obj))

    # 3. Animate pests
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = num_frames

    for pest_type, pest_obj in pests:
        pest_cfg = next(p for p in pest_configs if p["type"] == pest_type)
        speed = pest_cfg["params"].get("speed", 0.08)
        animate_pest(pest_obj, num_frames, plane_width, plane_height, speed)

    # 4. Render each frame and generate labels
    for frame in range(1, num_frames + 1):
        scene.frame_set(frame)

        # Render frame
        frame_path = os.path.join(frames_dir, f"frame_{frame:04d}.png")
        scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)

        # Generate label
        generate_frame_label(pests, scene, camera, frame, labels_dir)

    print(f"Blender rendering complete: {num_frames} frames for job {job_id}")


if __name__ == "__main__":
    main()
