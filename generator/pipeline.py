"""Pipeline orchestrator — runs in system Python (no bpy).

Calls Blender as a subprocess and assembles rendered frames into an MP4.
"""

import json
import os
import random
import subprocess
import uuid

import cv2

from generator.config import (
    FRAMES_DIR,
    LABELS_DIR,
    MAX_PESTS,
    MIN_PESTS,
    NUM_FRAMES,
    OUTPUT_DIR,
    PEST_PARAMS,
    PEST_TYPES,
    PLANE_HEIGHT,
    PLANE_WIDTH,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    UPLOAD_DIR,
    VIDEOS_DIR,
    FPS,
)


def generate_video(image_path, job_id=None, frames_root=None, labels_root=None, videos_root=None):
    """Run the full generation pipeline for one kitchen image.

    Args:
        image_path: Absolute path to the uploaded kitchen image.
        job_id: Optional job identifier. Generated if not provided.
        frames_root: Override for frames output directory.
        labels_root: Override for labels output directory.
        videos_root: Override for videos output directory.

    Returns:
        Dict with job_id, video_path, frames_dir, labels_dir.
    """
    if job_id is None:
        job_id = uuid.uuid4().hex[:8]

    frames_dir = os.path.join(frames_root or FRAMES_DIR, job_id)
    labels_dir = os.path.join(labels_root or LABELS_DIR, job_id)
    video_path = os.path.join(videos_root or VIDEOS_DIR, f"{job_id}.mp4")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # Choose random pests
    num_pests = random.randint(MIN_PESTS, MAX_PESTS)
    pest_configs = []
    for i in range(num_pests):
        pest_type = random.choice(PEST_TYPES)
        params = dict(PEST_PARAMS[pest_type])
        # Convert tuples to lists for JSON serialization
        for key in ["body_scale", "head_scale", "head_offset", "color"]:
            if params[key] is not None:
                params[key] = list(params[key])
        pest_configs.append({"type": pest_type, "params": params})

    # Build config for Blender
    config = {
        "image_path": os.path.abspath(image_path),
        "job_id": job_id,
        "frames_dir": os.path.abspath(frames_dir),
        "labels_dir": os.path.abspath(labels_dir),
        "width": RENDER_WIDTH,
        "height": RENDER_HEIGHT,
        "num_frames": NUM_FRAMES,
        "plane_width": PLANE_WIDTH,
        "plane_height": PLANE_HEIGHT,
        "pests": pest_configs,
    }
    config_json = json.dumps(config)

    # Call Blender as subprocess
    blender_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "blender_script.py"
    )

    cmd = [
        "blender",
        "--background",
        "--python", blender_script,
        "--", config_json,
    ]

    print(f"Running Blender for job {job_id}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"Blender stderr: {result.stderr}")
        raise RuntimeError(f"Blender failed with return code {result.returncode}")

    print(result.stdout)

    # Assemble frames into MP4
    _assemble_video(frames_dir, video_path)

    return {
        "job_id": job_id,
        "video_path": video_path,
        "frames_dir": frames_dir,
        "labels_dir": labels_dir,
    }


def _assemble_video(frames_dir, video_path):
    """Combine rendered PNG frames into an MP4 video using OpenCV.

    Args:
        frames_dir: Directory containing frame_XXXX.png files.
        video_path: Output MP4 path.
    """
    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.endswith(".png")
    )
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

    for fname in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, fname))
        writer.write(frame)

    writer.release()
    print(f"Video assembled: {video_path} ({len(frame_files)} frames)")
