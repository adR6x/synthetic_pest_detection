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
    PEST_PARAMS,
    PEST_TYPES,
    PLANE_HEIGHT,
    PLANE_WIDTH,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    VIDEOS_DIR,
    FPS,
)
from generator.depth_estimator import (
    build_surface_probability_map,
    compute_surface_normals,
    estimate_depth,
    estimate_surface_normals_pretrained,
    save_depth_preview,
    save_mask_preview,
    save_probability_preview,
    save_surface_preview,
    sample_pest_positions_from_probability,
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
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Choose random pests
    num_pests = random.randint(MIN_PESTS, MAX_PESTS)
    pest_configs = []
    for _ in range(num_pests):
        pest_type = random.choice(PEST_TYPES)
        params = dict(PEST_PARAMS[pest_type])
        # Convert tuples to lists for JSON serialization
        for key in ["body_scale", "head_scale", "head_offset", "color"]:
            if params[key] is not None:
                params[key] = list(params[key])
        pest_configs.append({"type": pest_type, "params": params})

    # Depth-aware placement (system Python side)
    depth_map = estimate_depth(image_path)
    save_depth_preview(depth_map, os.path.join(frames_dir, "depth_preview.jpg"))
    normals = compute_surface_normals(depth_map)  # kept for existing preview/debug use
    predicted_normals = estimate_surface_normals_pretrained(image_path)
    save_surface_preview(predicted_normals, os.path.join(frames_dir, "surface_preview.jpg"))
    save_probability_preview(normals, os.path.join(frames_dir, "probability_preview.jpg"))

    mask_path_cache = {}
    mask_cache = {}
    for pest_cfg in pest_configs:
        pest_type = pest_cfg["type"]
        min_nz = float(pest_cfg["params"].get("min_nz", 0.8))
        mask_key = (pest_type, min_nz)

        if mask_key not in mask_cache:
            placement_prob = build_surface_probability_map(predicted_normals, min_nz)
            mask_cache[mask_key] = placement_prob
            threshold_tag = str(min_nz).replace(".", "p")
            mask_path = os.path.join(
                frames_dir, f"placement_mask_{pest_type}_{threshold_tag}.png"
            )
            save_mask_preview(placement_prob, mask_path)
            mask_path_cache[mask_key] = os.path.abspath(mask_path)

        start_position = sample_pest_positions_from_probability(
            mask_cache[mask_key],
            n=1,
            plane_width=PLANE_WIDTH,
            plane_height=PLANE_HEIGHT,
        )[0]
        pest_cfg["start_position"] = [float(start_position[0]), float(start_position[1])]
        pest_cfg["placement_mask_path"] = mask_path_cache[mask_key]

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
    if result.stderr.strip():
        print(f"Blender stderr (non-fatal): {result.stderr}")

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
