"""Pipeline orchestrator — runs in system Python (no bpy).

Performs depth/normal analysis and assembles rendered frames into an MP4.
"""

import os
import random
import shutil
import subprocess
import time as _time
import uuid

import cv2
from generator.config import (
    FRAMES_DIR,
    LABELS_DIR,
    MAX_PESTS,
    MIN_PESTS,
    NUM_FRAMES,
    PEST_FORWARD_AXIS,
    PEST_PARAMS,
    PEST_REAL_SIZES_M,
    PEST_TYPES,
    PLANE_HEIGHT,
    PLANE_WIDTH,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    SPRITES_DIR,
    VIDEOS_DIR,
    FPS,
)
from concurrent.futures import ThreadPoolExecutor

from generator.compositing import composite_frames

from generator.depth_estimator import (
    build_movement_mask,
    build_surface_group_masks,
    compute_inference_strategy,
    estimate_gravity,
    estimate_metric3d,
    save_depth_preview,
    save_gravity_preview,
    save_movement_mask_preview,
    save_surface_preview,
    sample_pest_positions_from_probability,
)

SURFACE_GROUPS = ("up", "side_left", "side_right", "side_toward", "down")
DEFAULT_SPAWN_PROBS = {
    "up": 0.78,
    "side_left": 0.067,
    "side_right": 0.067,
    "side_toward": 0.066,
    "down": 0.02,
}


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
        # Convert world-units/second speeds to per-frame world units using FPS.
        # This ensures max speed is physically grounded regardless of frame rate.
        base_speed_wps = float(params.pop("base_speed_wps", 0.1))
        max_speed_wps  = float(params.pop("max_speed_wps",  1.0))
        params["base_speed_wps"] = base_speed_wps
        params["max_speed_wps"]  = max_speed_wps
        params["speed"]          = base_speed_wps / FPS
        params["max_step_world"] = max_speed_wps / FPS
        pest_configs.append({"type": pest_type, "params": params})

    # Depth-aware placement — Metric3D v2 yields depth + normals in one pass.
    # Gravity estimation (classical VP) is independent and can run in parallel.
    # Strategy: parallel on CPU or multi-GPU, sequential on single GPU.
    strategy = compute_inference_strategy()
    print(f"Inference strategy: {strategy}")

    if strategy == "parallel":
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_metric3d = ex.submit(estimate_metric3d, image_path)
            f_gravity  = ex.submit(estimate_gravity, image_path)
        metric3d_result = f_metric3d.result()
        gravity_result  = f_gravity.result()
    else:
        metric3d_result = estimate_metric3d(image_path)
        gravity_result  = estimate_gravity(image_path)

    depth_map         = metric3d_result["depth"]
    predicted_normals = metric3d_result["normals"]
    focal_length_px   = metric3d_result["fx"]
    img_h, img_w      = depth_map.shape

    save_depth_preview(depth_map, os.path.join(frames_dir, "depth_preview.jpg"))
    save_surface_preview(predicted_normals, os.path.join(frames_dir, "surface_preview.jpg"))
    save_gravity_preview(image_path, gravity_result, os.path.join(frames_dir, "gravity_preview.jpg"))
    print(f"Gravity estimate: up={gravity_result['gravity_cam'].tolist()}  "
          f"conf={gravity_result['confidence']:.2f}")

    # Build directional surface-group probability maps once (shared across pests).
    surface_group_masks = build_surface_group_masks(predicted_normals)
    for pest_idx, pest_cfg in enumerate(pest_configs):
        pest_type   = pest_cfg["type"]
        params      = pest_cfg["params"]
        spawn_probs = _resolve_spawn_probs(params.get("spawn_probs"))
        stickiness  = float(params.get("surface_stickiness", 0.97))

        # Choose which surface group this pest spawns on (weighted random).
        groups  = [g for g in SURFACE_GROUPS if g in surface_group_masks]
        if not groups:
            groups = list(surface_group_masks.keys())
        weights = [spawn_probs.get(g, 0.0) for g in groups]
        total_w = sum(weights)
        if total_w <= 0:
            weights = [1.0 if g == "up" else 0.0 for g in groups]
        if sum(weights) <= 0:
            weights = [1.0 / max(len(groups), 1)] * len(groups)
        spawn_surface = random.choices(groups, weights=weights)[0]

        # Sample start position from the chosen surface group's probability map.
        start_position = sample_pest_positions_from_probability(
            surface_group_masks[spawn_surface],
            n=1,
            plane_width=PLANE_WIDTH,
            plane_height=PLANE_HEIGHT,
        )[0]

        # Build blended movement mask: high weight on spawn surface, low on others.
        movement_prob = build_movement_mask(surface_group_masks, spawn_surface, stickiness)
        mask_filename = f"movement_mask_{pest_type}_{pest_idx}.png"
        mask_path     = os.path.abspath(os.path.join(frames_dir, mask_filename))
        save_movement_mask_preview(movement_prob, spawn_surface, mask_path)

        pest_cfg["start_position"]    = [float(start_position[0]), float(start_position[1])]
        pest_cfg["placement_mask_path"] = mask_path

        # Depth-based scale: apparent pixel width = real_size_m * fx / depth_m
        # => blender_scale = real_size_m * fx * PLANE_WIDTH / (depth_m * RENDER_WIDTH)
        wx, wy = start_position
        px = int(round((wx / PLANE_WIDTH + 0.5) * (img_w - 1)))
        py = int(round((0.5 - wy / PLANE_HEIGHT) * (img_h - 1)))
        px = max(0, min(img_w - 1, px))
        py = max(0, min(img_h - 1, py))
        depth_val = float(depth_map[py, px])

        real_size      = PEST_REAL_SIZES_M[pest_type]
        default_body_x = PEST_PARAMS[pest_type]["body_scale"][0]
        if depth_val > 0.1:
            raw_scale = real_size * focal_length_px * PLANE_WIDTH / (depth_val * RENDER_WIDTH)
            blender_scale = float(max(default_body_x * 0.4, min(default_body_x * 3.0, raw_scale)))
        else:
            blender_scale = default_body_x

        params["blender_scale"]  = blender_scale
        params["forward_axis"]   = PEST_FORWARD_AXIS.get(pest_type, "X")
        # Pass depth info through for per-frame depth-aware speed cap in compute_walk.
        params["focal_length_px"] = float(focal_length_px)

    # Composite pest sprites onto background image
    print(f"Running 2D compositing for job {job_id}...")
    _t0 = _time.monotonic()
    composite_frames(
        image_path=os.path.abspath(image_path),
        pest_configs=pest_configs,
        frames_dir=frames_dir,
        labels_dir=labels_dir,
        num_frames=NUM_FRAMES,
        sprites_dir=SPRITES_DIR,
        render_width=RENDER_WIDTH,
        render_height=RENDER_HEIGHT,
        plane_width=PLANE_WIDTH,
        plane_height=PLANE_HEIGHT,
        depth_map=depth_map,
        fps=FPS,
        surface_group_masks=surface_group_masks,
        normals=predicted_normals,
    )
    print(f"Compositing finished in {_time.monotonic() - _t0:.1f}s (job {job_id})")

    # Assemble frames into MP4
    _assemble_video(frames_dir, video_path)

    return {
        "job_id": job_id,
        "video_path": video_path,
        "frames_dir": frames_dir,
        "labels_dir": labels_dir,
    }


def _resolve_spawn_probs(spawn_probs):
    """Resolve spawn probabilities with backward compatibility for legacy 'side'."""
    if not isinstance(spawn_probs, dict):
        return dict(DEFAULT_SPAWN_PROBS)

    probs = {g: max(float(spawn_probs.get(g, 0.0)), 0.0) for g in SURFACE_GROUPS}
    side_groups = ("side_left", "side_right", "side_toward")
    side_total = sum(probs[g] for g in side_groups)
    legacy_side = max(float(spawn_probs.get("side", 0.0)), 0.0)

    if side_total <= 1e-8 and legacy_side > 0.0:
        split = legacy_side / len(side_groups)
        for g in side_groups:
            probs[g] = split

    if sum(probs.values()) <= 1e-8:
        return dict(DEFAULT_SPAWN_PROBS)
    return probs


def _assemble_video(frames_dir, video_path):
    """Combine rendered PNG frames into an H.264 MP4 using ffmpeg (browser-compatible).

    Falls back to OpenCV (mp4v) if ffmpeg is not available.

    Args:
        frames_dir: Directory containing frame_XXXX.png files.
        video_path: Output MP4 path.
    """
    frame_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    if ffmpeg:
        pattern = os.path.join(frames_dir, "frame_%04d.png")
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(FPS),
            "-i", pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
        print(f"Video assembled (H.264): {video_path} ({len(frame_files)} frames)")
        return

    # Fallback: OpenCV with mp4v (may not play in all browsers)
    print("WARNING: ffmpeg not found, falling back to OpenCV mp4v codec.")
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))
    for fname in frame_files:
        writer.write(cv2.imread(os.path.join(frames_dir, fname)))
    writer.release()
    print(f"Video assembled (mp4v): {video_path} ({len(frame_files)} frames)")
