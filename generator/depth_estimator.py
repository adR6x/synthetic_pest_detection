"""Depth-aware pest placement using Depth Anything V2 (runs in system Python)."""

import threading

import numpy as np
from PIL import Image

from generator.config import DEPTH_MODEL

_DEPTH_PIPELINE = None
_DEPTH_PIPELINE_LOCK = threading.Lock()


def _get_depth_pipeline():
    """Lazily construct and cache the HF depth pipeline (process-local)."""
    global _DEPTH_PIPELINE
    if _DEPTH_PIPELINE is None:
        with _DEPTH_PIPELINE_LOCK:
            if _DEPTH_PIPELINE is None:
                from transformers import pipeline

                _DEPTH_PIPELINE = pipeline(
                    "depth-estimation", model=DEPTH_MODEL, device=-1
                )
    return _DEPTH_PIPELINE


def preload_depth_model(run_warmup_inference=False):
    """Preload the depth model into memory to reduce first-request latency."""
    pipe = _get_depth_pipeline()
    if run_warmup_inference:
        dummy = Image.new("RGB", (64, 64), color=(127, 127, 127))
        try:
            pipe(dummy)
        except Exception:
            # Warmup inference is best-effort; model load is the main win.
            pass


def estimate_depth(image_path):
    """Estimate metric depth from a kitchen image using Depth Anything V2.

    Args:
        image_path: Path to the kitchen image.

    Returns:
        depth_map: np.ndarray of shape (H, W) with depth in meters.
    """
    pipe = _get_depth_pipeline()
    image = Image.open(image_path).convert("RGB")
    result = pipe(image)

    # Prefer raw numeric depth if the pipeline exposes it.
    predicted_depth = result.get("predicted_depth")
    if predicted_depth is not None:
        if hasattr(predicted_depth, "detach"):
            predicted_depth = predicted_depth.detach().cpu().numpy()
        depth_map = np.array(predicted_depth, dtype=np.float32).squeeze()
    else:
        depth_map = np.array(result["depth"], dtype=np.float32)

    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    return depth_map


def compute_surface_normals(depth_map):
    """Compute surface normals from a depth map using np.gradient.

    Args:
        depth_map: np.ndarray (H, W) depth in meters.

    Returns:
        normals: np.ndarray (H, W, 3) with unit normals. The Z component
                 (normals[:,:,2]) indicates surface orientation:
                 ~1.0 = flat horizontal, ~0.0 = vertical wall, <0 = ceiling.
    """
    dy, dx = np.gradient(depth_map)
    # Normal = (-dz/dx, -dz/dy, 1), then normalize
    normals = np.stack([-dx, -dy, np.ones_like(depth_map)], axis=-1)
    magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)
    magnitude = np.clip(magnitude, 1e-8, None)
    normals /= magnitude
    return normals


def save_depth_preview(depth_map, output_path):
    """Save a normalized grayscale preview image for visual inspection."""
    depth = np.array(depth_map, dtype=np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        preview = np.zeros_like(depth, dtype=np.uint8)
    else:
        valid = depth[finite]
        d_min = float(np.percentile(valid, 2))
        d_max = float(np.percentile(valid, 98))
        if d_max <= d_min:
            d_min = float(valid.min())
            d_max = float(valid.max()) if float(valid.max()) > float(valid.min()) else d_min + 1.0
        depth = np.clip(depth, d_min, d_max)
        norm = (depth - d_min) / max(d_max - d_min, 1e-8)
        # Dark = far, bright = near for easier visual parsing
        preview = (255.0 * (1.0 - norm)).astype(np.uint8)

    Image.fromarray(preview, mode="L").save(output_path)


def save_surface_preview(normals, output_path):
    """Save an RGB preview of surface normals for visual inspection."""
    normals = np.array(normals, dtype=np.float32)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    # Map [-1, 1] -> [0, 255]
    rgb = np.clip((normals + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(output_path)


def save_mask_preview(mask, output_path):
    """Save a binary placement mask as an image (white=valid, black=invalid)."""
    mask_img = (np.array(mask, dtype=bool).astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(output_path)


def save_probability_preview(normals, output_path):
    """Save a grayscale probability preview derived from nz (surface flatness)."""
    nz = np.array(normals, dtype=np.float32)[:, :, 2]
    nz = np.nan_to_num(nz, nan=0.0, posinf=0.0, neginf=0.0)
    # Map nz from [-1, 1] to [0, 1]; higher means flatter/up-facing.
    prob = np.clip((nz + 1.0) * 0.5, 0.0, 1.0)
    img = (prob * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(output_path)


def build_placement_mask(normals, nz_threshold):
    """Build a binary mask of pixels where a pest can be placed.

    Args:
        normals: np.ndarray (H, W, 3) surface normals.
        nz_threshold: Minimum Z-component of the normal. Lower values
                      allow steeper surfaces (walls, overhangs).

    Returns:
        mask: np.ndarray (H, W) boolean mask.
    """
    nz = normals[:, :, 2]
    return nz > nz_threshold


def sample_pest_positions(mask, n, plane_width, plane_height):
    """Sample n pest starting positions from valid mask pixels.

    Converts pixel coordinates to world coordinates matching the Blender plane
    (centered at origin, spanning -plane_width/2..+plane_width/2 on X,
    -plane_height/2..+plane_height/2 on Y).

    Args:
        mask: (H, W) boolean placement mask.
        n: Number of positions to sample.
        plane_width: Width of the kitchen plane in world units.
        plane_height: Height of the kitchen plane in world units.

    Returns:
        List of (x, y) tuples in world coordinates.
    """
    valid_ys, valid_xs = np.where(mask)
    if len(valid_ys) == 0:
        # Fallback: if no valid pixels, return center positions
        return [(0.0, 0.0)] * n

    h, w = mask.shape
    margin_frac = 0.05  # stay slightly away from edges

    positions = []
    for _ in range(n):
        idx = np.random.randint(len(valid_ys))
        px, py = valid_xs[idx], valid_ys[idx]
        # Pixel to normalized [0, 1]
        nx = px / max(w - 1, 1)
        ny = py / max(h - 1, 1)
        # Normalized to world coords (centered at origin)
        wx = (nx - 0.5) * plane_width
        wy = (0.5 - ny) * plane_height  # flip Y: image top = +Y in world
        # Clamp to stay within margin
        x_lim = plane_width / 2.0 * (1.0 - margin_frac)
        y_lim = plane_height / 2.0 * (1.0 - margin_frac)
        wx = float(np.clip(wx, -x_lim, x_lim))
        wy = float(np.clip(wy, -y_lim, y_lim))
        positions.append((wx, wy))

    return positions
