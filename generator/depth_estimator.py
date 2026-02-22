"""Depth-aware pest placement using Depth Anything V2 (runs in system Python)."""

import threading

import numpy as np
from PIL import Image

from generator.config import DEPTH_MODEL

_DEPTH_PIPELINE = None
_DEPTH_PIPELINE_LOCK = threading.Lock()
_SURFACE_NORMAL_MODEL = None
_SURFACE_NORMAL_MODEL_LOCK = threading.Lock()
_SURFACE_NORMAL_DEVICE = None
_SURFACE_NORMAL_TRANSFORM = None


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


def _get_surface_normal_model():
    """Lazily construct and cache Omnidata surface-normal model (process-local)."""
    global _SURFACE_NORMAL_MODEL, _SURFACE_NORMAL_DEVICE, _SURFACE_NORMAL_TRANSFORM
    if _SURFACE_NORMAL_MODEL is None:
        with _SURFACE_NORMAL_MODEL_LOCK:
            if _SURFACE_NORMAL_MODEL is None:
                import torch
                import PIL
                from torchvision import transforms

                _SURFACE_NORMAL_DEVICE = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                model = torch.hub.load(
                    "alexsax/omnidata_models", "surface_normal_dpt_hybrid_384"
                )
                model.to(_SURFACE_NORMAL_DEVICE)
                model.eval()
                _SURFACE_NORMAL_MODEL = model
                _SURFACE_NORMAL_TRANSFORM = transforms.Compose(
                    [
                        transforms.Resize(384, interpolation=PIL.Image.BILINEAR),
                        transforms.CenterCrop(384),
                        transforms.ToTensor(),
                    ]
                )
    return _SURFACE_NORMAL_MODEL, _SURFACE_NORMAL_DEVICE, _SURFACE_NORMAL_TRANSFORM


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


def estimate_surface_normals_pretrained(image_path):
    """Estimate surface normals from RGB using Omnidata DPT-Hybrid.

    Returns:
        np.ndarray of shape (H, W, 3) with approximate unit normals in [-1, 1].
    """
    import torch
    import torch.nn.functional as F

    model, device, transform = _get_surface_normal_model()

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    with torch.no_grad():
        img_tensor = transform(image)[:3].unsqueeze(0).to(device)
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3, 1)
        output = model(img_tensor).clamp(min=0.0, max=1.0)
        output = F.interpolate(
            output, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        normals = output[0].permute(1, 2, 0).cpu().numpy()

    # Omnidata outputs normal channels encoded in [0, 1]; map to [-1, 1].
    normals = normals * 2.0 - 1.0
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)
    magnitude = np.clip(magnitude, 1e-8, None)
    normals /= magnitude
    return normals


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


def save_surface_preview_pretrained(image_path, output_path):
    """Save surface-normal preview predicted by a pretrained normal model."""
    normals = estimate_surface_normals_pretrained(image_path)
    save_surface_preview(normals, output_path)


def save_mask_preview(mask, output_path):
    """Save a placement map image (binary or probabilistic) to grayscale PNG."""
    arr = np.array(mask)
    if arr.dtype == np.bool_:
        img = (arr.astype(np.uint8) * 255)
    else:
        img = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(output_path)


def compute_depth_placement_score(depth_map):
    """Compute a normalized near-surface score from depth only (near=1, far=0)."""
    depth = np.array(depth_map, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros_like(depth, dtype=np.float32)

    valid = depth[finite]
    d_min = float(np.percentile(valid, 2))
    d_max = float(np.percentile(valid, 98))
    if d_max <= d_min:
        d_min = float(valid.min())
        d_max = float(valid.max()) if float(valid.max()) > float(valid.min()) else d_min + 1.0

    depth = np.clip(depth, d_min, d_max)
    norm = (depth - d_min) / max(d_max - d_min, 1e-8)
    # Higher score = closer to camera
    score = 1.0 - norm
    return score.astype(np.float32)


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


def build_depth_placement_mask(depth_map, threshold):
    """Build a binary placement mask from depth only using a near-score threshold."""
    score = compute_depth_placement_score(depth_map)
    return score > float(threshold)


def build_surface_probability_map(normals, nz_threshold, softness=1.0):
    """Build a per-pest probabilistic placement map from predicted normals.

    Formula:
        slope_prob = sigmoid((nz - nz_threshold) / softness)
        coherence = mean_neighbor_cosine_similarity(normals) mapped to [0, 1]
        p = slope_prob * coherence

    This uses all three normal channels via the coherence term, while keeping
    the pest-specific slope threshold behavior through nz.
    """
    normals = np.array(normals, dtype=np.float32)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure unit normals before cosine-similarity coherence.
    mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    mag = np.clip(mag, 1e-8, None)
    normals = normals / mag

    nz = normals[:, :, 2]
    nz = np.nan_to_num(nz, nan=0.0, posinf=0.0, neginf=0.0)
    softness = float(max(softness, 1e-4))
    logits = (nz - float(nz_threshold)) / softness
    logits = np.clip(logits, -30.0, 30.0)
    slope_prob = 1.0 / (1.0 + np.exp(-logits))

    coherence = _normal_coherence_map(normals)
    prob = slope_prob * coherence
    return prob.astype(np.float32)


def _normal_coherence_map(normals):
    """Return local normal coherence in [0,1] using all 3 channels."""
    # Cosine similarity with 4-neighbors (normals are unit length).
    c = np.zeros(normals.shape[:2], dtype=np.float32)
    n = 0

    # Up/down
    sim_ud = np.sum(normals[1:, :, :] * normals[:-1, :, :], axis=-1)
    c[1:, :] += sim_ud
    c[:-1, :] += sim_ud
    n += 2

    # Left/right
    sim_lr = np.sum(normals[:, 1:, :] * normals[:, :-1, :], axis=-1)
    c[:, 1:] += sim_lr
    c[:, :-1] += sim_lr
    n += 2

    c /= float(n)
    # Cosine similarity is in [-1,1]; map to [0,1].
    c = np.clip((c + 1.0) * 0.5, 0.0, 1.0)
    return c.astype(np.float32)


def sample_pest_positions_from_probability(probability_map, n, plane_width, plane_height):
    """Sample n starting positions from a probability map via weighted pixel sampling."""
    prob = np.array(probability_map, dtype=np.float32)
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)

    flat = prob.reshape(-1)
    total = float(flat.sum())
    if total <= 1e-8:
        return sample_pest_positions(prob > 0.5, n, plane_width, plane_height)

    weights = flat / total
    h, w = prob.shape
    margin_frac = 0.05
    idxs = np.random.choice(flat.size, size=n, replace=True, p=weights)

    positions = []
    for flat_idx in idxs:
        py = int(flat_idx // w)
        px = int(flat_idx % w)
        nx = px / max(w - 1, 1)
        ny = py / max(h - 1, 1)
        wx = (nx - 0.5) * plane_width
        wy = (0.5 - ny) * plane_height
        x_lim = plane_width / 2.0 * (1.0 - margin_frac)
        y_lim = plane_height / 2.0 * (1.0 - margin_frac)
        wx = float(np.clip(wx, -x_lim, x_lim))
        wy = float(np.clip(wy, -y_lim, y_lim))
        positions.append((wx, wy))

    return positions


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
