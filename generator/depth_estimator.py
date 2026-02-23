"""Depth-aware pest placement using Depth Anything V2 (runs in system Python)."""

import threading

import cv2
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


# --------------------------------------------------------------------------- #
#  Inference strategy                                                          #
# --------------------------------------------------------------------------- #

def compute_inference_strategy():
    """Return 'parallel' or 'sequential' for the three feed-forward models.

    Rules
    -----
    - No CUDA (CPU only)   → 'parallel'   (no VRAM contention)
    - Multiple GPUs        → 'parallel'   (models can use separate devices)
    - Single GPU           → 'sequential' (avoid VRAM contention / context switching)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "parallel"
        return "parallel" if torch.cuda.device_count() > 1 else "sequential"
    except ImportError:
        return "parallel"


# --------------------------------------------------------------------------- #
#  Gravity / camera-up estimation via vertical vanishing point                #
# --------------------------------------------------------------------------- #

def estimate_gravity(image_path):
    """Estimate the gravity (camera-up) direction from a single indoor image.

    Uses a fully classical vanishing-point pipeline — no additional ML model:
      1. Detect line segments with OpenCV's LSD (falls back to Canny+HoughP).
      2. Keep near-vertical segments (<=30 deg from vertical, >=20 px long).
      3. RANSAC to find the dominant vertical vanishing point (VP).
      4. Convert the VP pixel position to a unit up-vector in camera space.

    Camera-space convention (matching Omnidata surface normals):
        X = right,  Y = up,  Z = out-of-screen toward viewer.

    Falls back to [0, 1, 0] (level-camera prior) when no reliable VP is found.

    Args:
        image_path: Path to the kitchen image.

    Returns:
        dict with keys:
            "gravity_cam"  - np.ndarray (3,) unit vector pointing up in camera space.
            "vp"           - (float, float) VP pixel coords, or None.
            "vert_lines"   - list[(x1,y1,x2,y2)] near-vertical segments used.
            "confidence"   - float inlier fraction in [0, 1].
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return _gravity_fallback()

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Line detection
    all_lines = _detect_line_segments(gray)

    # 2. Keep near-vertical segments
    vert_lines = []
    for x1, y1, x2, y2 in all_lines:
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 20:
            continue
        # Angle from vertical: 0 deg = perfectly vertical
        angle_from_vert = np.degrees(abs(np.arctan2(abs(x2 - x1), abs(y2 - y1))))
        if angle_from_vert < 30:
            vert_lines.append((float(x1), float(y1), float(x2), float(y2)))

    if len(vert_lines) < 3:
        return {**_gravity_fallback(), "vert_lines": vert_lines}

    # 3. RANSAC vanishing point
    img_diag = float(np.hypot(w, h))
    vp, n_inliers = _ransac_vanishing_point(vert_lines, img_diag)
    confidence = n_inliers / len(vert_lines) if vert_lines else 0.0

    if vp is None or n_inliers < 3 or confidence < 0.25:
        return {**_gravity_fallback(), "vert_lines": vert_lines,
                "confidence": float(confidence)}

    # 4. VP -> camera-space up vector
    # Camera convention: X=right, Y=up, Z=out-of-screen
    # Image convention: x=right, y=DOWN  ->  flip Y
    cx, cy = w / 2.0, h / 2.0
    f = float(max(w, h))          # rough focal-length estimate (~60 deg FOV)
    gx = (vp[0] - cx) / f
    gy = -(vp[1] - cy) / f        # image-Y down -> camera-Y up
    gz = 1.0
    gravity_cam = np.array([gx, gy, gz], dtype=np.float32)
    mag = np.linalg.norm(gravity_cam)
    if mag < 1e-8:
        return {**_gravity_fallback(), "vert_lines": vert_lines,
                "confidence": float(confidence)}
    gravity_cam /= mag

    return {
        "gravity_cam": gravity_cam,
        "vp": vp,
        "vert_lines": vert_lines,
        "confidence": float(confidence),
    }


def save_gravity_preview(image_path, gravity_result, output_path):
    """Save a visualization of the estimated gravity / camera orientation.

    Draws on the image:
      - Detected near-vertical segments extended to full image height (dark green)
        so their convergence toward the vanishing point is visible even when the
        VP is off-screen.
      - The actual detected segments highlighted on top (bright green).
      - Estimated vanishing point when within / near image bounds (red dot).
      - Horizon line through the image centre, perpendicular to the gravity
        direction and accounting for camera pitch and roll (cyan double arrow).
        A level, non-rolled camera produces a perfectly horizontal line; pitch
        tilts it forward/backward; roll tilts it left/right.

    Below the image a dark panel shows numerical values:
      - gravity_cam vector (X right, Y up, Z out-of-screen)
      - pitch  = camera tilt forward/backward (0 deg = level)
      - roll   = camera roll left/right       (0 deg = upright)
      - detection confidence and number of vertical lines used

    Args:
        image_path:     Path to the original kitchen image.
        gravity_result: Dict returned by estimate_gravity().
        output_path:    Destination path for the preview (JPEG or PNG).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()

    gravity_cam = gravity_result["gravity_cam"]
    vp = gravity_result.get("vp")
    vert_lines = gravity_result.get("vert_lines", [])

    # 1. Extend each vertical segment to full image height (shows convergence)
    for x1, y1, x2, y2 in vert_lines:
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) > 1e-6:
            t_top = (0 - y1) / dy
            t_bot = (h - 1 - y1) / dy
            ext_x_top = int(round(x1 + t_top * dx))
            ext_x_bot = int(round(x1 + t_bot * dx))
            p_top = (int(np.clip(ext_x_top, -w, 2 * w)), 0)
            p_bot = (int(np.clip(ext_x_bot, -w, 2 * w)), h - 1)
            cv2.line(vis, p_top, p_bot, (0, 130, 0), 1, cv2.LINE_AA)

    # 2. Detected segments on top (bright green)
    for x1, y1, x2, y2 in vert_lines:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                 (0, 230, 0), 2, cv2.LINE_AA)

    # 3. Vanishing point if near/within the image
    if vp is not None:
        vpx, vpy = vp
        if -h <= vpy <= 2 * h and -w <= vpx <= 2 * w:
            dx = int(np.clip(vpx, 0, w - 1))
            dy = int(np.clip(vpy, 0, h - 1))
            cv2.circle(vis, (dx, dy), 10, (0, 0, 255), -1)
            cv2.circle(vis, (dx, dy), 12, (255, 255, 255), 2)

    # 4. Horizon line — perpendicular to gravity up, accounts for pitch + roll.
    #    Up direction in image coords: (gravity_cam[0], -gravity_cam[1])
    #    Rotate 90° CW to get horizon direction: (gravity_cam[1], gravity_cam[0])
    hz_dx = float(gravity_cam[1])
    hz_dy = float(gravity_cam[0])
    mag_hz = np.hypot(hz_dx, hz_dy)
    if mag_hz > 1e-6:
        hz_dx /= mag_hz
        hz_dy /= mag_hz
        cx, cy = w / 2.0, h / 2.0
        # Find t where the horizon line exits the image boundary
        t_candidates = []
        for t in ([(-cx) / hz_dx, (w - 1 - cx) / hz_dx] if abs(hz_dx) > 1e-6 else []):
            t_candidates.append(t)
        for t in ([(- cy) / hz_dy, (h - 1 - cy) / hz_dy] if abs(hz_dy) > 1e-6 else []):
            t_candidates.append(t)
        t_pos = min((t for t in t_candidates if t > 0), default=float(w))
        t_neg = max((t for t in t_candidates if t < 0), default=-float(w))
        p1 = (int(np.clip(cx + t_neg * hz_dx, 0, w - 1)),
              int(np.clip(cy + t_neg * hz_dy, 0, h - 1)))
        p2 = (int(np.clip(cx + t_pos * hz_dx, 0, w - 1)),
              int(np.clip(cy + t_pos * hz_dy, 0, h - 1)))
        icx, icy = int(round(cx)), int(round(cy))
        cv2.arrowedLine(vis, (icx, icy), p1, (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.04)
        cv2.arrowedLine(vis, (icx, icy), p2, (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.04)

    # 5. Text panel below the image
    panel_h = 52
    panel = np.full((panel_h, w, 3), 25, dtype=np.uint8)
    gx, gy, gz = float(gravity_cam[0]), float(gravity_cam[1]), float(gravity_cam[2])
    # pitch: forward/backward tilt (arctan2 of Z toward Y; 0 = level)
    pitch_deg = float(np.degrees(np.arctan2(gz, gy)))
    # roll: left/right tilt (arctan2 of X toward Y; 0 = upright)
    roll_deg = float(np.degrees(np.arctan2(gx, gy)))
    conf = gravity_result.get("confidence", 0.0)
    vp_str = f"({vp[0]:.0f}, {vp[1]:.0f})" if vp else "off-frame"
    cv2.putText(panel,
                f"gravity_cam = [{gx:+.3f}, {gy:+.3f}, {gz:+.3f}]",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(panel,
                f"pitch={pitch_deg:+.1f}deg  roll={roll_deg:+.1f}deg  "
                f"conf={conf:.2f}  lines={len(vert_lines)}  VP={vp_str}",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1, cv2.LINE_AA)

    combined = np.vstack([vis, panel])
    rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(output_path)


# ---- gravity helpers ------------------------------------------------------- #

def _gravity_fallback():
    """Level-camera prior: gravity points straight up in camera space."""
    return {
        "gravity_cam": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "vp": None,
        "vert_lines": [],
        "confidence": 0.0,
    }


def _detect_line_segments(gray):
    """Detect line segments. Tries LSD first, falls back to Canny+HoughLinesP."""
    try:
        lsd = cv2.createLineSegmentDetector(0)
        raw, _, _, _ = lsd.detect(gray)
        if raw is not None and len(raw) > 0:
            return raw.reshape(-1, 4).tolist()
    except (cv2.error, AttributeError):
        pass
    # Fallback: Canny edges + probabilistic Hough
    edges = cv2.Canny(gray, 50, 150)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                          minLineLength=30, maxLineGap=10)
    if raw is not None:
        return raw.reshape(-1, 4).tolist()
    return []


def _ransac_vanishing_point(lines, img_diag, n_iter=300):
    """RANSAC to find the dominant vanishing point for a set of line segments.

    Inlier criterion: perpendicular distance from the candidate VP to each
    extended line is less than 2% of the image diagonal.

    Returns:
        (vp, n_inliers) -- vp is (float, float) pixel coords, or (None, 0).
    """
    if len(lines) < 2:
        return None, 0

    arr = np.array(lines, dtype=np.float64)
    n = len(arr)
    thresh = img_diag * 0.02
    best_vp, best_n = None, 0

    for _ in range(n_iter):
        i, j = np.random.choice(n, 2, replace=False)
        vp = _intersect_lines(arr[i], arr[j])
        if vp is None:
            continue
        vpx, vpy = vp
        if not (np.isfinite(vpx) and np.isfinite(vpy)):
            continue
        if abs(vpx) > 1e5 or abs(vpy) > 1e5:
            continue
        dists = np.fromiter(
            (_pt_line_dist(vpx, vpy, *seg) for seg in arr),
            dtype=np.float64, count=n,
        )
        n_in = int((dists < thresh).sum())
        if n_in > best_n:
            best_n = n_in
            best_vp = vp

    return best_vp, best_n


def _intersect_lines(l1, l2):
    """Intersection of two infinite lines via homogeneous coordinates.

    Line through (x1,y1)-(x2,y2):  a*x + b*y + c = 0
        a = y1-y2,  b = x2-x1,  c = x1*y2 - x2*y1
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    a1, b1, c1 = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
    a2, b2, c2 = y3 - y4, x4 - x3, x3 * y4 - x4 * y3
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det
    return float(x), float(y)


def _pt_line_dist(px, py, x1, y1, x2, y2):
    """Perpendicular distance from (px,py) to the infinite line through (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    denom = np.hypot(dx, dy)
    if denom < 1e-10:
        return float(np.hypot(px - x1, py - y1))
    return abs(dy * (px - x1) - dx * (py - y1)) / denom
