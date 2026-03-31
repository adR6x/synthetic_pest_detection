"""Random-walk trajectory computation for pests — pure Python, no bpy."""

import math
import random

import numpy as np
from PIL import Image

_MASK_CACHE = {}


def compute_walk(
    num_frames,
    plane_width,
    plane_height,
    speed,
    start_position=None,
    placement_mask_path=None,
    forward_axis="X",
    max_step_world=None,
    depth_map=None,
    focal_length_px=None,
    base_speed_wps=None,
    max_speed_wps=None,
    fps=10,
    render_width=640,
    render_height=480,
    max_turn_deg=2.0,
    surface_group_masks=None,
    normals=None,
    surface_stickiness=0.97,
):
    """Compute a smooth steering trajectory for one pest.

    Uses Sagnik's steering model: holds a target angle for 40-120 frames,
    gradually turns toward it (max 2°/frame), smooths speed with a lerp,
    and occasionally pauses.

    When surface_group_masks and normals are provided, the movement mask is
    determined dynamically each frame: the pest's current pixel is looked up
    in the normals array to detect which surface group it's on (up/side/down),
    and the corresponding blended mask is used for the probabilistic move check.
    This means a pest that crosses from floor to wall gets wall-appropriate
    movement constraints automatically.

    Returns:
        List of (wx, wy, angle_rad, surface) tuples, one per frame.
        surface is "up", "side", or "down" — which surface the pest is on
        at that frame, used by the compositing layer for mask visualisation.
    """
    margin = 0.15
    x_range = plane_width / 2.0 - margin
    y_range = plane_height / 2.0 - margin

    # Static fallback mask (used when dynamic surface masks are unavailable).
    placement_mask = _load_mask(placement_mask_path)

    # Pre-compute one blended movement mask per surface group so we can switch
    # dynamically each frame without recomputing on the fly.
    _dynamic_masks = {}
    _norm_h = _norm_w = 0
    if surface_group_masks is not None and normals is not None:
        _norm_h, _norm_w = normals.shape[:2]
        ow = (1.0 - surface_stickiness) ** 2          # other-surface weight
        for surf in ("up", "side", "down"):
            same  = surface_group_masks[surf]
            other = sum(
                surface_group_masks[g] for g in ("up", "side", "down") if g != surf
            ) / 2.0
            m    = same + ow * other
            peak = float(m.max())
            _dynamic_masks[surf] = (m / peak if peak > 1e-6 else m).astype(np.float32)

    if start_position is not None and len(start_position) >= 2:
        x = _clamp(float(start_position[0]), -x_range, x_range)
        y = _clamp(float(start_position[1]), -y_range, y_range)
    else:
        x = random.uniform(-x_range, x_range)
        y = random.uniform(-y_range, y_range)

    if placement_mask is not None and _position_probability(
        x, y, placement_mask, plane_width, plane_height
    ) <= 1e-3:
        x, y = _sample_valid_world_position(placement_mask, plane_width, plane_height, margin)

    _AXIS_OFFSET = {"X": 0.0, "-X": math.pi, "Y": -math.pi / 2, "-Y": math.pi / 2}
    axis_offset = _AXIS_OFFSET.get(forward_axis, 0.0)
    if max_step_world is None:
        max_step_world = speed * 6.0
    max_step_world = max(float(max_step_world), float(speed))
    if base_speed_wps is None:
        base_speed_wps = float(speed) * float(max(fps, 1))
    base_speed_wps = float(base_speed_wps)

    # Sagnik steering state
    current_angle = random.uniform(0, 2 * math.pi)
    target_angle  = current_angle
    current_speed = float(speed)
    target_speed  = float(speed)
    pause_timer   = 0
    steer_timer   = 0
    burst_timer   = 0
    max_turn      = math.radians(float(max_turn_deg))

    frames = []

    for _ in range(num_frames):
        # Detect which surface group the pest is currently standing on.
        # Used both to select the active movement mask and to tag the frame
        # for the mask-preview visualisation in compositing.py.
        current_surface = "up"   # default
        if _dynamic_masks and _norm_h > 0 and _norm_w > 0:
            nx_px = int(_clamp(round((x / plane_width + 0.5) * (_norm_w - 1)), 0, _norm_w - 1))
            ny_px = int(_clamp(round((0.5 - y / plane_height) * (_norm_h - 1)), 0, _norm_h - 1))
            ny_val = float(normals[ny_px, nx_px, 1])
            if ny_val < -0.5:
                current_surface = "up"
            elif ny_val > 0.3:
                current_surface = "down"
            else:
                current_surface = "side"
        active_mask = _dynamic_masks.get(current_surface, placement_mask)

        # Record position, heading, and current surface for this frame.
        frames.append((x, y, current_angle - axis_offset, current_surface))

        # Depth-aware speed projection: convert physical speed (m/s-like) into
        # world-units/frame at the pest's current depth so distant pests move
        # fewer pixels and near pests can move slightly faster.
        dynamic_base_step = float(speed)
        dynamic_max_step  = float(max_step_world)
        if depth_map is not None and focal_length_px is not None and max_speed_wps is not None:
            proj_base = _projected_step_from_depth(
                x, y, depth_map, plane_width, plane_height,
                render_width, render_height, focal_length_px, base_speed_wps, fps)
            proj_max = _projected_step_from_depth(
                x, y, depth_map, plane_width, plane_height,
                render_width, render_height, focal_length_px, float(max_speed_wps), fps)
            # Clamp projection factors to keep motion stable and avoid depth outliers.
            if proj_base is not None and np.isfinite(proj_base):
                dynamic_base_step = float(np.clip(proj_base, speed * 0.5, speed * 2.0))
            if proj_max is not None and np.isfinite(proj_max):
                dynamic_max_step = float(np.clip(
                    proj_max, max_step_world * 0.5, max_step_world * 2.0))
        dynamic_max_step = max(dynamic_max_step, dynamic_base_step)

        # Pause: stay still, count down
        if pause_timer > 0:
            pause_timer -= 1
            continue

        # Pick a new target direction every 20-60 frames
        if steer_timer <= 0:
            # 5% chance of a high-speed dash in the current direction
            if random.random() < 0.05:
                burst_timer  = random.randint(10, 20)
                target_speed = min(dynamic_base_step * 6.0, dynamic_max_step)
                # Lock heading during dash so it looks like a purposeful sprint
                target_angle = current_angle
            else:
                target_angle = random.uniform(0, 2 * math.pi)
                target_speed = dynamic_base_step * random.uniform(0.5, 1.5)
            steer_timer = random.randint(40, 120)
        steer_timer -= 1

        # Count down burst; snap speed back to normal when it expires
        if burst_timer > 0:
            burst_timer  -= 1
            current_speed = min(target_speed, dynamic_max_step)   # skip lerp — instantly fast
            if burst_timer == 0:
                target_speed = dynamic_base_step   # ramp back to base after dash
        else:
            # Gradually steer toward target angle (max 8° per frame)
            diff = (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
            current_angle += max(-max_turn, min(max_turn, diff))

            # Smooth speed with lerp
            current_speed += (target_speed - current_speed) * 0.08

        # Global speed cap by pest type
        current_speed = min(current_speed, dynamic_max_step)

        # Occasional pause, then burst
        if burst_timer == 0 and random.random() < 0.008:
            pause_timer  = random.randint(12, 35)
            target_speed = min(dynamic_base_step * 1.4, dynamic_max_step)
            continue

        vx = math.cos(current_angle) * current_speed
        vy = math.sin(current_angle) * current_speed
        nx = x + vx
        ny = y + vy

        # Reflect off boundaries
        if nx < -x_range or nx > x_range:
            vx = -vx
            nx = _clamp(nx, -x_range, x_range)
            current_angle = math.atan2(vy, vx)
            target_angle  = current_angle
        if ny < -y_range or ny > y_range:
            vy = -vy
            ny = _clamp(ny, -y_range, y_range)
            current_angle = math.atan2(vy, vx)
            target_angle  = current_angle

        # Movement mask check — uses the dynamically selected mask for the
        # pest's current surface group (or the static fallback if unavailable).
        if active_mask is None or _accept_probabilistic_move(
            nx, ny, active_mask, plane_width, plane_height
        ):
            x, y = nx, ny
        else:
            escaped = False
            for _ in range(32):
                ta  = random.uniform(0, 2 * math.pi)
                tnx = _clamp(x + math.cos(ta) * dynamic_base_step, -x_range, x_range)
                tny = _clamp(y + math.sin(ta) * dynamic_base_step, -y_range, y_range)
                if active_mask is None or _accept_probabilistic_move(
                    tnx, tny, active_mask, plane_width, plane_height
                ):
                    x, y = tnx, tny
                    current_angle = ta
                    target_angle  = ta + random.uniform(-0.4, 0.4)
                    burst_timer   = 0
                    current_speed = dynamic_base_step
                    escaped = True
                    break
            if not escaped:
                current_angle = random.uniform(0, 2 * math.pi)
                target_angle  = current_angle
                burst_timer   = 0
                current_speed = dynamic_base_step

    return frames


def _clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def _projected_step_from_depth(
    wx,
    wy,
    depth_map,
    plane_width,
    plane_height,
    render_width,
    render_height,
    focal_length_px,
    speed_wps,
    fps,
):
    """Project physical speed into world-units/frame at the local depth.

    Perspective model:
      px_per_sec   = fx * speed_wps / depth_m
      px_per_frame = px_per_sec / fps
      world_step   = px_per_frame * plane_width / render_width
    """
    if depth_map is None or focal_length_px is None or fps <= 0 or render_width <= 0:
        return None

    h, w = depth_map.shape[:2]
    if h <= 0 or w <= 0:
        return None

    # World -> rendered pixel -> depth-map pixel.
    rx = (wx / plane_width + 0.5) * (render_width - 1)
    ry = (0.5 - wy / plane_height) * (render_height - 1)
    px = int(round((rx / max(render_width - 1, 1)) * (w - 1)))
    py = int(round((ry / max(render_height - 1, 1)) * (h - 1)))
    px = int(np.clip(px, 0, w - 1))
    py = int(np.clip(py, 0, h - 1))

    depth_m = float(depth_map[py, px])
    if not np.isfinite(depth_m) or depth_m <= 1e-3:
        return None
    depth_m = max(depth_m, 0.1)

    px_per_sec = float(focal_length_px) * float(speed_wps) / depth_m
    px_per_frame = px_per_sec / float(fps)
    return float(px_per_frame * plane_width / render_width)


def _load_mask(mask_path):
    """Load and cache a placement mask image."""
    if not mask_path:
        return None
    if mask_path not in _MASK_CACHE:
        if mask_path.lower().endswith(".png"):
            _MASK_CACHE[mask_path] = _load_mask_from_image(mask_path)
        else:
            print(f"WARNING: Unsupported placement mask format: {mask_path}")
            _MASK_CACHE[mask_path] = None
    return _MASK_CACHE[mask_path]


def _position_probability(x, y, mask, plane_width, plane_height):
    """Return placement probability for a world-space point (0..1).

    mask may be either a numpy (H, W) float32 array (from dynamic masks) or
    the legacy dict format produced by _load_mask_from_image().
    """
    if isinstance(mask, np.ndarray):
        h, w = mask.shape[:2]
        u = (x / plane_width) + 0.5
        v = 0.5 - (y / plane_height)
        px = int(_clamp(round(u * max(w - 1, 1)), 0, w - 1))
        py = int(_clamp(round(v * max(h - 1, 1)), 0, h - 1))
        return float(mask[py, px])
    py, px = _world_to_mask_indices(x, y, mask, plane_width, plane_height)
    return float(mask["values"][py * mask["width"] + px])


def _accept_probabilistic_move(x, y, mask, plane_width, plane_height):
    """Accept a move with probability given by the placement map at the target pixel."""
    p = _position_probability(x, y, mask, plane_width, plane_height)
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return random.random() < p


def _world_to_mask_indices(x, y, mask, plane_width, plane_height):
    """Map world-plane coordinates to mask pixel indices (dict-format mask only)."""
    h = mask["height"]
    w = mask["width"]
    u = (x / plane_width) + 0.5
    v = 0.5 - (y / plane_height)
    px = int(round(u * max(w - 1, 1)))
    py = int(round(v * max(h - 1, 1)))
    px = int(_clamp(px, 0, w - 1))
    py = int(_clamp(py, 0, h - 1))
    return py, px


def _sample_valid_world_position(mask, plane_width, plane_height, margin):
    """Sample a valid world position from the placement mask CDF."""
    coords = mask["coords"]
    cdf = mask["cdf"]
    if not coords:
        return 0.0, 0.0

    r = random.random()
    lo = 0
    hi = len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < r:
            lo = mid + 1
        else:
            hi = mid
    py, px = coords[lo]
    h = mask["height"]
    w = mask["width"]

    nx = px / max(w - 1, 1)
    ny = py / max(h - 1, 1)
    x = (nx - 0.5) * plane_width
    y = (0.5 - ny) * plane_height

    x_range = plane_width / 2.0 - margin
    y_range = plane_height / 2.0 - margin
    return _clamp(x, -x_range, x_range), _clamp(y, -y_range, y_range)


def _load_mask_from_image(mask_path):
    """Load a grayscale placement probability map via Pillow."""
    img = Image.open(mask_path).convert("L")  # 8-bit grayscale
    width, height = img.size
    raw = np.array(img, dtype=np.float32) / 255.0  # shape (H, W), values 0..1

    values = []
    coords = []
    weights = []
    for py in range(height):
        for px in range(width):
            p = float(raw[py, px])
            values.append(p)
            if p > 1e-3:
                coords.append((py, px))
                weights.append(p)

    if weights:
        total = sum(weights)
        cdf = []
        running = 0.0
        for w in weights:
            running += w / total
            cdf.append(running)
        cdf[-1] = 1.0
    else:
        cdf = []

    return {
        "width": int(width),
        "height": int(height),
        "values": values,
        "coords": coords,
        "cdf": cdf,
    }
