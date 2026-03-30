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
):
    """Compute a random-walk trajectory for one pest.

    Replaces animate_pest() — no bpy required.

    Args:
        num_frames:           Total number of frames to generate.
        plane_width:          Width of the plane in world units.
        plane_height:         Height of the plane in world units.
        speed:                Maximum step size per frame in world units.
        start_position:       Optional [wx, wy] in world units.
        placement_mask_path:  Optional path to the grayscale placement mask PNG.
        forward_axis:         Local axis facing the head: "X", "-X", "Y", "-Y".

    Returns:
        List of (wx, wy, angle_rad) tuples, one per frame (length == num_frames).
        angle_rad is the world heading (0 = facing +X, pi/2 = facing +Y).
    """
    margin = 0.15
    x_range = plane_width / 2.0 - margin
    y_range = plane_height / 2.0 - margin

    placement_mask = _load_mask(placement_mask_path)

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

    frames = []
    prev_x, prev_y = x, y

    # Frame 1: starting position, facing forward axis direction
    frames.append((x, y, axis_offset))

    for _ in range(num_frames - 1):
        moved = False
        for _ in range(12):
            dx = random.uniform(-speed, speed)
            dy = random.uniform(-speed, speed)
            next_x = _clamp(x + dx, -x_range, x_range)
            next_y = _clamp(y + dy, -y_range, y_range)
            if placement_mask is None or _accept_probabilistic_move(
                next_x, next_y, placement_mask, plane_width, plane_height
            ):
                x, y = next_x, next_y
                moved = True
                break

        dx_move = x - prev_x
        dy_move = y - prev_y
        if abs(dx_move) > 1e-6 or abs(dy_move) > 1e-6:
            heading = math.atan2(dy_move, dx_move) - axis_offset
            angle = heading + random.uniform(-0.08, 0.08)
        else:
            angle = frames[-1][2]  # no movement — keep prior heading

        frames.append((x, y, angle))
        prev_x, prev_y = x, y

    return frames


def _clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


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
    """Return placement probability for a world-space point (0..1)."""
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
    """Map world-plane coordinates to mask pixel indices."""
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
