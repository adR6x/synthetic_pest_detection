"""Keyframed random-walk animation for pests (runs inside Blender)."""

import random

import bpy

_MASK_CACHE = {}


def animate_pest(
    pest_obj,
    num_frames,
    plane_width,
    plane_height,
    speed,
    start_position=None,
    placement_mask_path=None,
):
    """Animate a pest with a random-walk scurrying motion.

    Args:
        pest_obj: The pest body mesh object.
        num_frames: Total number of frames.
        plane_width: Width of the kitchen plane (world units).
        plane_height: Height of the kitchen plane (world units).
        speed: Step size per frame in world units.
        start_position: Optional [x, y] starting coordinates in world units.
        placement_mask_path: Optional path to a boolean .npy placement mask.
    """
    # Random starting position within the plane bounds.
    # The plane goes from -plane_width/2 to +plane_width/2, so use half-dimensions.
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

    z = pest_obj.location.z  # keep height constant

    # Frame 1 starts at the sampled position.
    pest_obj.location = (x, y, z)
    pest_obj.keyframe_insert(data_path="location", frame=1)
    pest_obj.keyframe_insert(data_path="rotation_euler", frame=1)

    for frame in range(2, num_frames + 1):
        # Random walk step, with rejection if it leaves valid placement regions.
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

        if not moved and placement_mask is not None:
            # Stay in place if no valid move was found.
            pass

        pest_obj.location = (x, y, z)
        pest_obj.keyframe_insert(data_path="location", frame=frame)

        # Random slight rotation for natural movement
        angle_z = random.uniform(-0.3, 0.3)
        pest_obj.rotation_euler.z += angle_z
        pest_obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def _clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def _load_mask(mask_path):
    """Load and cache a placement mask image into a Python-friendly structure."""
    if not mask_path:
        return None
    if mask_path not in _MASK_CACHE:
        if mask_path.lower().endswith(".png"):
            _MASK_CACHE[mask_path] = _load_mask_from_image(mask_path)
        else:
            print(f"WARNING: Unsupported placement mask format in Blender: {mask_path}")
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
    """Map world-plane coordinates back to mask pixel indices."""
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
    """Fallback sampler inside Blender if the provided start position is invalid."""
    coords = mask["coords"]
    cdf = mask["cdf"]
    if not coords:
        return 0.0, 0.0

    r = random.random()
    idx = 0
    lo = 0
    hi = len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < r:
            lo = mid + 1
        else:
            hi = mid
    idx = lo
    py, px = coords[idx]
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
    """Load a grayscale/rgba placement probability map via Blender."""
    img = bpy.data.images.load(mask_path, check_existing=True)
    width, height = img.size[:2]
    pixels = list(img.pixels[:])  # flat RGBA floats in [0,1]

    values = []
    coords = []
    weights = []
    for py in range(height):
        row_offset = py * width
        for px in range(width):
            idx = (row_offset + px) * 4
            p = float(_clamp(pixels[idx], 0.0, 1.0))  # grayscale saved in red channel
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
