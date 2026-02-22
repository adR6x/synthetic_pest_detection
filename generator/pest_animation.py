"""Keyframed random-walk animation for pests (runs inside Blender)."""

import random
import bpy
import mathutils


def animate_pest(pest_obj, num_frames, plane_width, plane_height, speed):
    """Animate a pest with a random-walk scurrying motion.

    Args:
        pest_obj: The pest body mesh object.
        num_frames: Total number of frames.
        plane_width: Width of the kitchen plane (world units).
        plane_height: Height of the kitchen plane (world units).
        speed: Step size per frame in world units.
    """
    # Random starting position within the plane bounds.
    # The plane goes from -plane_width/2 to +plane_width/2, so use half-dimensions.
    margin = 0.15
    x_range = plane_width / 2.0 - margin
    y_range = plane_height / 2.0 - margin

    x = random.uniform(-x_range, x_range)
    y = random.uniform(-y_range, y_range)
    z = pest_obj.location.z  # keep height constant

    for frame in range(1, num_frames + 1):
        # Random walk step
        dx = random.uniform(-speed, speed)
        dy = random.uniform(-speed, speed)
        x = _clamp(x + dx, -x_range, x_range)
        y = _clamp(y + dy, -y_range, y_range)

        pest_obj.location = (x, y, z)
        pest_obj.keyframe_insert(data_path="location", frame=frame)

        # Random slight rotation for natural movement
        angle_z = random.uniform(-0.3, 0.3)
        pest_obj.rotation_euler.z += angle_z
        pest_obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def _clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))
