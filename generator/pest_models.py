"""Sprite loader for 2D pest compositing.

Looks for PNG sprite files under generator/sprites/{pest_type}/*.png.
Falls back to a procedurally drawn PIL ellipse if no sprites are found.
"""

import os
import random

from PIL import Image, ImageDraw

from generator.config import PEST_PARAMS

_SPRITE_BASE_SIZE = 128  # canonical sprite canvas width in pixels


def load_sprite(pest_type, sprites_dir):
    """Return an RGBA PIL Image for the given pest type.

    Resolution order:
    1. A random PNG from sprites_dir/{pest_type}/*.png  (if any exist).
    2. A procedurally drawn PIL ellipse matching the pest's color/shape.

    The returned image has mode "RGBA". The caller (compositing.py) resizes
    it to the target pixel dimensions before pasting.

    Args:
        pest_type:   One of "mouse", "rat", "cockroach".
        sprites_dir: Absolute path to the sprites root directory.

    Returns:
        RGBA PIL Image.
    """
    sprite_dir = os.path.join(sprites_dir, pest_type)
    sprite_files = []
    if os.path.isdir(sprite_dir):
        sprite_files = [
            f for f in os.listdir(sprite_dir)
            if f.lower().endswith(".png")
        ]

    if sprite_files:
        chosen = random.choice(sprite_files)
        return Image.open(os.path.join(sprite_dir, chosen)).convert("RGBA")

    return _draw_procedural_sprite(pest_type)


def _draw_procedural_sprite(pest_type):
    """Draw a top-down ellipse sprite for the given pest type.

    The canvas is _SPRITE_BASE_SIZE wide. The body aspect ratio comes from
    body_scale[1] / body_scale[0] (width / length). A head circle is drawn
    near the right edge for mouse and rat (head_scale is not None).

    The sprite is drawn with the head pointing right (+X), matching
    PEST_FORWARD_AXIS = "X".

    Returns an RGBA image with a transparent background.
    """
    params = PEST_PARAMS[pest_type]
    color_f = params["color"]                          # (R, G, B, A) floats 0..1
    color_i = tuple(int(c * 255) for c in color_f)    # (R, G, B, A) ints

    body_scale = params["body_scale"]                  # (length, width, height)
    aspect = body_scale[1] / max(body_scale[0], 1e-8) # width / length

    canvas_w = _SPRITE_BASE_SIZE
    canvas_h = max(8, int(canvas_w * aspect))

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = 4
    draw.ellipse(
        [margin, margin, canvas_w - margin, canvas_h - margin],
        fill=color_i,
    )

    # Head circle for mouse / rat (near the right/front edge)
    if params.get("head_scale") is not None:
        head_scale = params["head_scale"]
        head_r_fraction = head_scale[0] / max(body_scale[0], 1e-8)
        head_px_r = max(2, int(head_r_fraction * canvas_w * 0.5))
        head_cx = canvas_w - margin - head_px_r
        head_cy = canvas_h // 2
        draw.ellipse(
            [
                head_cx - head_px_r,
                head_cy - head_px_r,
                head_cx + head_px_r,
                head_cy + head_px_r,
            ],
            fill=color_i,
        )

    return img
