"""2D sprite compositing pipeline — replaces Blender rendering.

Composites RGBA pest sprites onto the background kitchen image frame-by-frame,
using the random-walk trajectory from pest_animation.compute_walk() and sprites
from pest_models.load_sprite(). Writes frame PNGs and a COCO annotations.json.
"""

import math
import os

from PIL import Image

from generator.pest_animation import compute_walk
from generator.pest_models import load_sprite
from generator.labeler import save_coco_dataset
from generator.config import RENDER_WIDTH, RENDER_HEIGHT, PLANE_WIDTH, PLANE_HEIGHT


_CATEGORY_MAP = {"mouse": 1, "rat": 2, "cockroach": 3}
_CATEGORIES = [
    {"id": 1, "name": "mouse",     "supercategory": "pest"},
    {"id": 2, "name": "rat",       "supercategory": "pest"},
    {"id": 3, "name": "cockroach", "supercategory": "pest"},
]


def composite_frames(
    image_path,
    pest_configs,
    frames_dir,
    labels_dir,
    num_frames,
    sprites_dir,
    render_width=RENDER_WIDTH,
    render_height=RENDER_HEIGHT,
    plane_width=PLANE_WIDTH,
    plane_height=PLANE_HEIGHT,
):
    """Render all frames by compositing pest sprites onto the background image.

    For each frame i in [1..num_frames]:
      1. Copy the background image (resized to render_width x render_height).
      2. For each pest: look up (wx, wy, angle_rad) from its walk trajectory,
         resize sprite, rotate, and alpha-composite onto the frame.
      3. Save as frames_dir/frame_{i:04d}.png.

    Also writes a COCO annotations.json to labels_dir.

    Args:
        image_path:    Path to the kitchen background image.
        pest_configs:  List of pest config dicts from pipeline.py. Each must have:
                           "type"                – pest type string
                           "start_position"      – [wx, wy] world units
                           "placement_mask_path" – path to placement mask PNG
                           "params": {
                               "speed"           – float
                               "forward_axis"    – str
                               "blender_scale"   – float (world-unit body length)
                           }
        frames_dir:    Directory to write frame_XXXX.png files into.
        labels_dir:    Directory to write annotations.json into.
        num_frames:    Number of frames to render.
        sprites_dir:   Root directory for per-pest-type sprite PNG files.
        render_width:  Output frame width in pixels.
        render_height: Output frame height in pixels.
        plane_width:   World-space width of the scene.
        plane_height:  World-space height of the scene.
    """
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Load and resize background once
    bg = Image.open(image_path).convert("RGBA")
    bg = bg.resize((render_width, render_height), Image.LANCZOS)

    # Pre-compute walk trajectories and sprites for all pests
    pest_data = []
    for pest_cfg in pest_configs:
        pest_type     = pest_cfg["type"]
        params        = pest_cfg["params"]
        speed         = float(params.get("speed", 0.08))
        fwd_axis      = params.get("forward_axis", "X")
        blender_scale = float(params.get("blender_scale", 0.12))

        walk = compute_walk(
            num_frames=num_frames,
            plane_width=plane_width,
            plane_height=plane_height,
            speed=speed,
            start_position=pest_cfg.get("start_position"),
            placement_mask_path=pest_cfg.get("placement_mask_path"),
            forward_axis=fwd_axis,
        )

        sprite   = load_sprite(pest_type, sprites_dir)
        pixel_w, pixel_h = _compute_pixel_size(
            sprite, blender_scale, plane_width, render_width
        )

        pest_data.append((pest_type, sprite, walk, pixel_w, pixel_h))

    # Render each frame
    coco_images      = []
    coco_annotations = []
    ann_id = 1

    for frame_idx in range(num_frames):
        frame_num = frame_idx + 1   # 1-based for filenames and COCO

        frame_img = bg.copy()

        for pest_type, sprite, walk, pixel_w, pixel_h in pest_data:
            wx, wy, angle_rad = walk[frame_idx]

            paste_cx, paste_cy = _world_to_pixel(
                wx, wy, render_width, render_height, plane_width, plane_height
            )

            resized = sprite.resize((pixel_w, pixel_h), Image.LANCZOS)

            # PIL.rotate is counter-clockwise; atan2 is also CCW — signs match.
            angle_deg = math.degrees(angle_rad)
            rotated   = resized.rotate(angle_deg, expand=True)

            paste_x = paste_cx - rotated.width  // 2
            paste_y = paste_cy - rotated.height // 2

            # Alpha-composite (rotated is RGBA; expand=True gives transparent corners)
            frame_img.paste(rotated, (paste_x, paste_y), rotated)

            # COCO bbox from rotated sprite bounds, clamped to image
            bbox_x  = max(0, paste_x)
            bbox_y  = max(0, paste_y)
            bbox_x2 = min(render_width,  paste_x + rotated.width)
            bbox_y2 = min(render_height, paste_y + rotated.height)
            bbox_w  = bbox_x2 - bbox_x
            bbox_h  = bbox_y2 - bbox_y

            if bbox_w > 0 and bbox_h > 0:
                coco_annotations.append({
                    "id":          ann_id,
                    "image_id":    frame_num,
                    "category_id": _CATEGORY_MAP.get(pest_type, 0),
                    "bbox":        [bbox_x, bbox_y, bbox_w, bbox_h],
                    "area":        bbox_w * bbox_h,
                    "iscrowd":     0,
                })
                ann_id += 1

        frame_path = os.path.join(frames_dir, f"frame_{frame_num:04d}.png")
        frame_img.convert("RGB").save(frame_path)

        coco_images.append({
            "id":        frame_num,
            "file_name": f"frame_{frame_num:04d}.png",
            "width":     render_width,
            "height":    render_height,
        })

    save_coco_dataset(
        coco_images,
        coco_annotations,
        _CATEGORIES,
        os.path.join(labels_dir, "annotations.json"),
    )


def _world_to_pixel(wx, wy, render_width, render_height, plane_width, plane_height):
    """Convert world-space coordinates to pixel coordinates.

    World origin (0, 0) = centre of image.
    World (+plane_width/2, +plane_height/2) = top-right corner.
    Pixel (0, 0) = top-left corner.
    """
    px = int((wx / plane_width  + 0.5) * render_width)
    py = int((0.5 - wy / plane_height) * render_height)
    px = max(0, min(render_width  - 1, px))
    py = max(0, min(render_height - 1, py))
    return px, py


def _compute_pixel_size(sprite, blender_scale, plane_width, render_width):
    """Compute target pixel width and height for a sprite.

    blender_scale is the pest body length in world units.
    pixel_w = blender_scale * render_width / plane_width
    pixel_h preserves the sprite's aspect ratio.
    Both dimensions are clamped to at least 4 pixels.
    """
    pixel_w = max(4, int(blender_scale * render_width / plane_width))
    sw, sh  = sprite.size
    pixel_h = max(4, int(pixel_w * sh / max(sw, 1)))
    return pixel_w, pixel_h
