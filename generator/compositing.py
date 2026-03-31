"""2D sprite compositing pipeline — replaces Blender rendering.

Composites RGBA pest sprites onto the background kitchen image frame-by-frame,
using the random-walk trajectory from pest_animation.compute_walk() and sprites
from pest_models.load_sprite(). Writes frame PNGs and a COCO annotations.json.
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw

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
    depth_map=None,
    fps=10,
    surface_group_masks=None,
    normals=None,
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
                           "placement_mask_path" – path to movement mask PNG
                           "params": {
                               "speed"             – float (wu/frame)
                               "max_speed_wps"     – float (wu/s, for depth-aware cap)
                               "focal_length_px"   – float
                               "forward_axis"      – str
                               "blender_scale"     – float (world-unit body length)
                           }
        frames_dir:    Directory to write frame_XXXX.png files into.
        labels_dir:    Directory to write annotations.json into.
        num_frames:    Number of frames to render.
        sprites_dir:   Root directory for per-pest-type sprite PNG files.
        render_width:  Output frame width in pixels.
        render_height: Output frame height in pixels.
        plane_width:   World-space width of the scene.
        plane_height:  World-space height of the scene.
        depth_map:     Optional (H, W) float32 depth array in metres.
        fps:           Frames per second (used for depth-aware speed cap).
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
        speed            = float(params.get("speed", 0.08))
        fwd_axis         = params.get("forward_axis", "X")
        blender_scale    = float(params.get("blender_scale", 0.12))
        max_step_world   = params.get("max_step_world")
        if max_step_world is not None:
            max_step_world = float(max_step_world)
        base_speed_wps   = params.get("base_speed_wps")
        if base_speed_wps is not None:
            base_speed_wps = float(base_speed_wps)
        max_speed_wps    = params.get("max_speed_wps")
        if max_speed_wps is not None:
            max_speed_wps = float(max_speed_wps)
        focal_length_px  = params.get("focal_length_px")
        if focal_length_px is not None:
            focal_length_px = float(focal_length_px)
        max_turn_deg     = params.get("max_turn_deg", 2.0)
        stickiness       = float(params.get("surface_stickiness", 0.97))

        walk = compute_walk(
            num_frames=num_frames,
            plane_width=plane_width,
            plane_height=plane_height,
            speed=speed,
            start_position=pest_cfg.get("start_position"),
            placement_mask_path=pest_cfg.get("placement_mask_path"),
            forward_axis=fwd_axis,
            max_step_world=max_step_world,
            depth_map=depth_map,
            focal_length_px=focal_length_px,
            base_speed_wps=base_speed_wps,
            max_speed_wps=max_speed_wps,
            fps=fps,
            render_width=render_width,
            render_height=render_height,
            max_turn_deg=max_turn_deg,
            surface_group_masks=surface_group_masks,
            normals=normals,
            surface_stickiness=stickiness,
        )

        sprite   = load_sprite(pest_type, sprites_dir)
        pixel_w, pixel_h = _compute_pixel_size(
            sprite, blender_scale, plane_width, render_width
        )

        pest_data.append((pest_type, sprite, walk, pixel_w, pixel_h))

    # Pre-load movement masks as numpy arrays for per-frame overlay rendering.
    _DOT_COLORS = {"mouse": (0, 220, 0), "rat": (255, 120, 0), "cockroach": (255, 30, 30)}
    mask_arrays = []
    surface_groups = tuple(surface_group_masks.keys()) if surface_group_masks is not None else ()
    for pest_cfg in pest_configs:
        mpath = pest_cfg.get("placement_mask_path")
        if mpath and os.path.exists(mpath):
            m = Image.open(mpath).convert("L").resize(
                (render_width, render_height), Image.LANCZOS)
            mask_arrays.append(np.array(m, dtype=np.float32) / 255.0)
        else:
            mask_arrays.append(np.ones((render_height, render_width), dtype=np.float32))

    # Pre-compute per-pest dynamic mask arrays (one per surface group) resized to
    # render dimensions. Used for per-frame "pest vision" mask preview.
    _viz_dynamic_masks = []
    for pest_cfg in pest_configs:
        stickiness = float(pest_cfg["params"].get("surface_stickiness", 0.97))
        if surface_group_masks is not None and surface_groups:
            ow = (1.0 - stickiness) ** 2
            dyn_viz = {}
            for surf in surface_groups:
                same  = surface_group_masks[surf]
                other_groups = [g for g in surface_groups if g != surf]
                if other_groups:
                    other = sum(surface_group_masks[g] for g in other_groups) / float(len(other_groups))
                else:
                    other = np.zeros_like(same, dtype=np.float32)
                m     = same + ow * other
                peak  = float(m.max())
                m_norm = (m / peak if peak > 1e-6 else m).astype(np.float32)
                m_pil  = Image.fromarray((m_norm * 255).astype(np.uint8))
                m_pil  = m_pil.resize((render_width, render_height), Image.LANCZOS)
                dyn_viz[surf] = np.array(m_pil, dtype=np.float32) / 255.0
            _viz_dynamic_masks.append(dyn_viz)
        else:
            _viz_dynamic_masks.append(None)

    # Render each frame
    coco_images      = []
    coco_annotations = []
    ann_id = 1

    for frame_idx in range(num_frames):
        frame_num = frame_idx + 1   # 1-based for filenames and COCO

        frame_img = bg.copy()

        for pest_type, sprite, walk, pixel_w, pixel_h in pest_data:
            wx, wy, angle_rad, _surface = walk[frame_idx]

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

        # --- Per-pest dynamic "pest vision" mask preview ---
        # Each pest gets its own preview: the movement probability mask for the
        # surface it currently stands on, with a dot showing its position.
        # The mask switches automatically when the pest crosses to a new surface.
        for pest_idx, ((ptype, _sprite, pwalk, pw, ph), dyn_masks, mask_arr) in enumerate(
            zip(pest_data, _viz_dynamic_masks, mask_arrays)
        ):
            wx, wy, _, surface = pwalk[frame_idx]
            active_viz = dyn_masks.get(surface, mask_arr) if dyn_masks is not None else mask_arr
            gray_u8 = (np.clip(active_viz, 0.0, 1.0) * 255).astype(np.uint8)
            pest_mask_img = Image.fromarray(
                np.stack([gray_u8, gray_u8, gray_u8], axis=-1), mode="RGB"
            )
            draw = ImageDraw.Draw(pest_mask_img)
            pcx, pcy = _world_to_pixel(
                wx, wy, render_width, render_height, plane_width, plane_height
            )
            r     = max(pw, ph) // 2 + 6
            color = _DOT_COLORS.get(ptype, (255, 255, 0))
            draw.ellipse([pcx - r, pcy - r, pcx + r, pcy + r], outline=color, width=3)
            draw.ellipse([pcx - 5, pcy - 5, pcx + 5, pcy + 5], fill=color)
            pest_mask_img.save(
                os.path.join(frames_dir, f"mask_preview_pest{pest_idx}_{ptype}_{frame_num:04d}.png")
            )

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
