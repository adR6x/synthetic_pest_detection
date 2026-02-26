"""Project 3D mesh bounding boxes to 2D and output COCO-format labels (runs inside Blender)."""

import json
import os

import bpy
from bpy_extras.object_utils import world_to_camera_view


def compute_bbox_2d(obj, scene, camera):
    """Project an object's bounding box corners to 2D pixel coordinates.

    Args:
        obj: The Blender mesh object.
        scene: The current scene.
        camera: The active camera object.

    Returns:
        Dict with x_min, y_min, x_max, y_max in pixel coordinates,
        or None if the object is off-screen.
    """
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y

    # Collect all bounding box corners from this object and its children
    corners_2d = []
    objects = [obj] + list(obj.children)

    for o in objects:
        if o.type != "MESH":
            continue
        mesh = o.data
        mat = o.matrix_world

        for vert in mesh.vertices:
            world_pos = mat @ vert.co
            co_2d = world_to_camera_view(scene, camera, world_pos)
            # co_2d: (x, y, depth) where x,y are 0..1 normalized
            px = co_2d.x * res_x
            py = (1.0 - co_2d.y) * res_y  # flip Y for image coordinates
            corners_2d.append((px, py))

    if not corners_2d:
        return None

    xs = [c[0] for c in corners_2d]
    ys = [c[1] for c in corners_2d]

    return {
        "x_min": max(0, int(min(xs))),
        "y_min": max(0, int(min(ys))),
        "x_max": min(res_x, int(max(xs))),
        "y_max": min(res_y, int(max(ys))),
    }


def collect_frame_annotations(pests, scene, camera, frame_num, category_map):
    """Return COCO annotation dicts for all visible pests in the current frame.

    Args:
        pests: List of (pest_type, pest_object) tuples.
        scene: The current Blender scene.
        camera: The active camera.
        frame_num: Current frame number (used as image_id).
        category_map: Dict mapping pest_type string to COCO category_id int.

    Returns:
        List of annotation dicts (without 'id' — caller assigns it).
    """
    annotations = []
    for pest_type, pest_obj in pests:
        bbox = compute_bbox_2d(pest_obj, scene, camera)
        if bbox is None:
            continue
        x = bbox["x_min"]
        y = bbox["y_min"]
        w = bbox["x_max"] - bbox["x_min"]
        h = bbox["y_max"] - bbox["y_min"]
        if w <= 0 or h <= 0:
            continue
        annotations.append({
            "image_id": frame_num,
            "category_id": category_map.get(pest_type, 0),
            "bbox": [x, y, w, h],  # COCO format: [x, y, width, height]
            "area": w * h,
            "iscrowd": 0,
        })
    return annotations


def save_coco_dataset(images, annotations, categories, output_path):
    """Write a COCO-format annotations.json.

    Args:
        images: List of COCO image dicts.
        annotations: List of COCO annotation dicts.
        categories: List of COCO category dicts.
        output_path: Full path to write annotations.json.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
