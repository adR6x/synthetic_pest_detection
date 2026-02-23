"""Create 3D pest geometry from Blender primitives or imported models (runs inside Blender)."""

import os

import bpy
import mathutils


def create_pest(pest_type, params, index):
    """Create a pest mesh from UV spheres scaled into ellipsoids.

    Args:
        pest_type: One of 'mouse', 'rat', 'cockroach'.
        params: Dict with body_scale, head_scale, head_offset, color.
        index: Unique index for naming.

    Returns:
        The body mesh object (parent of the pest).
    """
    name = f"{pest_type}_{index}"

    # Body: UV sphere scaled into an ellipsoid
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16, ring_count=8, radius=1.0, location=(0, 0, 0)
    )
    body = bpy.context.active_object
    body.name = f"{name}_body"
    body.scale = mathutils.Vector(params["body_scale"])

    # Apply material
    mat = bpy.data.materials.new(name=f"{name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = params["color"]
    bsdf.inputs["Roughness"].default_value = 0.8
    body.data.materials.append(mat)

    # Head (mouse/rat only)
    if params["head_scale"] is not None:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=12, ring_count=6, radius=1.0, location=(0, 0, 0)
        )
        head = bpy.context.active_object
        head.name = f"{name}_head"
        head.scale = mathutils.Vector(params["head_scale"])
        head.location = mathutils.Vector(params["head_offset"])
        head.data.materials.append(mat)
        head.parent = body

    # Apply scale so bounding box calculations work correctly
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Lift body so it sits on the plane surface
    body.location.z = params["body_scale"][2]

    return body


def load_pest(pest_type, params, index):
    """Load a pest object: import from file if model_path is set, else use procedural geometry.

    When a model file is provided, it is imported and uniformly scaled so its longest
    horizontal bounding-box dimension equals params['blender_scale'].  The procedural
    fallback rescales the UV-sphere ellipsoid proportionally to the same target size.

    Args:
        pest_type: One of 'mouse', 'rat', 'cockroach'.
        params: Dict from pipeline config.  Relevant keys:
                  blender_scale  – target body length in Blender world units.
                  model_path     – path to a .obj or .glb file, or None.
                  body_scale     – procedural proportions (used as ratio baseline).
        index: Unique integer for object naming.

    Returns:
        The root Blender object (body mesh or imported root).
    """
    model_path   = params.get("model_path")
    blender_scale = float(params.get("blender_scale", params["body_scale"][0]))

    if model_path and os.path.isfile(model_path):
        obj = _import_model(pest_type, model_path, index, blender_scale)
        if obj is not None:
            return obj
        print(f"WARNING: model import failed for {model_path}, falling back to procedural.")

    return _create_scaled_pest(pest_type, params, index, blender_scale)


# ---- helpers ----------------------------------------------------------------

def _create_scaled_pest(pest_type, params, index, blender_scale):
    """Call create_pest() with all spatial params scaled to match blender_scale."""
    original_body_x = float(params["body_scale"][0])
    if original_body_x < 1e-8:
        return create_pest(pest_type, params, index)

    factor = blender_scale / original_body_x
    scaled = dict(params)
    scaled["body_scale"] = [v * factor for v in params["body_scale"]]
    if params.get("head_scale"):
        scaled["head_scale"] = [v * factor for v in params["head_scale"]]
    if params.get("head_offset"):
        scaled["head_offset"] = [v * factor for v in params["head_offset"]]
    return create_pest(pest_type, scaled, index)


def _import_model(pest_type, model_path, index, blender_scale):
    """Import a .obj or .glb model, normalise it, and return the root object.

    The imported mesh is:
      1. Joined into a single object if multiple meshes were imported.
      2. Origin reset to geometry centre.
      3. Uniformly scaled so its longest horizontal dimension equals blender_scale.
      4. Lifted so its bottom sits on Z = 0.

    Returns the root Blender object, or None on failure.
    """
    ext = os.path.splitext(model_path)[1].lower()
    bpy.ops.object.select_all(action="DESELECT")

    try:
        if ext == ".obj":
            bpy.ops.import_scene.obj(filepath=model_path)
        elif ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=model_path)
        else:
            print(f"Unsupported model format: {ext}")
            return None
    except Exception as exc:
        print(f"Model import error: {exc}")
        return None

    imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not imported:
        print("No mesh objects found after import.")
        return None

    # Join all imported meshes into one object
    bpy.context.view_layer.objects.active = imported[0]
    for o in imported:
        o.select_set(True)
    if len(imported) > 1:
        bpy.ops.object.join()

    root = bpy.context.active_object
    root.name = f"{pest_type}_{index}_body"

    # Reset transforms so we start from a clean slate
    root.location = (0.0, 0.0, 0.0)
    root.rotation_euler = (0.0, 0.0, 0.0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Centre origin on geometry
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    # Scale to target: longest horizontal (X or Y) dimension = blender_scale
    dims = root.dimensions
    ref_dim = max(dims.x, dims.y, 1e-6)
    uniform_scale = blender_scale / ref_dim
    root.scale = (uniform_scale, uniform_scale, uniform_scale)
    bpy.ops.object.transform_apply(scale=True)

    # Move origin back to geometry centre after scale apply
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    # Lift so the model's bottom face sits on the plane (Z = 0)
    root.location = (0.0, 0.0, root.dimensions.z / 2.0)

    return root
