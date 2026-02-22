"""Create 3D pest geometry from Blender primitives (runs inside Blender)."""

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
