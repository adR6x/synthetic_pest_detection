"""Set up kitchen background, camera, and lighting (runs inside Blender)."""

import bpy
import mathutils


def clear_scene():
    """Remove all default objects."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_background_plane(image_path, plane_width, plane_height):
    """Create a plane with the kitchen image as its texture.

    Args:
        image_path: Path to the uploaded kitchen image.
        plane_width: Width of the plane in world units.
        plane_height: Height of the plane in world units.

    Returns:
        The plane object.
    """
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "kitchen_background"
    plane.scale = (plane_width, plane_height, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Load image and create material
    img = bpy.data.images.load(image_path)
    mat = bpy.data.materials.new(name="kitchen_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes["Principled BSDF"]
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    # Make it fully emissive so lighting doesn't darken the background
    bsdf.inputs["Roughness"].default_value = 1.0

    plane.data.materials.append(mat)
    return plane


def setup_camera(plane_width, plane_height):
    """Add an orthographic camera looking straight down at the plane.

    Args:
        plane_width: Width of the background plane.
        plane_height: Height of the background plane.

    Returns:
        The camera object.
    """
    bpy.ops.object.camera_add(location=(0, 0, 3))
    camera = bpy.context.active_object
    camera.name = "main_camera"
    camera.rotation_euler = (0, 0, 0)  # looking down -Z

    camera.data.type = "ORTHO"
    # ortho_scale = visible width in world units.
    # The plane spans plane_width total, so set ortho_scale to match exactly.
    # For 640x480 render (4:3) and plane_width/plane_height = 4:3, this fills
    # the frame with no empty space.
    camera.data.ortho_scale = plane_width

    bpy.context.scene.camera = camera
    return camera


def setup_lighting():
    """Add a sun light for even illumination."""
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))
    light = bpy.context.active_object
    light.name = "sun_light"
    light.data.energy = 3.0
    return light


def configure_render(width, height):
    """Set render engine and resolution.

    Args:
        width: Output image width in pixels.
        height: Output image height in pixels.
    """
    scene = bpy.context.scene

    # Dynamically pick the EEVEE engine name by querying Blender's own engine
    # registry. BLENDER_EEVEE was renamed to BLENDER_EEVEE_NEXT in 4.2 and
    # removed in 4.3+. Hardcoding a version number would break again on future
    # releases; checking the registry works on any Blender version.
    _available_engines = {
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    }
    if "BLENDER_EEVEE_NEXT" in _available_engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in _available_engines:
        scene.render.engine = "BLENDER_EEVEE"
    else:
        raise RuntimeError(f"No EEVEE engine found. Available: {_available_engines}")

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    # EEVEE settings for speed
    scene.eevee.taa_render_samples = 16
