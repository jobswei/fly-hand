__all___ = ["Renderer"]

import math
from pathlib import Path
from typing import Iterable, Literal

import bpy


def create_material_with_texture(
    name: str,
    texture,
    specular: float,
    reflectance: float,
    shininess: float,
):
    mat = bpy.data.materials.new(name=name)
    mat.specular_intensity = specular
    mat.metallic = reflectance
    mat.roughness = 1 - shininess
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.image = texture.image
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (texture.repeat_x,
                                             texture.repeat_y, 1)
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
    links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])

    return mat


def create_texture_from_image(
        name: str,
        file_path: Path,
        texrepeat: tuple[float, float] = (1, 1),
):
    img = bpy.data.images.load(str(file_path))
    tex = bpy.data.textures.new(name, type='IMAGE')
    tex.image = img
    tex.extension = 'REPEAT'
    tex.repeat_x = texrepeat[0]
    tex.repeat_y = texrepeat[1]
    return tex


def create_material_with_color(
    name: str,
    color: tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0),
    specular_intensity: float = 0.8,
):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = False
    mat.diffuse_color = color
    mat.specular_intensity = specular_intensity
    return mat


class Renderer:

    def __init__(
        self,
        camera_position: tuple[float, float, float],
        camera_fov: float,
        camera_target_position: tuple[float, float, float],
        camera_up_axis: Literal["x", "y", "z"] = "y",
        resolution: tuple[int, int] = (960, 640),
        light_position: tuple[float, float, float] = (0, 0, 10),
        light_rotation: tuple[float, float, float] = (0, 0, 0),
        light_energy: float = 500,
        plane_size: float = 20,
        plane_position: tuple[float, float, float] = (0, 0, 0),
        plane_rotation: tuple[float, float, float] = (0, 0, 0),
        plane_texture: Path | None = None,
        has_wall: bool = False,
        wall_texture: Path = Path("scene_assets/textures/gray_wall.png"),
    ) -> None:

        self._setup_environment(
            resolution,
            light_position,
            light_rotation,
            light_energy,
            plane_size,
            plane_position,
            plane_rotation,
            plane_texture,
            has_wall,
            wall_texture,
        )
        self._setup_camera(
            camera_position,
            camera_fov,
            camera_target_position,
            camera_up_axis,
        )
        self._objects = []

    def _setup_environment(
        self,
        resolution: tuple[int, int],
        light_position: tuple[float, float, float],
        light_rotation: tuple[float, float, float],
        light_energy: float,
        plane_size: float,
        plane_position: tuple[float, float, float],
        plane_rotation: tuple[float, float, float],
        plane_texture: Path | None,
        has_wall: bool,
        wall_texture: Path,
    ) -> None:
        # Clear current scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Get or create world
        world = bpy.data.worlds[0]

        # Ensure scene has a world setting
        bpy.context.scene.world = world

        # Set background color (using node system, applicable for Blender 2.8+)
        world.use_nodes = True
        background_node = world.node_tree.nodes.get('Background')
        # RGBA
        background_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

        # Create top environment light
        bpy.ops.object.light_add(
            type='POINT',
            align='WORLD',
            location=light_position,
        )
        light = bpy.context.object
        light.data.energy = light_energy
        light.data.color = (0.8, 0.8, 0.8)
        light.rotation_euler = tuple(map(math.radians, light_rotation))

        # plane settings
        bpy.ops.mesh.primitive_plane_add(
            size=plane_size,
            enter_editmode=False,
            align='WORLD',
            location=plane_position,
        )
        plane = bpy.context.object
        plane.name = "GroundPlane"
        plane.rotation_euler = tuple(map(math.radians, plane_rotation))
        if plane_texture is not None:
            texture = create_texture_from_image(
                "PlaneTexture",
                plane_texture,
                (30, 30),
            )
            material = create_material_with_texture(
                "GroundMaterial",
                texture,
                0.0,
                0.0,
                0.0,
            )
            plane.data.materials.append(material)

        if has_wall:
            texture = create_texture_from_image(
                "WallTexture",
                wall_texture,
                texrepeat=(30, 30),
            )
            material = create_material_with_texture(
                "WallMaterial",
                texture,
                0.8,
                0.0,
                0.3,
            )
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -1, 5))
            wall = bpy.context.active_object
            wall.name = "WallBack"
            wall.scale = (5, 0.1, 10)
            wall.data.materials.append(material)
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-1.5, 0, 5))
            wall = bpy.context.active_object
            wall.name = "WallLeft"
            wall.scale = (0.1, 5, 10)
            wall.data.materials.append(material)
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1.5, 0, 5))
            wall = bpy.context.active_object
            wall.name = "WallRight"
            wall.scale = (0.1, 5, 10)
            wall.data.materials.append(material)

        # scene settings
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Use Eevee engine
        scene.eevee.use_gtao = True  # Enable AO. (AO is off by default)
        scene.render.image_settings.file_format = 'PNG'  # Save format png
        scene.render.image_settings.color_mode = 'RGB'  # Save RGB channels,
        scene.render.image_settings.color_depth = '8'  # 8-bit color
        # Use sRGB color format for window display
        scene.display_settings.display_device = 'sRGB'
        # Use Filmic color transform for window display
        scene.view_settings.view_transform = 'Filmic'
        # For saving images
        scene.sequencer_colorspace_settings.name = 'Filmic sRGB'
        scene.render.film_transparent = False  # Non-transparent background
        scene.render.resolution_x, scene.render.resolution_y = resolution
        scene.eevee.taa_render_samples = 2
        scene.eevee.taa_samples = 0
        scene.eevee.gi_diffuse_bounces = 0

    def _setup_camera(
        self,
        position: tuple[float, float, float],
        fov: float,
        target_position: tuple[float, float, float],
        up_axis: Literal["x", "y", "z"],
    ) -> None:

        bpy.ops.object.camera_add(location=position)
        # Get the newly created camera object and set it as render camera
        bpy.context.scene.camera = bpy.context.object

        # Convert POV-Ray angle to Blender focal length
        bpy.data.cameras['Camera'].lens = 16 / math.tan(math.radians(fov / 2))
        # Create a target empty object
        bpy.ops.object.empty_add(
            type='PLAIN_AXES',
            location=tuple(target_position),
        )
        target = bpy.context.object
        target.name = "CameraTarget"

        # Add Track To constraint
        constraint = bpy.data.objects['Camera'].constraints.new('TRACK_TO')
        constraint.target = target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = f'UP_{up_axis.upper()}'

    def _load_obj(self, path: str):
        bpy.ops.wm.obj_import(filepath=path)
        for obj in bpy.context.selected_objects[1:]:
            bpy.data.objects.remove(obj, do_unlink=True)
        return bpy.context.selected_objects[0]

    def _setup_object(
        self,
        asset_path: str,
        size: list[float],
        color: tuple[float, float, float] = None,
    ):
        bpy.ops.object.select_all(action='DESELECT')

        obj = self._load_obj(asset_path)

        # must init the args
        obj.rotation_euler = (0, 0, 0)
        obj.location = (0, 0, 0)
        obj.scale = (1, 1, 1)

        # 为了和mujoco对齐，不再移动参考点到几何中心。这会造成position设置的不直观，
        # 因为用户不知道参考点在哪里
        # move reference point to geometric center
        # min_coordinate = [min(v[i] for v in obj.bound_box) for i in range(3)]
        # max_coordinate = [max(v[i] for v in obj.bound_box) for i in range(3)]
        # geometric_center = [(min_coordinate[i] + max_coordinate[i]) / 2
        #                     for i in range(3)]
        # obj.location = tuple(map(lambda x: -x, geometric_center))
        # bpy.ops.object.transform_apply(location=True)

        min_coordinate = [min(v[i] for v in obj.bound_box) for i in range(3)]
        max_coordinate = [max(v[i] for v in obj.bound_box) for i in range(3)]
        keep_ratio = isinstance(size, (int, float))
        size = [size] * 3 if keep_ratio else size
        bbox_size = [max_coordinate[i] - min_coordinate[i] for i in range(3)]
        base_size = [max(bbox_size)] * 3 if keep_ratio else bbox_size
        obj.scale = [i / j for i, j in zip(size, base_size)]
        bpy.ops.object.transform_apply(scale=True)

        # obj.rotation_euler = tuple(map(math.radians, (90,0,0)))
        # bpy.ops.object.transform_apply(rotation=True)

        if color is not None or not obj.data.materials:
            color = (1, 0, 0) if color is None else color
            obj.data.materials.clear()
            material = bpy.data.materials.new(name="MyMaterial")
            material.diffuse_color = (*color, 1.0)
            obj.data.materials.append(material)
        self._objects.append(obj)
        return obj

    def add_surface(
            self,
            position: tuple[float, float, float],
            color: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        bpy.ops.mesh.primitive_plane_add(
            size=0.1,
            enter_editmode=False,
            align='WORLD',
            location=position,
        )
        plane = bpy.context.object
        plane.name = "TargetPlane"
        plane.rotation_euler = (0, 0, 0)  # Convert angle to radians
        material = bpy.data.materials.new(name="TargetPlaneMaterial")
        material.diffuse_color = color + (1.0, )  # Set to white
        plane.data.materials.append(material)

    def render(
        self,
        object_positions: Iterable[tuple[float, float, float]],
        object_rotations: Iterable[tuple[float, float, float]],
        output_file: Path,
    ) -> None:

        for object_, position, rotation in zip(self._objects, object_positions,
                                               object_rotations):
            object_.rotation_euler = tuple(map(math.radians, rotation))
            object_.location = position

        bpy.ops.render.render()
        bpy.data.images["Render Result"].save_render(str(output_file))
