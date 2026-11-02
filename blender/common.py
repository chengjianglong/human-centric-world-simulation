"""Shared bpy helpers: argument parsing, materials, primitive meshes."""
import math
import os
import sys

import bpy
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ASSETS = os.path.join(ROOT, "assets")


def script_args():
    """Arguments after `--` on the blender command line."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def get_collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(col)
    return col


# ---------------------------------------------------------------- materials
def material(name, color, roughness=0.6, metallic=0.0, transmission=0.0, emission=None, alpha=1.0):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if transmission:
        bsdf.inputs["Transmission Weight"].default_value = transmission
        bsdf.inputs["IOR"].default_value = 1.45
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission[3] if len(emission) > 3 else 5.0
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
    mat.diffuse_color = (*color[:3], 1.0)
    return mat


def _noise_color(mat, color, scale, detail, var, stretch=(1, 1, 1)):
    """Multiply a material's base colour by a procedural noise/wave pattern."""
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = stretch
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = detail
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    lo = [max(0.0, c * (1 - var)) for c in color]
    hi = [min(1.0, c * (1 + var)) for c in color]
    ramp.color_ramp.elements[0].color = (*lo, 1)
    ramp.color_ramp.elements[1].color = (*hi, 1)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return nt, mp, ramp


def wood_material(name, color, roughness=0.45, plank_width=0.18):
    """Wood floor: plank stripes along x plus grain noise."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = material(name, color, roughness)
    nt, mp, ramp = _noise_color(mat, color, scale=3.0, detail=8.0, var=0.25, stretch=(1.0, 12.0, 1.0))
    bsdf = nt.nodes["Principled BSDF"]
    # plank seams: wave bands across y (period = plank_width) darken thin lines
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Y"
    wave.wave_profile = "SAW"
    wave.inputs["Scale"].default_value = 2 * math.pi / (10.0 * plank_width)
    seam = nt.nodes.new("ShaderNodeMath")
    seam.operation = "LESS_THAN"  # 1 on the seam, 0 elsewhere
    seam.inputs[1].default_value = 0.04
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Color2"].default_value = (0.55, 0.55, 0.55, 1)
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], seam.inputs[0])
    nt.links.new(seam.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs["Color1"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def fabric_material(name, color, roughness=0.95):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = material(name, color, roughness)
    _noise_color(mat, color, scale=120.0, detail=2.0, var=0.12)
    mat.node_tree.nodes["Principled BSDF"].inputs["Sheen Weight"].default_value = 0.4
    return mat


# ---------------------------------------------------------------- meshes
def _assign(obj, mat):
    if mat is not None:
        obj.data.materials.append(mat)


def box(name, center, size, mat=None, parent=None, rot_z=0.0, bevel=0.0, collection=None):
    """Axis-aligned box (in parent frame) centred at `center` with full extents `size`."""
    sx, sy, sz = (s / 2 for s in size)
    verts = [(x, y, z) for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    obj = link(bpy.data.objects.new(name, me), collection)
    obj.location = center
    obj.rotation_euler = (0, 0, rot_z)
    if bevel > 0:
        m = obj.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = 3
        m.limit_method = "ANGLE"
    _assign(obj, mat)
    if parent is not None:
        obj.parent = parent
    return obj


def cylinder(name, center, radius, depth, mat=None, parent=None, verts=32, collection=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=center)
    obj = bpy.context.active_object
    obj.name = name
    _move_to(obj, collection)
    bpy.ops.object.shade_smooth()
    _assign(obj, mat)
    if parent is not None:
        obj.parent = parent
    return obj


def ellipsoid(name, center, radii, mat=None, parent=None, collection=None, segments=24):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=segments // 2, radius=1.0, location=center)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = radii
    _move_to(obj, collection)
    bpy.ops.object.shade_smooth()
    _assign(obj, mat)
    if parent is not None:
        obj.parent = parent
    return obj


def _move_to(obj, collection):
    if collection is None:
        return
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)


def empty(name, location=(0, 0, 0), rot_z=0.0, collection=None):
    obj = link(bpy.data.objects.new(name, None), collection)
    obj.location = location
    obj.rotation_euler = (0, 0, rot_z)
    return obj


def look_at(obj, target):
    from mathutils import Vector
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def set_world_matrix_keep(obj, mw):
    obj.matrix_world = Matrix(mw)
