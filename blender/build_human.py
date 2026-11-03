"""Procedural rigged digital human (armature + body meshes bone-parented).

    tools/blender/blender -b -P blender/build_human.py -- [--out assets]

Writes assets/human.blend and assets/human.glb with a looping walk-cycle action.
`make_human()` / `pose_human()` are reused by render_rollout.py.
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from worldsim import gait  # noqa: E402
from worldsim.skeleton import BONE_DRIVE, BONES  # noqa: E402

OUTFITS = [  # shirt, trousers, skin, shoes, hair
    ((0.55, 0.12, 0.10), (0.12, 0.14, 0.22), (0.80, 0.60, 0.48), (0.08, 0.06, 0.05), (0.10, 0.07, 0.05)),
    ((0.15, 0.35, 0.55), (0.30, 0.28, 0.25), (0.55, 0.38, 0.27), (0.9, 0.9, 0.9), (0.04, 0.03, 0.03)),
    ((0.85, 0.75, 0.30), (0.20, 0.22, 0.20), (0.92, 0.75, 0.62), (0.30, 0.18, 0.10), (0.55, 0.35, 0.15)),
    ((0.25, 0.50, 0.30), (0.10, 0.10, 0.12), (0.40, 0.27, 0.18), (0.15, 0.15, 0.15), (0.05, 0.04, 0.04)),
    ((0.60, 0.40, 0.65), (0.45, 0.40, 0.35), (0.85, 0.68, 0.55), (0.5, 0.1, 0.1), (0.25, 0.15, 0.08)),
    ((0.90, 0.90, 0.88), (0.15, 0.25, 0.40), (0.70, 0.52, 0.40), (0.1, 0.1, 0.1), (0.15, 0.10, 0.06)),
]

# body part -> (bone, material slot, radii scale (x, y), extra) ; ellipsoid spans the bone
PARTS = {
    "pelvis": ("pelvis", "trousers", (0.155, 0.105, 2.2)),
    "torso": ("spine", "shirt", (0.17, 0.11, 1.2)),
    "neck": ("neck", "skin", (0.05, 0.05)),
    "head": ("head", "skin", (0.095, 0.11)),
    "thigh.L": ("thigh.L", "trousers", (0.075, 0.08)), "thigh.R": ("thigh.R", "trousers", (0.075, 0.08)),
    "shin.L": ("shin.L", "trousers", (0.058, 0.06)), "shin.R": ("shin.R", "trousers", (0.058, 0.06)),
    "upper_arm.L": ("upper_arm.L", "shirt", (0.048, 0.048)), "upper_arm.R": ("upper_arm.R", "shirt", (0.048, 0.048)),
    "forearm.L": ("forearm.L", "skin", (0.038, 0.038)), "forearm.R": ("forearm.R", "skin", (0.038, 0.038)),
}


def _bone_world(arm_obj, name):
    b = arm_obj.data.bones[name]
    return arm_obj.matrix_world @ b.head_local, arm_obj.matrix_world @ b.tail_local


def _parent_to_bone(obj, arm_obj, bone):
    mw = obj.matrix_world.copy()
    obj.parent = arm_obj
    obj.parent_type = "BONE"
    obj.parent_bone = bone
    bpy.context.view_layer.update()
    obj.matrix_world = mw


def make_human(name="human", outfit=0, collection=None, height=1.75):
    """Create an armature + body meshes. Returns the armature object (root at feet, facing -Y)."""
    col = collection or C.get_collection(name)
    shirt, trousers, skin, shoes, hair = OUTFITS[outfit % len(OUTFITS)]
    mats = dict(shirt=C.fabric_material(f"{name}_shirt", shirt), trousers=C.fabric_material(f"{name}_trousers", trousers),
                skin=C.material(f"{name}_skin", skin, 0.45), shoes=C.material(f"{name}_shoes", shoes, 0.4),
                hair=C.material(f"{name}_hair", hair, 0.6), eye=C.material("eye", (0.03, 0.03, 0.03), 0.1))
    mats["skin"].node_tree.nodes["Principled BSDF"].inputs["Subsurface Weight"].default_value = 0.15

    arm = bpy.data.armatures.new(f"{name}_rig")
    ao = C.link(bpy.data.objects.new(name, arm), col)
    bpy.context.view_layer.objects.active = ao
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ao.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    s = height / 1.75
    for bname, parent, head, tail, _ in BONES:
        eb = arm.edit_bones.new(bname)
        eb.head = Vector(head) * s
        eb.tail = Vector(tail) * s
        eb.roll = 0.0
        if parent:
            eb.parent = arm.edit_bones[parent]
            eb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()

    def add(obj, bone):
        C._move_to(obj, col)
        _parent_to_bone(obj, ao, bone)

    for part, (bone, slot, radii) in PARTS.items():
        rx, ry, zmul = (*radii, 1.12)[:3]
        h, t = _bone_world(ao, bone)
        L = (t - h).length
        o = C.ellipsoid(f"{name}_{part}", (h + t) / 2, (rx * s, ry * s, L / 2 * zmul), mats[slot])
        d = (t - h).normalized()
        o.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        add(o, bone)
    # head details: hair cap, nose, eyes (face is -Y)
    h, t = _bone_world(ao, "head")
    c = (h + t) / 2
    add(C.ellipsoid(f"{name}_hair", c + Vector((0, 0.012, 0.035)) * s, (0.1 * s, 0.112 * s, 0.1 * s), mats["hair"]), "head")
    add(C.ellipsoid(f"{name}_nose", c + Vector((0, -0.105, -0.01)) * s, (0.014 * s, 0.02 * s, 0.022 * s), mats["skin"], segments=10), "head")
    for sx in (-1, 1):
        add(C.ellipsoid(f"{name}_eye", c + Vector((sx * 0.035, -0.09, 0.015)) * s, (0.012 * s,) * 3, mats["eye"], segments=10), "head")
    # hands, feet, shoulders
    for side in ("L", "R"):
        h, t = _bone_world(ao, f"forearm.{side}")
        add(C.ellipsoid(f"{name}_hand.{side}", t + Vector((0, 0, -0.04)) * s, (0.03 * s, 0.045 * s, 0.07 * s), mats["skin"], segments=12), f"forearm.{side}")
        h, t = _bone_world(ao, f"foot.{side}")
        add(C.ellipsoid(f"{name}_shoe.{side}", (h + t) / 2 + Vector((0, 0, -0.005)) * s, (0.05 * s, 0.13 * s, 0.045 * s), mats["shoes"], segments=14), f"foot.{side}")
        sx = 1 if side == "L" else -1
        add(C.ellipsoid(f"{name}_shoulder.{side}", Vector((sx * 0.18, 0, 1.37)) * s, (0.07 * s, 0.07 * s, 0.06 * s), mats["shirt"], segments=14), "spine")
    for pb in ao.pose.bones:
        pb.rotation_mode = "XYZ"
    return ao


def channels(angles):
    """Map gait joint angles [..., len(gait.JOINTS)] to pose channels: list of (bone, prop, index, values)."""
    import numpy as np
    A = np.asarray(angles, dtype=np.float64)
    J = {n: A[..., i] for i, n in enumerate(gait.JOINTS)}
    out = [(bone, "rotation_euler", 0, sign * J[joint]) for bone, (joint, sign) in BONE_DRIVE.items()]
    zero = np.zeros_like(J["spine_twist"])
    out += [("spine", "rotation_euler", 0, zero + 0.04),  # slight forward lean
            ("spine", "rotation_euler", 1, J["spine_twist"]),
            ("pelvis", "location", 1, J["pelvis_bob"]),  # pelvis local Y == world up
            ("pelvis", "rotation_euler", 1, -0.5 * J["spine_twist"])]
    return out


def pose_human(ao, angles, frame=None):
    """Apply one row of gait.joint_angles to an armature; optionally keyframe it."""
    for bone, prop, idx, val in channels(angles):
        pb = ao.pose.bones[bone]
        getattr(pb, prop)[idx] = float(val)
        if frame is not None:
            pb.keyframe_insert(prop, index=idx, frame=frame)


def _fcurve(action, path, index, frames, values, group=None, interp="LINEAR"):
    import numpy as np
    fc = action.fcurves.new(path, index=index, action_group=group or "")
    fc.keyframe_points.add(len(frames))
    co = np.empty(2 * len(frames), dtype=np.float32)
    co[0::2], co[1::2] = frames, values
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", [bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items[interp].value] * len(frames))
    fc.update()
    return fc


def bake_motion(ao, pos, yaw, joints, frame_start=1):
    """Bake a whole trajectory (pos [T,2], yaw [T], joints [T,J]) into one action, fast (no per-key ops)."""
    import numpy as np
    T = len(pos)
    frames = np.arange(T, dtype=np.float32) + frame_start
    ao.animation_data_create()
    act = bpy.data.actions.new(f"{ao.name}_motion")
    ao.animation_data.action = act
    _fcurve(act, "location", 0, frames, pos[:, 0], "root")
    _fcurve(act, "location", 1, frames, pos[:, 1], "root")
    _fcurve(act, "location", 2, frames, np.zeros(T), "root")
    _fcurve(act, "rotation_euler", 2, frames, np.unwrap(yaw) + math.pi / 2, "root")  # rest pose faces -Y
    for bone, prop, idx, val in channels(joints):
        _fcurve(act, f'pose.bones["{bone}"].{prop}', idx, frames, val, bone)
    return act


def add_walk_cycle(ao, speed=1.3, fps=30):
    """Bake one looping walk cycle in place as an action named '<name>_walk'."""
    period = 1.0 / float(gait.cadence_hz(speed))
    n = int(round(period * fps))
    ao.animation_data_create()
    ao.animation_data.action = bpy.data.actions.new(f"{ao.name}_walk")
    for i in range(n + 1):
        phase = 2 * math.pi * i / n
        pose_human(ao, gait.joint_angles(phase, speed), frame=i + 1)
    for fc in ao.animation_data.action.fcurves:
        fc.modifiers.new("CYCLES")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.ASSETS)
    ap.add_argument("--outfit", type=int, default=0)
    a = ap.parse_args(C.script_args())
    os.makedirs(a.out, exist_ok=True)
    C.clear_scene()
    ao = make_human("digital_human", a.outfit)
    n = add_walk_cycle(ao)
    sc = bpy.context.scene
    sc.render.fps = 30
    sc.frame_start, sc.frame_end = 1, n
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.out, "human.blend"))
    bpy.ops.export_scene.gltf(filepath=os.path.join(a.out, "human.glb"), export_format="GLB", export_animations=True)
    print(f"[build_human] rig with {len(ao.pose.bones)} bones, walk cycle {n} frames -> {a.out}")


if __name__ == "__main__":
    main()
