"""Photo-realistic digital humans: Microsoft Rocketbox avatars (MIT) driven by Rocketbox mocap.

The trained policy provides each human's root path (x, y, heading, speed). Limb motion comes
from real motion-capture clips layered in Blender's NLA:

    idle (in place)  <  walk_slow  <  walk_neutral  <  walk_fast       (REPLACE blending)

The walk influences follow the policy's speed, and every walk strip's clip time follows ONE
shared gait phase that advances by distance travelled / stride length, so the feet stay
planted instead of skating. Clips are made in-place (forward drift removed) and
phase-aligned automatically (from the left/right foot oscillation).

Get the assets first:  python scripts/fetch_rocketbox.py
"""
import math
import os

import bpy
import numpy as np

import common as C

RB = os.path.join(C.ASSETS, "rocketbox")
CLIP_FPS = 30.0
WALKS = ("walk_slow_01", "walk_neutral_01", "walk_fast_01")
IDLE = "idle_neutral_01"
_CLIPS = {}
_REST = {}


def rest_locals(arm):
    """Each bone's rest rotation relative to its parent (armature space for root bones)."""
    out = {}
    for b in arm.data.bones:
        q = b.matrix_local.to_quaternion()
        if b.parent:
            q = b.parent.matrix_local.to_quaternion().inverted() @ q
        out[b.name] = q
    return out


def _qmul(a, b):
    """Hamilton product, a: (4,), b: (T, 4) -> (T, 4)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def available_avatars():
    d = os.path.join(RB, "avatars")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def gender(name):
    return "f" if name.lower().startswith("female") else "m"


# ------------------------------------------------------------------ materials
def _img(path, non_color=False):
    img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def _find_tex(tex_dir, stem):
    for f in os.listdir(tex_dir):
        if f.lower() == stem.lower():
            return os.path.join(tex_dir, f)
    return None


def build_material(mat, tex_dir):
    """Rebuild a Rocketbox material from its textures (FBX paths point at the authors' machine)."""
    nt = mat.node_tree
    old = [n.image.filepath for n in nt.nodes if n.type == "TEX_IMAGE" and n.image]
    stems = {os.path.basename(p.replace("\\", "/")).lower() for p in old}
    color = next((s for s in stems if s.endswith("_color.tga")), None)
    normal = next((s for s in stems if s.endswith("_normal.tga")), None)
    spec = next((s for s in stems if s.endswith("_specular.tga")), None)
    is_opacity = "opacity" in mat.name.lower() or (color and "opacity" in color)
    is_head = "head" in mat.name.lower()
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if color and _find_tex(tex_dir, color):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _img(_find_tex(tex_dir, color))
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        if is_opacity:  # hair cards / eyelashes: RGBA with alpha cut-out
            t.image.alpha_mode = "STRAIGHT"
            nt.links.new(t.outputs["Alpha"], bsdf.inputs["Alpha"])
        else:  # light subsurface scattering for skin (head is mostly skin, body has clothes)
            bsdf.inputs["Subsurface Weight"].default_value = 0.25 if is_head else 0.06
            bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
            bsdf.inputs["Subsurface Scale"].default_value = 0.01
    if normal and _find_tex(tex_dir, normal):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _img(_find_tex(tex_dir, normal), non_color=True)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 1.0
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if spec and _find_tex(tex_dir, spec):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _img(_find_tex(tex_dir, spec), non_color=True)
        bw = nt.nodes.new("ShaderNodeRGBToBW")
        rough = nt.nodes.new("ShaderNodeMapRange")  # bright specular map -> glossier
        rough.inputs["To Min"].default_value = 0.85
        rough.inputs["To Max"].default_value = 0.35
        nt.links.new(t.outputs["Color"], bw.inputs["Color"])
        nt.links.new(bw.outputs["Val"], rough.inputs["Value"])
        nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
        nt.links.new(bw.outputs["Val"], bsdf.inputs["Specular IOR Level"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.6
    mat.blend_method = "HASHED" if is_opacity else "OPAQUE"
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    mat.use_backface_culling = False


# ------------------------------------------------------------------ avatar
def import_avatar(name, collection, tag):
    """Import a Rocketbox avatar; returns its armature object (named `tag`)."""
    path = os.path.join(RB, "avatars", name, "Export", f"{name}.fbx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run python scripts/fetch_rocketbox.py --avatars {name}")
    before = set(bpy.data.objects)
    mats_before = set(bpy.data.materials)
    bpy.ops.import_scene.fbx(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    tex_dir = os.path.join(RB, "avatars", name, "Textures")
    for m in set(bpy.data.materials) - mats_before:
        build_material(m, tex_dir)
    arm = next(o for o in new if o.type == "ARMATURE")
    for o in new:
        if o.type == "EMPTY":  # Biped footsteps/helpers
            bpy.data.objects.remove(o)
            continue
        C._move_to(o, collection)
        if o.type == "MESH":
            o.name = f"{tag}_{o.name}"
    arm.name = tag
    arm["avatar"] = name
    _REST[tag] = rest_locals(arm)
    if arm.animation_data:
        arm.animation_data.action = None
    # pelvis height of this avatar (to adapt clip root height to body size)
    arm["pelvis_z"] = float(arm.location.z) if arm.location.z > 0.3 else float(
        (arm.matrix_world @ arm.pose.bones["Bip01 Pelvis"].head).z)
    return arm


# ------------------------------------------------------------------ clips
def _foot_signal(arm, act, f0, f1, n=16):
    arm.animation_data.action = act
    sc = bpy.context.scene
    sig = []
    for k in range(n):
        sc.frame_set(int(round(f0 + (f1 - f0) * k / n)))
        l = arm.matrix_world @ arm.pose.bones["Bip01 L Foot"].head
        r = arm.matrix_world @ arm.pose.bones["Bip01 R Foot"].head
        sig.append((r - l).y)  # >0: left foot ahead (clips walk toward -Y)
    return np.array(sig)


def load_clip(clip):
    """Import a clip FBX once -> dict(action, f0, f1, stride, phase0). Walk clips are made in-place."""
    if clip in _CLIPS:
        return _CLIPS[clip]
    path = os.path.join(RB, "animations", f"{clip}.fbx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run python scripts/fetch_rocketbox.py")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == "ARMATURE")
    act = arm.animation_data.action
    act.name = f"clip_{clip}"
    act.use_fake_user = True
    f0, f1 = (float(v) for v in act.frame_range)
    loc = {fc.array_index: fc for fc in act.fcurves if fc.data_path == "location"}
    # forward drift per cycle (stride) before making it in place
    stride = math.hypot(*(loc[i].keyframe_points[-1].co[1] - loc[i].keyframe_points[0].co[1] for i in (0, 1)))
    phase0 = 0.0
    if "walk" in clip:
        sig = _foot_signal(arm, act, f0, f1)
        phase0 = (np.angle(np.fft.rfft(sig)[1]) / (2 * math.pi)) % 1.0
    for i in (0, 1):  # remove linear drift + start offset in x/y -> loops in place
        kp = loc[i].keyframe_points
        a, b = kp[0].co[1], kp[-1].co[1]
        for k in kp:
            t = (k.co[0] - f0) / max(f1 - f0, 1e-6)
            d = a + (b - a) * t if "walk" in clip else a
            k.co[1] -= d
            k.handle_left[1] -= d
            k.handle_right[1] -= d
        loc[i].update()
    rest = rest_locals(arm)
    for o in new:
        bpy.data.objects.remove(o)
    info = dict(action=act, rest=rest, f0=f0, f1=f1, stride=stride, phase0=phase0, z=loc[2].keyframe_points[0].co[1])
    _CLIPS[clip] = info
    return info


def clip_set(g):
    walks = [load_clip(f"{g}_{w}") for w in WALKS]
    ref = walks[1]["phase0"]
    for w in walks:  # clip time u = shared phase + offset puts every clip on the same foot
        w["offset"] = (ref - w["phase0"]) % 1.0
    for w in walks:
        w["speed"] = w["stride"] / ((w["f1"] - w["f0"]) / CLIP_FPS)
    return load_clip(f"{g}_{IDLE}"), walks


def _per_avatar_action(info, arm, tag):
    """Copy a clip action and retarget it to this avatar.

    Clip and avatar skeletons share bone names/hierarchy but not rest poses (clip rest is
    arms-down, avatar rest is a T-pose). With L = a bone's rest rotation relative to its parent,
    q_avatar = (L_avatar^-1 L_clip) q_clip gives every bone the clip's armature-space orientation.
    The root height is also rescaled to the avatar's body size.
    """
    act = info["action"].copy()
    act.name = f"{tag}_{info['action'].name}"
    rest_a, rest_c = _REST.get(tag, {}), info["rest"]
    quat = {}
    for fc in act.fcurves:
        if fc.data_path.startswith("pose.bones") and fc.data_path.endswith("rotation_quaternion"):
            quat.setdefault(fc.data_path.split('"')[1], {})[fc.array_index] = fc
    for bone, fcs in quat.items():
        if bone not in rest_a or bone not in rest_c or len(fcs) != 4:
            continue
        K = rest_a[bone].inverted() @ rest_c[bone]
        n = len(fcs[0].keyframe_points)
        if any(len(fcs[i].keyframe_points) != n for i in range(4)):
            continue
        q = np.empty((4, n))
        for i in range(4):
            co = np.empty(2 * n)
            fcs[i].keyframe_points.foreach_get("co", co)
            q[i] = co[1::2]
            frames = co[0::2]
        q2 = _qmul(np.array(K), q.T)
        # keep quaternion hemisphere continuous so linear interpolation never takes the long way
        for k in range(1, n):
            if (q2[k] * q2[k - 1]).sum() < 0:
                q2[k] *= -1
        for i in range(4):
            co = np.empty(2 * n)
            co[0::2], co[1::2] = frames, q2[:, i]
            fcs[i].keyframe_points.foreach_set("co", co)
            for kp in fcs[i].keyframe_points:
                kp.interpolation = "LINEAR"
            fcs[i].update()
    ratio = arm.get("pelvis_z", info["z"]) / info["z"] if info["z"] else 1.0
    for fc in act.fcurves:
        if fc.data_path == "location" and fc.array_index == 2 and abs(ratio - 1) > 1e-3:
            for k in fc.keyframe_points:
                k.co[1] *= ratio
                k.handle_left[1] *= ratio
                k.handle_right[1] *= ratio
            fc.update()
    return act


def _smooth(x, k):
    if k <= 1:
        return x
    pad = np.pad(x, (k, k), mode="edge")
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), "valid")


def locomotion_params(pos, yaw, dt, walks):
    """Per-frame walk weight, blend factors and synced gait phase from the policy trajectory."""
    T = len(pos)
    step = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    v = _smooth(step / dt, 3)
    w = np.r_[0.0, np.abs(np.diff(np.unwrap(yaw))) / dt]
    v_anim = np.maximum(v, 0.22 * _smooth(w, 3))  # turning on the spot still steps
    walk_w = np.clip((v_anim - 0.06) / 0.3, 0, 1)
    walk_w = walk_w * walk_w * (3 - 2 * walk_w)
    s = [c["speed"] for c in walks]
    a1 = np.clip((v_anim - s[0]) / max(s[1] - s[0], 1e-3), 0, 1)  # slow -> neutral
    a2 = np.clip((v_anim - s[1]) / max(s[2] - s[1], 1e-3), 0, 1)  # neutral -> fast
    st = [c["stride"] for c in walks]
    stride = np.where(a2 > 0, st[1] + (st[2] - st[1]) * a2, st[0] + (st[1] - st[0]) * a1)
    # walking backwards (motion opposite to heading) plays the cycle in reverse instead of moonwalking
    d = np.diff(pos, axis=0, prepend=pos[:1])
    fwd = d[:, 0] * np.cos(yaw) + d[:, 1] * np.sin(yaw)
    sign = np.where(_smooth(fwd, 3) < -1e-4, -1.0, 1.0)
    phase = np.cumsum(sign * v_anim * dt / stride)  # gait cycles
    return dict(walk=walk_w, a1=a1, a2=a2, phase=phase, v=v_anim)


def _strip_fcurve(strip, prop, frames, values):
    fc = strip.fcurves.find(prop)
    if fc is None:
        strip.keyframe_insert(prop, frame=float(frames[0]))
        fc = strip.fcurves.find(prop)
    kp = fc.keyframe_points
    while len(kp):
        kp.remove(kp[0], fast=True)
    kp.add(len(frames))
    co = np.empty(2 * len(frames), dtype=np.float32)
    co[0::2], co[1::2] = frames, values
    kp.foreach_set("co", co)
    lin = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value
    kp.foreach_set("interpolation", [lin] * len(frames))
    fc.update()


def animate(arm, root, pos, yaw, dt, frame_start=1):
    """Drive avatar `arm` (parented to empty `root`) along a policy trajectory with blended mocap."""
    from build_human import _fcurve
    T = len(pos)
    frames = np.arange(T, dtype=np.float32) + frame_start
    # root path: rest pose of the clips walks toward -Y, so heading = yaw + 90 deg
    root.animation_data_create()
    ract = bpy.data.actions.new(f"{root.name}_path")
    root.animation_data.action = ract
    _fcurve(ract, "location", 0, frames, pos[:, 0], "root")
    _fcurve(ract, "location", 1, frames, pos[:, 1], "root")
    _fcurve(ract, "location", 2, frames, np.zeros(T), "root")
    _fcurve(ract, "rotation_euler", 2, frames, np.unwrap(yaw) + math.pi / 2, "root")
    arm.parent = root
    arm.matrix_parent_inverse.identity()

    idle, walks = clip_set(gender(arm.get("avatar", "m")))
    p = locomotion_params(pos, yaw, dt, walks)
    t_sec = np.arange(T) * dt
    ad = arm.animation_data_create()
    ad.action = None
    span = lambda c: c["f1"] - c["f0"]  # noqa: E731
    layers = [("idle", idle, np.ones(T), idle["f0"] + np.mod(t_sec * CLIP_FPS + 37 * sum(map(ord, arm.name)), span(idle)))]
    infl = [p["walk"], p["walk"] * p["a1"], p["walk"] * p["a2"]]
    for name, c, inf in zip(("walk_slow", "walk_neutral", "walk_fast"), walks, infl):
        layers.append((name, c, inf, c["f0"] + np.mod(p["phase"] + c["offset"], 1.0) * span(c)))
    for name, c, inf, ctime in layers:
        tr = ad.nla_tracks.new()
        tr.name = name
        act = _per_avatar_action(c, arm, arm.name)
        st = tr.strips.new(name, int(frame_start), act)
        st.blend_type = "REPLACE"
        st.extrapolation = "HOLD"
        st.use_sync_length = False
        st.repeat = max(1.0, math.ceil(T / span(c)) + 1)
        st.use_animated_time = True
        _strip_fcurve(st, "strip_time", frames, ctime)
        if name != "idle":
            st.use_animated_influence = True
            _strip_fcurve(st, "influence", frames, inf)
    return p
