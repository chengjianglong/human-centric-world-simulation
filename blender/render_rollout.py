"""Animate the learned trajectories in the Blender room and render a video.

    tools/blender/blender -b -P blender/render_rollout.py -- --rollout renders/rollout.npz \
        [--out renders/rollout.mp4] [--camera corner|top|follow] [--engine eevee|cycles] \
        [--frames 300] [--res 1280x720] [--samples 32] [--still 120] [--room-scan scan.glb]
        [--humans rocketbox|mannequin] [--avatars Male_Adult_01 Female_Adult_04 ...]

--humans rocketbox (default when assets/rocketbox exists) uses photo-realistic Microsoft Rocketbox
avatars animated with mocap; use --engine cycles for realistic skin/hair.

Always also saves the animated scene as a .blend next to --out (open it in Blender to explore).
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from build_human import bake_motion, make_human  # noqa: E402


def setup_gpu_cycles(scene, samples):
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for t in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = t
            prefs.get_devices()
            if any(d.type == t for d in prefs.devices):
                break
        except Exception:
            continue
    for d in prefs.devices:
        d.use = d.type != "CPU"
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True


def goal_markers(data, col, start):
    """Small glowing floor discs showing where each human is heading (constant interpolation)."""
    from build_human import OUTFITS, _fcurve
    T, N = data["goal"].shape[:2]
    frames = np.arange(T, dtype=np.float32) + start
    for i in range(N):
        shirt = OUTFITS[i % len(OUTFITS)][0]
        m = C.material(f"goal_{i}", shirt, 0.5, emission=(*shirt, 2.0))
        o = C.cylinder(f"goal_{i}", (0, 0, 0.004), 0.12, 0.008, m, collection=col)
        o.animation_data_create()
        act = bpy.data.actions.new(f"goal_{i}_motion")
        o.animation_data.action = act
        for k in (0, 1):
            _fcurve(act, "location", k, frames, data["goal"][:, i, k], interp="CONSTANT")


def follow_camera(scene, data, start, target, subtarget="spine"):
    from build_human import _fcurve
    cam = bpy.data.cameras.new("cam_follow")
    cam.lens = 18
    co = C.link(bpy.data.objects.new("cam_follow", cam))
    pos = data["pos"][:, 0]
    k = 15  # smooth over ~0.75 s
    sm = np.stack([np.convolve(np.pad(pos[:, j], (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid") for j in (0, 1)], -1)
    yaw = np.unwrap(data["yaw"][:, 0])
    yaw = np.convolve(np.pad(yaw, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    camxy = sm - 1.6 * np.stack([np.cos(yaw), np.sin(yaw)], -1)
    W, D = data["room_wh"]
    camxy = np.clip(camxy, 0.15, [W - 0.15, D - 0.15])
    frames = np.arange(len(pos), dtype=np.float32) + start
    co.animation_data_create()
    act = bpy.data.actions.new("cam_follow_motion")
    co.animation_data.action = act
    _fcurve(act, "location", 0, frames, camxy[:, 0])
    _fcurve(act, "location", 1, frames, camxy[:, 1])
    _fcurve(act, "location", 2, frames, np.full(len(pos), 1.75))
    tr = co.constraints.new("TRACK_TO")
    tr.target, tr.subtarget = target, subtarget
    tr.track_axis, tr.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    return co


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--out")
    ap.add_argument("--room", default=os.path.join(C.ASSETS, "room.blend"))
    ap.add_argument("--room-scan", help="optional .glb/.gltf/.obj scan of the real room to render instead of the procedural shell")
    ap.add_argument("--camera", default="corner", choices=["corner", "top", "follow"])
    ap.add_argument("--engine", choices=["eevee", "cycles"], help="default: cycles for rocketbox, eevee for mannequin")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--frames", type=int, help="max frames to render")
    ap.add_argument("--still", type=int, help="render only this frame to PNG")
    ap.add_argument("--no-goals", action="store_true")
    ap.add_argument("--humans", choices=["rocketbox", "mannequin"])
    ap.add_argument("--avatars", nargs="*", help="Rocketbox avatar names (default: all downloaded, in order)")
    a = ap.parse_args(C.script_args())
    out = a.out or os.path.splitext(a.rollout)[0] + f"_{a.camera}.mp4"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=a.room)
    scene = bpy.context.scene
    data = dict(np.load(a.rollout, allow_pickle=True))
    floor = bpy.data.objects["floor"]
    data["room_wh"] = (floor.dimensions.x, floor.dimensions.y)
    T, N = data["pos"].shape[:2]
    T = min(T, a.frames) if a.frames else T

    if a.room_scan:
        for name in ("room_shell", "furniture"):
            bpy.data.collections[name].hide_render = True
        scan = C.get_collection("room_scan")
        before = set(bpy.data.objects)
        ext = os.path.splitext(a.room_scan)[1].lower()
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=a.room_scan)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=a.room_scan, up_axis="Z", forward_axis="Y")
        for o in set(bpy.data.objects) - before:
            C._move_to(o, scan)

    import rocketbox as RBX
    kind = a.humans or ("rocketbox" if RBX.available_avatars() else "mannequin")
    humans = C.get_collection("humans")
    rigs, follow_bone = [], "spine"
    if kind == "rocketbox":
        cast = a.avatars or RBX.available_avatars()
        if not cast:
            raise SystemExit("no Rocketbox avatars: run python scripts/fetch_rocketbox.py")
        for i in range(N):
            ao = RBX.import_avatar(cast[i % len(cast)], humans, f"human{i}")
            root = C.empty(f"human{i}_root", collection=humans)
            RBX.animate(ao, root, data["pos"][:T, i], data["yaw"][:T, i], float(data["dt"]))
            rigs.append(ao)
        follow_bone = "Bip01 Spine2"
    else:
        for i in range(N):
            ao = make_human(f"human{i}", outfit=i, collection=humans)
            bake_motion(ao, data["pos"][:T, i], data["yaw"][:T, i], data["joints"][:T, i])
            rigs.append(ao)
    print(f"[render] humans: {kind} x {N}")
    a.engine = a.engine or ("cycles" if kind == "rocketbox" else "eevee")
    if not a.no_goals:
        goal_markers({k: (v[:T] if k == "goal" else v) for k, v in data.items()}, C.get_collection("goals"), 1)

    res_x, res_y = (int(v) for v in a.res.lower().split("x"))
    scene.render.resolution_x, scene.render.resolution_y = res_x, res_y
    if a.camera == "corner":
        scene.camera = bpy.data.objects["cam_corner"]
    elif a.camera == "top":
        cam = bpy.data.objects["cam_top"]
        W, D = data["room_wh"]
        cam.data.ortho_scale = max(W, D * res_x / res_y) * 1.06
        for name in ("ceiling", "ceiling_fixture"):
            bpy.data.objects[name].hide_render = True
        scene.camera = cam
    else:
        scene.camera = follow_camera(scene, data, 1, rigs[0], follow_bone)

    fps = int(round(1.0 / float(data["dt"])))
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, T
    scene.view_settings.view_transform = "AgX"
    if a.engine == "cycles":
        setup_gpu_cycles(scene, a.samples)
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = a.samples
        scene.eevee.use_shadows = True
        scene.eevee.use_raytracing = True

    blend_path = os.path.splitext(out)[0] + ".blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[render] scene with {N} humans x {T} frames saved -> {blend_path}")
    if a.still is not None:
        scene.frame_set(a.still)
        scene.render.filepath = os.path.splitext(out)[0] + f"_f{a.still}.png"
        bpy.ops.render.render(write_still=True)
        print(f"[render] still -> {scene.render.filepath}")
        return
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = out
    bpy.ops.render.render(animation=True)
    print(f"[render] video -> {out} ({T} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
