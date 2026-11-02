"""Procedural cars for the street (bpy). Local frame: +x = forward, origin on the ground at the car's centre."""
import math

import bpy
import numpy as np

import common as C


def build_car(name, color, size=(4.5, 1.85, 1.45), collection=None):
    L, W, H = size
    root = C.empty(name, collection=collection)
    paint = C.material(f"carpaint_{name}", color, 0.25, metallic=0.6)
    glass = C.material("car_glass", (0.05, 0.07, 0.09), 0.05, metallic=0.2)
    rubber = C.material("tyre", (0.03, 0.03, 0.03), 0.85)
    chrome = C.material("chrome", (0.8, 0.8, 0.82), 0.15, metallic=1.0)
    head = C.material("headlight", (1.0, 1.0, 0.95), 0.1, emission=(1.0, 0.97, 0.9, 2.0))
    tail = C.material("taillight", (0.6, 0.02, 0.02), 0.2, emission=(0.9, 0.05, 0.03, 1.5))
    dark = C.material("car_trim", (0.04, 0.04, 0.045), 0.6)
    clearance, body_h = 0.22, H * 0.42
    C.box(f"{name}_body", (0, 0, clearance + body_h / 2), (L, W, body_h), paint, root, bevel=0.09)
    cab_l, cab_h = L * 0.5, H - clearance - body_h - 0.04
    cx = -L * 0.06
    zc = clearance + body_h + cab_h / 2 - 0.02
    C.box(f"{name}_cabin", (cx, 0, zc), (cab_l, W * 0.86, cab_h), glass, root, bevel=0.12)
    C.box(f"{name}_roof", (cx - 0.05, 0, clearance + body_h + cab_h - 0.02), (cab_l * 0.78, W * 0.84, 0.06), paint, root, bevel=0.03)
    for sx in (-1, 1):  # pillars between the windows
        C.box(f"{name}_pillar", (cx + sx * cab_l * 0.12, 0, zc), (0.07, W * 0.87, cab_h * 0.95), paint, root)
    C.box(f"{name}_bumper_f", (L / 2 - 0.02, 0, clearance + 0.12), (0.1, W * 0.96, 0.18), dark, root, bevel=0.03)
    C.box(f"{name}_bumper_r", (-L / 2 + 0.02, 0, clearance + 0.12), (0.1, W * 0.96, 0.18), dark, root, bevel=0.03)
    for sy in (-1, 1):
        C.box(f"{name}_headlight", (L / 2 + 0.005, sy * (W / 2 - 0.28), clearance + body_h * 0.7), (0.04, 0.32, 0.1), head, root)
        C.box(f"{name}_taillight", (-L / 2 - 0.005, sy * (W / 2 - 0.22), clearance + body_h * 0.72), (0.04, 0.26, 0.1), tail, root)
        C.box(f"{name}_mirror", (cx + cab_l * 0.38, sy * (W / 2 + 0.08), clearance + body_h + 0.1), (0.12, 0.16, 0.1), paint, root)
    C.box(f"{name}_grille", (L / 2 + 0.01, 0, clearance + body_h * 0.45), (0.03, W * 0.4, 0.14), dark, root)
    wheels = []
    r = 0.33
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        hub = C.empty(f"{name}_wheel{i}", (sx * L * 0.31, sy * (W / 2 - 0.12), r), collection=collection)
        hub.parent = root
        hub.rotation_mode = "ZXY"
        hub.rotation_euler = (math.pi / 2, 0, 0)
        t = C.cylinder(f"{name}_tyre{i}", (0, 0, 0), r, 0.24, rubber, hub, verts=24)
        t.location = (0, 0, 0)
        rim = C.cylinder(f"{name}_rim{i}", (0, 0, 0), r * 0.6, 0.25, chrome, hub, verts=12)
        rim.location = (0, 0, 0)
        spoke = C.box(f"{name}_spoke{i}", (0, 0, 0), (r * 1.1, 0.06, 0.255), dark, hub)
        spoke.location = (0, 0, 0)
        for o in (t, rim):
            C._move_to(o, collection)
        wheels.append(hub)
    return root, wheels, r


def animate_car(root, wheels, radius, pos, yaw, frame_start=1):
    """pos [T, 2], yaw [T] -> keyframes; wheels spin with the distance travelled (teleports on lane wrap ignored)."""
    from build_human import _fcurve
    T = len(pos)
    frames = np.arange(T, dtype=np.float32) + frame_start
    root.animation_data_create()
    act = bpy.data.actions.new(f"{root.name}_motion")
    root.animation_data.action = act
    _fcurve(act, "location", 0, frames, pos[:, 0], interp="CONSTANT")
    _fcurve(act, "location", 1, frames, pos[:, 1], interp="CONSTANT")
    _fcurve(act, "rotation_euler", 2, frames, yaw, interp="CONSTANT")
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    step[step > 3.0] = 0.0  # lane wrap
    dist = np.concatenate([[0.0], np.cumsum(step)])
    spin = -dist / radius
    for i, hub in enumerate(wheels):
        hub.animation_data_create()
        a = bpy.data.actions.new(f"{hub.name}_spin")
        hub.animation_data.action = a
        _fcurve(a, "rotation_euler", 0, frames, np.full(T, math.pi / 2, dtype=np.float32))
        _fcurve(a, "rotation_euler", 2, frames, spin.astype(np.float32))
