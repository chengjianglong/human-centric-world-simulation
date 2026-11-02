"""Build the 3D room from configs/room.json.

    tools/blender/blender -b -P blender/build_room.py -- [--spec configs/room.json] [--out assets]

Writes assets/room.blend, assets/room.glb and assets/room_layout.json (the colliders/POIs
the PyTorch env uses -- computed by the same roomsim.room_spec code, so they match).
"""
import argparse
import math
import os
import random
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from roomsim import room_spec  # noqa: E402


# ------------------------------------------------------------------ walls
def wall_segments(spec, wall):
    """Yield (u0, u1, z0, z1) solid pieces of a wall, cutting out its openings."""
    W, D, H, t = spec["width"], spec["depth"], spec["height"], spec["wall_thickness"]
    horiz = wall in ("south", "north")
    u_lo, u_hi = (-t, W + t) if horiz else (0.0, D)
    ops = sorted((o for o in spec.get("openings", []) if o["wall"] == wall), key=lambda o: o["offset"])
    u = u_lo
    for o in ops:
        a, b = o["offset"], o["offset"] + o["width"]
        if a > u:
            yield (u, a, 0.0, H)
        if o["sill"] > 0:
            yield (a, b, 0.0, o["sill"])
        top = o["sill"] + o["height"]
        if top < H:
            yield (a, b, top, H)
        u = b
    if u < u_hi:
        yield (u, u_hi, 0.0, H)


def wall_box(spec, wall, u0, u1, z0, z1, inset=0.0, thick=None):
    """Centre/size of a box on `wall` spanning u0..u1 along the wall and z0..z1 vertically.

    thick=None: the box fills the wall's thickness (outside the interior).
    thick=x:    a trim piece x thick standing `inset` metres inside the room against the wall.
    """
    W, D, t = spec["width"], spec["depth"], spec["wall_thickness"]
    um, du, zm, dz = (u0 + u1) / 2, u1 - u0, (z0 + z1) / 2, z1 - z0
    th, n = (t, -t / 2) if thick is None else (thick, inset + thick / 2)  # n = offset from interior face
    if wall == "south":
        return (um, n, zm), (du, th, dz)
    if wall == "north":
        return (um, D - n, zm), (du, th, dz)
    if wall == "west":
        return (n, um, zm), (th, du, dz)
    return (W - n, um, zm), (th, du, dz)


def build_walls(spec, col, mats):
    for wall in ("south", "north", "west", "east"):
        for i, (u0, u1, z0, z1) in enumerate(wall_segments(spec, wall)):
            c, s = wall_box(spec, wall, u0, u1, z0, z1)
            C.box(f"wall_{wall}_{i}", c, s, mats["walls"], collection=col)
        # baseboards (skirting), skipping doors
        doors = [o for o in spec.get("openings", []) if o["wall"] == wall and o["sill"] == 0]
        length = spec["width"] if wall in ("south", "north") else spec["depth"]
        u, cuts = 0.0, sorted((o["offset"], o["offset"] + o["width"]) for o in doors)
        for a, b in cuts + [(length, length)]:
            if a > u:
                c, s = wall_box(spec, wall, u, a, 0.0, 0.09, inset=0.0, thick=0.015)
                C.box(f"skirting_{wall}_{u:.2f}", c, s, mats["trim"], collection=col)
            u = b


def build_openings(spec, col, mats):
    t = spec["wall_thickness"]
    for o in spec.get("openings", []):
        a, b = o["offset"], o["offset"] + o["width"]
        z0, z1 = o["sill"], o["sill"] + o["height"]
        fw = 0.05  # frame width
        frame = [(a, a + fw, z0, z1), (b - fw, b, z0, z1), (a, b, z1 - fw, z1)]
        if o["type"] == "window":
            frame.append((a, b, z0, z0 + fw))
            frame.append(((a + b) / 2 - 0.02, (a + b) / 2 + 0.02, z0, z1))  # mullion
        for i, (u0, u1, v0, v1) in enumerate(frame):
            c, s = wall_box(spec, o["wall"], u0, u1, v0, v1)
            s = tuple(x * 1.02 if x == t else x for x in s)
            C.box(f"{o['name']}_frame_{i}", c, s, mats["trim"], collection=col)
        if o["type"] == "window":
            c, s = wall_box(spec, o["wall"], a + fw, b - fw, z0 + fw, z1 - fw)
            s = tuple(0.01 if x == t else x for x in s)
            C.box(f"{o['name']}_glass", c, s, mats["glass"], collection=col)
            # sill board
            c, s = wall_box(spec, o["wall"], a - 0.05, b + 0.05, z0 - 0.03, z0, inset=-t, thick=t + 0.05)
            C.box(f"{o['name']}_sill", c, s, mats["trim"], collection=col)
        elif room_spec.door_leaf(spec, o):  # swung open 90 deg (same footprint the physics uses)
            lf = room_spec.door_leaf(spec, o)
            h = lf["height"]
            C.box(f"{o['name']}_leaf", (lf["cx"], lf["cy"], z0 + h / 2), (2 * lf["hx"], 2 * lf["hy"], h), mats["door"],
                  collection=col, bevel=0.005)
            # handle near the free edge, on both faces
            n_far = (lf["length"] - 0.08) if lf["inward"] else -(t + lf["length"] - 0.08)
            hx, hy = room_spec.wall_point(spec, o["wall"], lf["hinge_u"], n_far)
            along_x = o["wall"] in ("south", "north")
            C.box(f"{o['name']}_handle", (hx, hy, 1.0), (0.1, 0.03, 0.03) if along_x else (0.03, 0.1, 0.03), mats["metal"], collection=col)
        else:
            c, s = wall_box(spec, o["wall"], a + fw, b - fw, z0, z1 - fw)
            s = tuple(0.04 if x == t else x for x in s)
            C.box(f"{o['name']}_leaf", c, s, mats["door"], collection=col, bevel=0.005)
            hu = b - fw - 0.08
            c, s = wall_box(spec, o["wall"], hu - 0.06, hu + 0.06, 1.0, 1.03, thick=0.06)
            C.box(f"{o['name']}_handle", c, s, mats["metal"], collection=col)


# ------------------------------------------------------------------ furniture (local frame: front = -y)
def f_sofa(f, p, mat, mats):
    w, d, h = f["size"]
    seat_h, arm_w, back_d = 0.44, 0.18, 0.22
    for sx in (-1, 1):
        for sy in (-1, 1):
            C.box(f"{f['name']}_leg", (sx * (w / 2 - 0.08), sy * (d / 2 - 0.08), 0.04), (0.05, 0.05, 0.08), mats["wood_dark"], p)
    C.box(f"{f['name']}_base", (0, 0, 0.08 + (seat_h - 0.12 - 0.08) / 2), (w, d, seat_h - 0.2), mat, p, bevel=0.02)
    inner = w - 2 * arm_w
    n = 3 if inner > 1.5 else 2
    cw = inner / n
    for i in range(n):
        x = -inner / 2 + cw * (i + 0.5)
        C.box(f"{f['name']}_cushion{i}", (x, -back_d / 2, seat_h - 0.06), (cw - 0.01, d - back_d, 0.14), mat, p, bevel=0.04)
        C.box(f"{f['name']}_backcushion{i}", (x, d / 2 - back_d - 0.06, seat_h + 0.2), (cw - 0.02, 0.16, 0.36), mat, p, bevel=0.05)
    C.box(f"{f['name']}_back", (0, d / 2 - back_d / 2, (0.08 + h) / 2), (w, back_d, h - 0.08), mat, p, bevel=0.03)
    for sx in (-1, 1):
        C.box(f"{f['name']}_arm", (sx * (w / 2 - arm_w / 2), 0, (0.08 + 0.64) / 2), (arm_w, d, 0.56), mat, p, bevel=0.04)


def f_table(f, p, mat, mats):
    w, d, h = f["size"]
    C.box(f"{f['name']}_top", (0, 0, h - 0.02), (w, d, 0.04), mat, p, bevel=0.006)
    for sx in (-1, 1):
        for sy in (-1, 1):
            C.box(f"{f['name']}_leg", (sx * (w / 2 - 0.06), sy * (d / 2 - 0.06), (h - 0.04) / 2), (0.05, 0.05, h - 0.04), mat, p)
    if h < 0.5:  # coffee table: lower shelf + a book/mug
        C.box(f"{f['name']}_shelf", (0, 0, 0.1), (w - 0.1, d - 0.1, 0.02), mat, p)
        C.box(f"{f['name']}_book", (0.2, 0.05, h + 0.015), (0.22, 0.16, 0.03), C.material("book_red", (0.5, 0.1, 0.08), 0.7), p, rot_z=0.3)
        C.cylinder(f"{f['name']}_mug", (-0.25, -0.05, h + 0.045), 0.04, 0.09, mats["ceramic"], p)


def f_chair(f, p, mat, mats):
    w, d, h = f["size"]
    seat = 0.46
    C.box(f"{f['name']}_seat", (0, 0, seat), (w, d * 0.92, 0.04), mat, p, bevel=0.005)
    for sx in (-1, 1):
        for sy in (-1, 1):
            C.box(f"{f['name']}_leg", (sx * (w / 2 - 0.03), sy * (d * 0.46 - 0.03), seat / 2), (0.035, 0.035, seat), mat, p)
    for sx in (-1, 1):
        C.box(f"{f['name']}_post", (sx * (w / 2 - 0.03), d * 0.46 - 0.02, (seat + h) / 2), (0.035, 0.035, h - seat), mat, p)
    C.box(f"{f['name']}_backrest", (0, d * 0.46 - 0.02, h - 0.12), (w - 0.04, 0.025, 0.2), mat, p, bevel=0.005)


def f_tv_stand(f, p, mat, mats):
    w, d, h = f["size"]
    C.box(f"{f['name']}_body", (0, 0, h / 2 + 0.03), (w, d, h - 0.06), mat, p, bevel=0.01)
    for sx in (-1, 1):
        C.box(f"{f['name']}_foot", (sx * (w / 2 - 0.05), 0, 0.015), (0.06, d - 0.04, 0.03), mats["metal"], p)
    for i, sx in enumerate((-1, 1)):
        C.box(f"{f['name']}_door{i}", (sx * w / 4, -d / 2 - 0.004, h / 2 + 0.03), (w / 2 - 0.02, 0.008, h - 0.1), mats["wood_dark"], p)
    tw = min(1.45, w * 0.85)
    th = tw * 9 / 16
    C.box(f"{f['name']}_tvbase", (0, 0.02, h + 0.01), (0.3, 0.2, 0.02), mats["metal"], p)
    C.box(f"{f['name']}_tvneck", (0, 0.05, h + 0.06), (0.06, 0.03, 0.1), mats["metal"], p)
    C.box(f"{f['name']}_tv", (0, 0.04, h + 0.08 + th / 2), (tw, 0.04, th), mats["tv_bezel"], p, bevel=0.004)
    C.box(f"{f['name']}_screen", (0, 0.018, h + 0.08 + th / 2), (tw - 0.03, 0.002, th - 0.03), mats["screen"], p)


def f_bookshelf(f, p, mat, mats):
    w, d, h = f["size"]
    t = 0.022
    for sx in (-1, 1):
        C.box(f"{f['name']}_side", (sx * (w / 2 - t / 2), 0, h / 2), (t, d, h), mat, p)
    C.box(f"{f['name']}_back", (0, d / 2 - 0.005, h / 2), (w, 0.01, h), mat, p)
    n = 5
    rng = random.Random(hash(f["name"]) & 0xFFFF)
    palette = [(0.45, 0.08, 0.08), (0.08, 0.2, 0.4), (0.15, 0.3, 0.15), (0.75, 0.65, 0.45), (0.2, 0.2, 0.2), (0.55, 0.35, 0.1)]
    for i in range(n + 1):
        z = 0.06 + i * (h - 0.08) / n
        C.box(f"{f['name']}_shelf{i}", (0, 0, z), (w - 2 * t, d - 0.01, t), mat, p)
        if i == n:
            break
        x = -w / 2 + t + 0.02
        gap = (h - 0.08) / n
        while x < w / 2 - t - 0.06:
            bw = rng.uniform(0.02, 0.05)
            bh = rng.uniform(0.6, 0.85) * gap
            if rng.random() < 0.12:
                x += 0.08
                continue
            col = palette[rng.randrange(len(palette))]
            m = C.material(f"book_{col}", col, 0.7)
            C.box(f"{f['name']}_book", (x + bw / 2, -0.02, z + t / 2 + bh / 2), (bw, d * 0.75, bh), m, p)
            x += bw + 0.003


def f_plant(f, p, mat, mats):
    w, d, h = f["size"]
    r = min(w, d) * 0.38
    C.cylinder(f"{f['name']}_pot", (0, 0, 0.17), r, 0.34, mats["ceramic"], p)
    C.cylinder(f"{f['name']}_soil", (0, 0, 0.33), r * 0.93, 0.02, C.material("soil", (0.12, 0.08, 0.05), 1.0), p)
    rng = random.Random(7)
    leaf = C.material("leaf", f.get("color", (0.2, 0.42, 0.18)), 0.55)
    C.cylinder(f"{f['name']}_stem", (0, 0, (0.34 + h * 0.7) / 2), 0.015, h * 0.7 - 0.34, C.material("stem", (0.25, 0.18, 0.1), 0.8), p, verts=8)
    for i in range(14):
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.05, w * 0.38)
        z = rng.uniform(0.55, h - 0.12)
        C.ellipsoid(f"{f['name']}_leaf{i}", (rr * math.cos(a), rr * math.sin(a), z),
                    (rng.uniform(0.08, 0.14), rng.uniform(0.04, 0.07), rng.uniform(0.03, 0.05)), leaf, p, segments=12)


def f_lamp(f, p, mat, mats):
    w, d, h = f["size"]
    C.cylinder(f"{f['name']}_base", (0, 0, 0.015), w * 0.45, 0.03, mats["metal"], p)
    C.cylinder(f"{f['name']}_pole", (0, 0, h / 2), 0.012, h - 0.25, mats["metal"], p, verts=12)
    shade = C.material("lampshade", f.get("color", (0.92, 0.88, 0.78)), 0.8, transmission=0.3)
    C.cylinder(f"{f['name']}_shade", (0, 0, h - 0.13), w * 0.48, 0.26, shade, p)
    ld = bpy.data.lights.new(f"{f['name']}_bulb", "POINT")
    ld.energy = 40
    ld.color = (1.0, 0.82, 0.6)
    ld.shadow_soft_size = 0.05
    lo = bpy.data.objects.new(f"{f['name']}_bulb", ld)
    p.users_collection[0].objects.link(lo)
    lo.parent = p
    lo.location = (0, 0, h - 0.15)


def f_rug(f, p, mat, mats):
    w, d, h = f["size"]
    C.box(f"{f['name']}_rug", (0, 0, h / 2), (w, d, h), C.fabric_material(f"rug_{f['name']}", f["color"]), p)
    C.box(f"{f['name']}_border", (0, 0, h / 2 + 0.0005), (w - 0.1, d - 0.1, h), C.fabric_material(f"rugb_{f['name']}", [c * 0.7 for c in f["color"]]), p)


def f_generic(f, p, mat, mats):
    w, d, h = f["size"]
    C.box(f["name"], (0, 0, h / 2), (w, d, h), mat, p, bevel=0.01)


BUILDERS = dict(sofa=f_sofa, table=f_table, chair=f_chair, tv_stand=f_tv_stand, bookshelf=f_bookshelf,
                plant=f_plant, lamp=f_lamp, rug=f_rug)


def furniture_material(f):
    col = f.get("color", (0.5, 0.5, 0.5))
    if f["type"] == "sofa":
        return C.fabric_material(f"mat_{f['name']}", col)
    if f["type"] in ("table", "chair", "bookshelf"):
        m = C.material(f"mat_{f['name']}", col, 0.5)
        C._noise_color(m, col, scale=6.0, detail=10.0, var=0.18, stretch=(1, 15, 1))
        return m
    return C.material(f"mat_{f['name']}", col, 0.5)


def build_furniture(spec, col, mats):
    for f in spec.get("furniture", []):
        parent = C.empty(f["name"], (f["center"][0], f["center"][1], 0.0), math.radians(f.get("rot", 0.0)), col)
        BUILDERS.get(f["type"], f_generic)(f, parent, furniture_material(f), mats)
        for ch in parent.children:  # objects created by bpy.ops land in the active collection
            if col not in ch.users_collection:
                C._move_to(ch, col)


# ------------------------------------------------------------------ lighting / camera
def build_lighting(spec, col):
    W, D, H = spec["width"], spec["depth"], spec["height"]
    L = spec.get("lights", {})
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(L.get("sun_elevation_deg", 35))
    sky.sun_rotation = math.radians(L.get("sun_azimuth_deg", 200))
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.35
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = L.get("sun_strength", 3.0)
    sun.angle = math.radians(1.5)
    so = C.link(bpy.data.objects.new("sun", sun), col)
    el, az = math.radians(L.get("sun_elevation_deg", 35)), math.radians(L.get("sun_azimuth_deg", 200))
    so.rotation_euler = (math.pi / 2 - el, 0, az + math.pi / 2)

    area = bpy.data.lights.new("ceiling_light", "AREA")
    area.shape = "DISK"
    area.size = 0.5
    area.energy = L.get("ceiling_light_power", 120.0)
    area.color = (1.0, 0.9, 0.78)
    ao = C.link(bpy.data.objects.new("ceiling_light", area), col)
    ao.location = (W / 2, D / 2, H - 0.06)
    fixture = C.material("fixture", (1, 1, 1), 0.3, emission=(1.0, 0.93, 0.82, 4.0))
    C.cylinder("ceiling_fixture", (W / 2, D / 2, H - 0.025), 0.26, 0.05, fixture, collection=col)


def build_cameras(spec, col):
    W, D, H = spec["width"], spec["depth"], spec["height"]
    cam = bpy.data.cameras.new("cam_corner")
    cam.lens = 14
    cam.clip_start = 0.05
    co = C.link(bpy.data.objects.new("cam_corner", cam), col)
    co.location = (W - 0.25, 0.25, H - 0.35)
    C.look_at(co, (W * 0.42, D * 0.6, 0.6))
    top = bpy.data.cameras.new("cam_top")
    top.type = "ORTHO"
    top.ortho_scale = max(W, D) * 1.12
    to = C.link(bpy.data.objects.new("cam_top", top), col)
    to.location = (W / 2, D / 2, H + 4.0)
    bpy.context.scene.camera = co


# ------------------------------------------------------------------ main
def build(spec):
    W, D, H = spec["width"], spec["depth"], spec["height"]
    M = spec.get("materials", {})
    mats = dict(
        walls=C.material("walls", M.get("walls", {}).get("color", (0.86, 0.84, 0.8)), M.get("walls", {}).get("roughness", 0.9)),
        ceiling=C.material("ceiling", M.get("ceiling", {}).get("color", (0.93, 0.93, 0.92)), 0.95),
        trim=C.material("trim", (0.95, 0.95, 0.94), 0.4),
        door=C.material("door", (0.92, 0.91, 0.88), 0.45),
        glass=C.material("glass", (0.9, 0.95, 1.0), 0.02, transmission=1.0),
        metal=C.material("metal", (0.6, 0.6, 0.62), 0.3, metallic=1.0),
        wood_dark=C.material("wood_dark", (0.12, 0.08, 0.05), 0.5),
        ceramic=C.material("ceramic", (0.88, 0.86, 0.82), 0.25),
        tv_bezel=C.material("tv_bezel", (0.02, 0.02, 0.02), 0.3),
        screen=C.material("screen", (0.01, 0.01, 0.012), 0.08),
    )
    fl = M.get("floor", {})
    if fl.get("type", "wood") == "wood":
        mats["floor"] = C.wood_material("floor", fl.get("color", (0.42, 0.28, 0.16)), fl.get("roughness", 0.45))
    else:
        mats["floor"] = C.material("floor", fl.get("color", (0.5, 0.5, 0.5)), fl.get("roughness", 0.6))

    shell = C.get_collection("room_shell")
    furn = C.get_collection("furniture")
    rig = C.get_collection("lights_cameras")
    C.box("floor", (W / 2, D / 2, -0.01), (W, D, 0.02), mats["floor"], collection=shell)
    C.box("ceiling", (W / 2, D / 2, H + 0.01), (W + 0.3, D + 0.3, 0.02), mats["ceiling"], collection=shell)
    build_walls(spec, shell, mats)
    build_openings(spec, shell, mats)
    build_furniture(spec, furn, mats)
    build_lighting(spec, rig)
    build_cameras(spec, rig)
    return mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=room_spec.DEFAULT_SPEC)
    ap.add_argument("--out", default=C.ASSETS)
    ap.add_argument("--no-glb", action="store_true")
    a = ap.parse_args(C.script_args())
    spec = room_spec.load_spec(a.spec)
    os.makedirs(a.out, exist_ok=True)
    C.clear_scene()
    build(spec)
    room_spec.write_layout(spec, os.path.join(a.out, "room_layout.json"))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.out, "room.blend"))
    if not a.no_glb:
        bpy.ops.export_scene.gltf(filepath=os.path.join(a.out, "room.glb"), export_format="GLB", export_lights=True)
    print(f"[build_room] {len(bpy.data.objects)} objects -> {a.out}")


if __name__ == "__main__":
    main()
