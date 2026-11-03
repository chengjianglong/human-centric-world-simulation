"""Build the house + back garden + front street world from configs/world.json.

    tools/blender/blender -b -P blender/build_world.py -- [--spec configs/world.json] [--out assets]

The house interior is the room from build_room.py (same room.json); this adds the exterior
(siding, gable roof), the lot (lawn, patio, garden beds, trees, shed, fences, gate arbour),
the front yard, sidewalks, a two-lane street with markings and a zebra crossing, a bus stop,
street lamps and neighbouring houses as backdrop. Writes assets/world.blend, world.glb and
world_layout.json (the colliders/POIs/road/traffic used by the PyTorch env).
"""
import argparse
import math
import os
import random
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_room as R  # noqa: E402
import common as C  # noqa: E402
from worldsim import world_spec  # noqa: E402


# ------------------------------------------------------------------ helpers
def mesh(name, verts, faces, mat=None, collection=None, parent=None, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    o = C.link(bpy.data.objects.new(name, me), collection)
    if mat is not None:
        o.data.materials.append(mat)
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    if parent is not None:
        o.parent = parent
    return o


def gable_roof(name, x0, x1, y0, y1, z, rise, thick, mat, col, parent=None):
    """Gable roof with the ridge along x: two sloped slabs + triangular gable ends (wall material)."""
    ym = (y0 + y1) / 2
    zr = z + rise
    v = [(x0, y0, z), (x1, y0, z), (x1, ym, zr), (x0, ym, zr),           # south slope (top)
         (x0, y1, z), (x1, y1, z), (x1, ym, zr), (x0, ym, zr)]           # north slope (top)
    vb = [(x, y, zz - thick) for x, y, zz in v]
    verts = v + vb
    faces = [(0, 1, 2, 3), (5, 4, 7, 6), (8 + 3, 8 + 2, 8 + 1, 8 + 0), (8 + 6, 8 + 7, 8 + 4, 8 + 5),
             (0, 8, 9, 1), (4, 5, 13, 12), (0, 3, 11, 8), (1, 9, 10, 2), (4, 12, 15, 7), (5, 6, 14, 13)]
    return mesh(name, verts, faces, mat, col, parent)


def gable_ends(name, x_list, y0, y1, z, rise, mat, col):
    ym = (y0 + y1) / 2
    for i, x in enumerate(x_list):
        mesh(f"{name}_{i}", [(x, y0, z), (x, y1, z), (x, ym, z + rise)], [(0, 1, 2)], mat, col)


def grass_material(name, color):
    m = C.material(name, color, 0.95)
    C._noise_color(m, color, scale=1.5, detail=12.0, var=0.35)
    return m


def stone_material(name, color, roughness=0.85, scale=3.0):
    m = C.material(name, color, roughness)
    C._noise_color(m, color, scale=scale, detail=8.0, var=0.15)
    return m


def wood(name, color, roughness=0.7):
    m = C.material(name, color, roughness)
    C._noise_color(m, color, scale=6.0, detail=8.0, var=0.2, stretch=(1, 12, 1))
    return m


# ------------------------------------------------------------------ house exterior
def build_exterior(room, col, mats):
    W, D, H, t = room["width"], room["depth"], room["height"], room["wall_thickness"]
    sk = 0.012
    for wall in ("south", "north", "west", "east"):
        for i, (u0, u1, z0, z1) in enumerate(R.wall_segments(room, wall)):
            if wall in ("west", "east"):
                u0, u1 = (u0 - t - sk if u0 <= 0 else u0), (u1 + t + sk if u1 >= D else u1)
            else:
                u0, u1 = (u0 - sk if u0 <= -t else u0), (u1 + sk if u1 >= W + t else u1)
            c, s = R.wall_box(room, wall, u0, u1, z0, z1, inset=-(t + sk), thick=sk)
            C.box(f"siding_{wall}_{i}", c, s, mats["siding"], collection=col)
    # corner boards and a foundation band
    for x, y in ((-t, -t), (W + t, -t), (-t, D + t), (W + t, D + t)):
        C.box("corner_board", (x, y, H / 2), (0.16, 0.16, H), mats["trim"], collection=col)
    for name, c, s in (("found_s", (W / 2, -t - 0.02, 0.1), (W + 2 * t + 0.08, 0.05, 0.2)),
                       ("found_n", (W / 2, D + t + 0.02, 0.1), (W + 2 * t + 0.08, 0.05, 0.2)),
                       ("found_w", (-t - 0.02, D / 2, 0.1), (0.05, D + 2 * t, 0.2)),
                       ("found_e", (W + t + 0.02, D / 2, 0.1), (0.05, D + 2 * t, 0.2))):
        C.box(name, c, s, mats["foundation"], collection=col)
    # roof: gable along x with overhangs, fascia boards; gable triangles above the walls
    ov, th, pitch = 0.4, 0.1, math.tan(math.radians(28))
    rise = (D / 2 + t + ov) * pitch
    roof = C.get_collection("roof")
    # eave height chosen so the roof's underside meets the top of the walls
    gable_roof("roof", -t - ov, W + t + ov, -t - ov, D + t + ov, H + th - ov * pitch, rise, th, mats["roof"], roof)
    gable_ends("gable", [-t - 0.005, W + t + 0.005], -t, D + t, H, (D / 2 + t) * pitch, mats["siding"], roof)
    for name, c, s in (("fascia_s", (W / 2, -t - ov, H - 0.06), (W + 2 * t + 2 * ov, 0.04, 0.2)),
                       ("fascia_n", (W / 2, D + t + ov, H - 0.06), (W + 2 * t + 2 * ov, 0.04, 0.2))):
        C.box(name, c, s, mats["trim"], collection=roof)
    # porch light over the front door
    for o in room.get("openings", []):
        if o["type"] == "door" and o["wall"] == "south":
            x = o["offset"] + o["width"] + 0.2
            lamp = C.material("porch_lamp", (1, 0.9, 0.7), 0.3, emission=(1.0, 0.85, 0.6, 3.0))
            C.box("porch_light", (x, -t - 0.06, 2.0), (0.12, 0.1, 0.22), lamp, collection=col)
            C.box("porch_canopy", (o["offset"] + o["width"] / 2, -t - 0.45, 2.35), (o["width"] + 0.8, 0.9, 0.06), mats["trim"], collection=col)


# ------------------------------------------------------------------ outdoor objects (local frame: front = -y)
def o_patio(f, p, M):
    w, d, h = f["size"]
    rng = random.Random(3)
    n_x, n_y = max(1, int(w / 0.6)), max(1, int(d / 0.6))
    for i in range(n_x):
        for j in range(n_y):
            col = [c * rng.uniform(0.88, 1.08) for c in f["color"]]
            m = C.material(f"paver_{i % 3}{j % 2}", col, 0.85)
            C.box(f"{f['name']}_paver", (-w / 2 + (i + 0.5) * w / n_x, -d / 2 + (j + 0.5) * d / n_y, h / 2),
                  (w / n_x - 0.02, d / n_y - 0.02, h), m, p)


def o_flat(f, p, M):
    w, d, h = f["size"]
    C.box(f["name"], (0, 0, h / 2), (w, d, h), stone_material(f"mat_{f['name']}", f["color"]), p)


def o_patio_table(f, p, M):
    w, d, h = f["size"]
    top = C.material("patio_table_top", f["color"], 0.4)
    C.cylinder(f"{f['name']}_top", (0, 0, h - 0.02), w / 2, 0.04, top, p)
    C.cylinder(f"{f['name']}_leg", (0, 0, (h - 0.04) / 2), 0.04, h - 0.04, M["metal"], p, verts=12)
    C.cylinder(f"{f['name']}_foot", (0, 0, 0.02), 0.3, 0.04, M["metal"], p)
    C.cylinder(f"{f['name']}_pole", (0, 0, 1.3), 0.02, 1.4, M["metal"], p, verts=8)
    canopy = C.material("umbrella", (0.75, 0.2, 0.15), 0.8)
    mesh(f"{f['name']}_umbrella_cone", [(1.2 * math.cos(a * math.pi / 4), 1.2 * math.sin(a * math.pi / 4), 1.95) for a in range(8)] + [(0, 0, 2.3)],
         [(i, (i + 1) % 8, 8) for i in range(8)], canopy, parent=p, collection=p.users_collection[0])


def o_bbq(f, p, M):
    w, d, h = f["size"]
    C.ellipsoid(f"{f['name']}_kettle", (0, 0, h - 0.25), (w * 0.38, d * 0.48, 0.26), C.material("bbq_black", f["color"], 0.4, metallic=0.6), p)
    for i in range(3):
        a = i * 2 * math.pi / 3
        C.cylinder(f"{f['name']}_leg{i}", (0.18 * math.cos(a), 0.18 * math.sin(a), (h - 0.4) / 2), 0.015, h - 0.4, M["metal"], p, verts=8)


def o_tree(f, p, M):
    w, d, h = f["size"]
    bark = C.material("bark", (0.22, 0.15, 0.1), 0.95)
    C._noise_color(bark, (0.22, 0.15, 0.1), scale=8.0, detail=10.0, var=0.3, stretch=(1, 1, 6))
    trunk_h = h * 0.45
    C.cylinder(f"{f['name']}_trunk", (0, 0, trunk_h / 2), w * 0.42, trunk_h, bark, p, verts=16)
    leaf = grass_material(f"foliage_{f['name']}", f["color"])
    rng = random.Random(f["name"])
    rc = f.get("canopy", 2.0)
    C.ellipsoid(f"{f['name']}_canopy0", (0, 0, trunk_h + rc * 0.8), (rc, rc, rc * 0.85), leaf, p, segments=20)
    for i in range(6):
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.4, 0.75) * rc
        z = trunk_h + rc * rng.uniform(0.4, 1.3)
        s = rc * rng.uniform(0.45, 0.65)
        C.ellipsoid(f"{f['name']}_canopy{i + 1}", (rr * math.cos(a), rr * math.sin(a), z), (s, s, s * 0.85), leaf, p, segments=16)


def o_shrub(f, p, M):
    w, d, h = f["size"]
    leaf = grass_material(f"foliage_{f['name']}", f["color"])
    rng = random.Random(f["name"])
    C.ellipsoid(f"{f['name']}_main", (0, 0, h * 0.5), (w * 0.48, d * 0.48, h * 0.52), leaf, p, segments=16)
    for i in range(3):
        C.ellipsoid(f"{f['name']}_lobe{i}", (rng.uniform(-w, w) * 0.2, rng.uniform(-d, d) * 0.2, h * rng.uniform(0.55, 0.75)),
                    (w * 0.3, d * 0.3, h * 0.3), leaf, p, segments=12)


def o_garden_bed(f, p, M):
    w, d, h = f["size"]
    frame = wood(f"bed_wood_{f['name']}", f["color"])
    t = 0.05
    for sy in (-1, 1):
        C.box(f"{f['name']}_side", (0, sy * (d / 2 - t / 2), h / 2), (w, t, h), frame, p)
    for sx in (-1, 1):
        C.box(f"{f['name']}_end", (sx * (w / 2 - t / 2), 0, h / 2), (t, d - 2 * t, h), frame, p)
    C.box(f"{f['name']}_soil", (0, 0, h - 0.06), (w - 2 * t, d - 2 * t, 0.04), C.material("soil", (0.12, 0.08, 0.05), 1.0), p)
    rng = random.Random(f["name"])
    green = C.material("veg_leaf", (0.18, 0.42, 0.12), 0.6)
    petals = [C.material(f"petal_{i}", c, 0.5) for i, c in enumerate([(0.85, 0.1, 0.15), (0.95, 0.75, 0.1), (0.6, 0.2, 0.7), (0.95, 0.95, 0.95)])]
    for i in range(int(w / 0.3)):
        for j in range(int(d / 0.35)):
            x = -w / 2 + 0.2 + i * 0.3 + rng.uniform(-0.04, 0.04)
            y = -d / 2 + 0.22 + j * 0.35 + rng.uniform(-0.04, 0.04)
            if f.get("flowers"):
                C.cylinder(f"{f['name']}_stem", (x, y, h + 0.12), 0.008, 0.26, green, p, verts=6)
                C.ellipsoid(f"{f['name']}_bloom", (x, y, h + 0.27), (0.06, 0.06, 0.04), petals[rng.randrange(4)], p, segments=8)
            else:
                C.ellipsoid(f"{f['name']}_veg", (x, y, h + 0.07), (0.12, 0.12, 0.09), green, p, segments=10)


def o_bench(f, p, M):
    w, d, h = f["size"]
    slat = wood(f"bench_wood_{f['name']}", f["color"])
    for i in range(4):
        C.box(f"{f['name']}_seat{i}", (0, -d / 2 + 0.06 + i * 0.11, 0.45), (w, 0.09, 0.03), slat, p)
    for i in range(3):
        C.box(f"{f['name']}_back{i}", (0, d / 2 - 0.08, 0.58 + i * 0.1), (w, 0.03, 0.08), slat, p)
    for sx in (-1, 1):
        C.box(f"{f['name']}_frame", (sx * (w / 2 - 0.08), 0, 0.22), (0.05, d - 0.05, 0.44), M["metal"], p)
        C.box(f"{f['name']}_arm", (sx * (w / 2 - 0.06), -0.02, 0.66), (0.06, d * 0.8, 0.04), slat, p)


def o_shed(f, p, M):
    w, d, h = f["size"]
    body = wood(f"shed_{f['name']}", f["color"])
    wall_h = h * 0.72
    C.box(f"{f['name']}_body", (0, 0, wall_h / 2), (w, d, wall_h), body, p)
    roof = C.material("shed_roof", (0.18, 0.16, 0.15), 0.85)
    o = gable_roof(f"{f['name']}_roof", -w / 2 - 0.12, w / 2 + 0.12, -d / 2 - 0.15, d / 2 + 0.15, wall_h, h - wall_h, 0.05, roof, p.users_collection[0])
    o.parent = p
    for i, x in enumerate((-w / 2, w / 2)):
        g = mesh(f"{f['name']}_gable{i}", [(x, -d / 2, wall_h), (x, d / 2, wall_h), (x, 0, h - 0.02)], [(0, 1, 2)], body, p.users_collection[0])
        g.parent = p
    C.box(f"{f['name']}_door", (0, -d / 2 - 0.01, 0.95), (0.9, 0.03, 1.9), C.material("shed_door", (0.85, 0.85, 0.8), 0.6), p)
    C.box(f"{f['name']}_window", (w / 2 - 0.5, -d / 2 - 0.012, 1.3), (0.5, 0.02, 0.4), M["glass"], p)


def o_bins(f, p, M):
    w, d, h = f["size"]
    cols = [f["color"], (0.1, 0.12, 0.3)]
    for i, sy in enumerate((-1, 1)):
        m = C.material(f"bin_{i}", cols[i], 0.6)
        C.box(f"{f['name']}_bin{i}", (0, sy * d / 4, (h - 0.06) / 2), (w * 0.9, d / 2 - 0.06, h - 0.06), m, p, bevel=0.02)
        C.box(f"{f['name']}_lid{i}", (0, sy * d / 4, h - 0.03), (w, d / 2 - 0.03, 0.05), m, p, bevel=0.01)


def o_mailbox(f, p, M):
    w, d, h = f["size"]
    C.box(f"{f['name']}_post", (0, 0, (h - 0.25) / 2), (0.09, 0.09, h - 0.25), wood("mail_post", (0.4, 0.3, 0.2)), p)
    C.box(f"{f['name']}_box", (0, 0, h - 0.12), (w, d, 0.24), C.material("mailbox", f["color"], 0.4, metallic=0.5), p, bevel=0.03)
    C.box(f"{f['name']}_flag", (w / 2 + 0.01, 0.05, h - 0.02), (0.01, 0.04, 0.16), C.material("mail_flag", (0.8, 0.05, 0.05), 0.5), p)


def o_street_lamp(f, p, M):
    w, d, h = f["size"]
    pole = C.material("lamp_pole", f["color"], 0.4, metallic=0.7)
    C.cylinder(f"{f['name']}_pole", (0, 0, h / 2), w * 0.3, h, pole, p, verts=12)
    C.box(f"{f['name']}_arm", (0, -0.6, h - 0.05), (0.06, 1.2, 0.06), pole, p)
    head = C.material("lamp_head", (1.0, 0.95, 0.85), 0.3, emission=(1.0, 0.9, 0.75, 1.5))
    C.box(f"{f['name']}_head", (0, -1.15, h - 0.12), (0.22, 0.45, 0.1), head, p)


def o_bus_stop(f, p, M):
    w, d, h = f["size"]
    frame = C.material("shelter_frame", f["color"], 0.35, metallic=0.8)
    for sx in (-1, 1):
        for sy in (-1, 1):
            C.box(f"{f['name']}_post", (sx * (w / 2 - 0.04), sy * (d / 2 - 0.04), h / 2), (0.07, 0.07, h), frame, p)
    C.box(f"{f['name']}_roof", (0, 0, h + 0.03), (w + 0.2, d + 0.25, 0.06), frame, p)
    C.box(f"{f['name']}_back", (0, d / 2 - 0.04, h / 2 + 0.1), (w - 0.1, 0.02, h - 0.4), M["glass"], p)
    for sx in (-1, 1):
        C.box(f"{f['name']}_side", (sx * (w / 2 - 0.04), 0.1, h / 2 + 0.1), (0.02, d - 0.3, h - 0.4), M["glass"], p)
    C.box(f"{f['name']}_bench", (0, d / 2 - 0.3, 0.45), (w * 0.7, 0.35, 0.05), frame, p)
    ad = C.material("bus_ad", (0.9, 0.55, 0.1), 0.5, emission=(0.9, 0.6, 0.2, 0.6))
    C.box(f"{f['name']}_ad", (w / 2 - 0.05, 0.1, 1.3), (0.03, 0.9, 1.4), ad, p)
    C.cylinder(f"{f['name']}_signpole", (-w / 2 - 0.4, -d / 2, 1.3), 0.03, 2.6, frame, p, verts=8)
    C.box(f"{f['name']}_sign", (-w / 2 - 0.4, -d / 2, 2.5), (0.04, 0.45, 0.35), C.material("bus_sign", (0.05, 0.3, 0.7), 0.4), p)


def o_generic(f, p, M):
    w, d, h = f["size"]
    C.box(f["name"], (0, 0, h / 2), (w, d, h), C.material(f"mat_{f['name']}", f["color"], 0.5), p, bevel=0.02)


BUILDERS = dict(patio=o_patio, path=o_flat, step=o_flat, patio_table=o_patio_table, bbq=o_bbq, tree=o_tree, shrub=o_shrub,
                garden_bed=o_garden_bed, bench=o_bench, shed=o_shed, bins=o_bins, mailbox=o_mailbox,
                street_lamp=o_street_lamp, bus_stop=o_bus_stop, generic=o_generic)


def build_objects(w, col, M):
    for f in w.get("objects", []):
        parent = C.empty(f["name"], (f["center"][0], f["center"][1], 0.0), math.radians(f.get("rot", 0.0)), col)
        BUILDERS.get(f["type"], o_generic)(f, parent, M)
        for ch in parent.children:
            if col not in ch.users_collection:
                C._move_to(ch, col)


# ------------------------------------------------------------------ lot, street, fences
def build_ground(w, col, M):
    C.box("lawn", (5, -2, -0.02), (130, 90, 0.02), M["grass"], collection=col)
    for i, (x0, y0, x1, y1) in enumerate(w.get("sidewalks", [])):
        C.box(f"sidewalk_{i}", ((x0 + x1) / 2, (y0 + y1) / 2, 0.004), (x1 - x0, y1 - y0, 0.012), M["concrete"], collection=col)
        # expansion joints every 1.5 m
        for k in range(int((x1 - x0) / 1.5)):
            x = x0 + 1.5 * (k + 1)
            C.box(f"sidewalk_joint_{i}", (x, (y0 + y1) / 2, 0.0105), (0.015, y1 - y0, 0.002), M["joint"], collection=col)
    tr = w.get("traffic", {})
    xr = tr.get("x_range", [-45, 55])
    for i, (x0, y0, x1, y1) in enumerate(w.get("road", [])):
        y0r, y1r = y0, y1
        C.box(f"road_{i}", ((xr[0] + xr[1]) / 2, (y0r + y1r) / 2, -0.002), (xr[1] - xr[0], y1r - y0r, 0.012), M["asphalt"], collection=col)
        for y in (y0r, y1r):
            C.box(f"curb_{i}", ((xr[0] + xr[1]) / 2, y, 0.02), (xr[1] - xr[0], 0.16, 0.05), M["curb"], collection=col)
        # markings: dashed yellow centre line, white edge lines
        ym = (y0r + y1r) / 2
        cw = w.get("crosswalks", [])
        x = xr[0]
        while x < xr[1]:
            if not any(c[0] - 1.0 < x + 1.5 < c[2] + 1.0 for c in cw):
                C.box(f"lane_dash_{i}", (x + 1.5, ym, 0.0045), (3.0, 0.12, 0.002), M["paint_yellow"], collection=col)
            x += 7.0
        for y in (y0r + 0.35, y1r - 0.35):
            C.box(f"edge_line_{i}", ((xr[0] + xr[1]) / 2, y, 0.0045), (xr[1] - xr[0], 0.1, 0.002), M["paint_white"], collection=col)
    for i, (x0, y0, x1, y1) in enumerate(w.get("crosswalks", [])):
        y = y0 + 0.5
        while y < y1 - 0.3:
            C.box(f"zebra_{i}", ((x0 + x1) / 2, y + 0.25, 0.006), (x1 - x0, 0.5, 0.002), M["paint_white"], collection=col)
            y += 1.0
        for yy in (y0 - 0.9, y1 + 0.9):  # tactile paving at both kerbs
            C.box(f"tactile_{i}", ((x0 + x1) / 2, yy if yy < y0 else yy, 0.012), (x1 - x0, 0.6, 0.004), M["tactile"], collection=col)


def build_fences(w, col, M):
    panel = wood("fence_wood", (0.48, 0.34, 0.22))
    post = wood("fence_post", (0.36, 0.25, 0.16))
    for fe in w.get("fences", []):
        h = fe.get("height", 1.8)
        for name, (ax, ay), (bx, by) in world_spec.fence_pieces(fe):
            length = math.dist((ax, ay), (bx, by))
            th = math.atan2(by - ay, bx - ax)
            C.box(name, ((ax + bx) / 2, (ay + by) / 2, h / 2 + 0.03), (length, 0.035, h - 0.06), panel, rot_z=th, collection=col)
            n = max(1, int(round(length / 2.0)))
            for k in range(n + 1):
                u = k / n
                C.box(f"{name}_post", (ax + (bx - ax) * u, ay + (by - ay) * u, (h + 0.08) / 2), (0.09, 0.09, h + 0.08), post, rot_z=th, collection=col)
    # gate arbours: two posts and a crossbeam over each gate
    arbour = wood("arbour_wood", (0.85, 0.83, 0.78), 0.6)
    for g in world_spec.gates(w):
        ux, uy = math.cos(g["theta"]), math.sin(g["theta"])
        for s in (-1, 1):
            x, y = g["x"] + s * ux * (g["width"] / 2 + 0.05), g["y"] + s * uy * (g["width"] / 2 + 0.05)
            C.box(f"{g['name']}_post", (x, y, 1.15), (0.1, 0.1, 2.3), arbour, rot_z=g["theta"], collection=col)
        C.box(f"{g['name']}_beam", (g["x"], g["y"], 2.33), (g["width"] + 0.5, 0.5, 0.08), arbour, rot_z=g["theta"], collection=col)


def build_backdrop(w, col, M):
    rng = random.Random(11)
    for b in w.get("backdrop", []):
        (cx, cy), (bw, bd, bh) = b["center"], b["size"]
        m = C.material(f"facade_{b['name']}", b["color"], 0.85)
        C.box(b["name"], (cx, cy, bh / 2), (bw, bd, bh), m, collection=col)
        gable_roof(f"{b['name']}_roof", cx - bw / 2 - 0.3, cx + bw / 2 + 0.3, cy - bd / 2 - 0.3, cy + bd / 2 + 0.3, bh, 2.0, 0.12,
                   M["roof"], col)
        gable_ends(f"{b['name']}_gable", [cx - bw / 2, cx + bw / 2], cy - bd / 2, cy + bd / 2, bh, 2.0 * bd / (bd + 0.6), m, col)
        face_y = cy + bd / 2 + 0.01 if cy < 0 else cy - bd / 2 - 0.01
        face_x = cx + bw / 2 + 0.01 if cx < -4 else cx - bw / 2 - 0.01
        lit = C.material("window_lit", (0.7, 0.75, 0.8), 0.05, emission=(0.9, 0.8, 0.6, 0.15))
        if abs(cy) > 15:  # across the street: facade faces +y
            for k in range(int(bw / 2.2)):
                x = cx - bw / 2 + 1.2 + k * 2.2
                for z in (1.4, 3.9) if bh > 4.5 else (1.4,):
                    C.box(f"{b['name']}_win", (x, face_y, z), (1.1, 0.04, 1.2), lit, collection=col)
            C.box(f"{b['name']}_door", (cx, face_y, 1.05), (1.0, 0.05, 2.1), C.material("door_red", (0.45, 0.08, 0.06), 0.5), collection=col)
        else:  # beside the lot: facade faces the lot
            for k in range(int(bd / 2.4)):
                y = cy - bd / 2 + 1.3 + k * 2.4
                C.box(f"{b['name']}_win", (face_x, y, 1.5), (0.04, 1.1, 1.2), lit, collection=col)
    leaf = grass_material("foliage_backdrop", (0.17, 0.33, 0.11))
    bark = C.material("bark", (0.22, 0.15, 0.1), 0.95)
    for k in range(14):  # street trees along the far side, outside the walkable world
        x = -40 + k * 7.0 + rng.uniform(-1, 1)
        if -6 < x < 12:
            continue
        for y in (-16.4, -3.0):
            if y == -3.0 and -14 < x < 22:
                continue
            C.cylinder("bg_trunk", (x, y, 1.4), 0.18, 2.8, bark, collection=col, verts=10)
            r = rng.uniform(1.6, 2.3)
            C.ellipsoid("bg_canopy", (x, y, 2.8 + r * 0.7), (r, r, r * 0.9), leaf, collection=col, segments=14)


# ------------------------------------------------------------------ cameras / lighting
def build_cameras(w, col):
    x0, y0, x1, y1 = w["bounds"]

    def cam(name, loc, target, lens):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.clip_start = 0.05
        cd.clip_end = 400
        o = C.link(bpy.data.objects.new(name, cd), col)
        o.location = loc
        C.look_at(o, target)
        return o

    cam("cam_aerial", (17.0, -27.0, 15.0), (3.0, -2.0, 0.0), 24)
    cam("cam_cutaway", (-8.5, -11.5, 13.5), (3.0, 1.2, 0.3), 22)
    cam("cam_street", (9.6, -15.1, 1.7), (1.8, -2.5, 1.3), 22)
    cam("cam_garden", (5.6, 11.7, 2.6), (1.8, 5.0, 0.9), 16)
    cd = bpy.data.cameras.new("cam_world_top")
    cd.type = "ORTHO"
    cd.ortho_scale = max(x1 - x0, y1 - y0) * 1.04
    cd.clip_end = 200
    o = C.link(bpy.data.objects.new("cam_world_top", cd), col)
    o.location = ((x0 + x1) / 2, (y0 + y1) / 2, 40.0)
    bpy.context.scene.camera = bpy.data.objects["cam_aerial"]


def tune_lighting(w):
    L = w.get("lights", {})
    world = bpy.context.scene.world
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = L.get("sky_strength", 0.8)
    sky = next(n for n in world.node_tree.nodes if n.type == "TEX_SKY")
    sky.sun_elevation = math.radians(L.get("sun_elevation_deg", 38))
    sky.sun_rotation = math.radians(L.get("sun_azimuth_deg", 215))
    sun = bpy.data.objects["sun"]
    sun.data.energy = L.get("sun_strength", 4.0)
    el, az = math.radians(L.get("sun_elevation_deg", 38)), math.radians(L.get("sun_azimuth_deg", 215))
    sun.rotation_euler = (math.pi / 2 - el, 0, az + math.pi / 2)
    if "ceiling_light_power" in L:
        bpy.data.objects["ceiling_light"].data.energy = L["ceiling_light_power"]


# ------------------------------------------------------------------ main
def build(w):
    room = w["_room"]
    room_mats = R.build(room)                       # interior, ceiling light, sun + sky, room cameras
    MM = w.get("materials", {})
    M = dict(room_mats)
    M.update(
        siding=C.material("siding", MM.get("siding", {}).get("color", (0.74, 0.72, 0.66)), 0.8),
        roof=stone_material("roof_shingles", MM.get("roof", {}).get("color", (0.22, 0.2, 0.2)), 0.9, scale=12.0),
        foundation=stone_material("foundation", (0.45, 0.44, 0.42)),
        grass=grass_material("grass", MM.get("grass", {}).get("color", (0.16, 0.3, 0.09))),
        asphalt=stone_material("asphalt", MM.get("asphalt", {}).get("color", (0.07, 0.07, 0.075)), 0.75, scale=40.0),
        concrete=stone_material("concrete", MM.get("concrete", {}).get("color", (0.55, 0.54, 0.51)), 0.8, scale=8.0),
        curb=stone_material("curb", (0.62, 0.61, 0.58)),
        joint=C.material("joint", (0.3, 0.3, 0.29), 0.9),
        paint_white=C.material("paint_white", (0.92, 0.92, 0.9), 0.5),
        paint_yellow=C.material("paint_yellow", (0.9, 0.7, 0.1), 0.5),
        tactile=C.material("tactile", (0.75, 0.6, 0.1), 0.7),
    )
    ext = C.get_collection("house_exterior")
    lot = C.get_collection("lot")
    street = C.get_collection("street")
    objs = C.get_collection("outdoor_objects")
    bg = C.get_collection("backdrop")
    build_exterior(room, ext, M)
    build_ground(w, street, M)
    build_fences(w, lot, M)
    build_objects(w, objs, M)
    build_backdrop(w, bg, M)
    build_cameras(w, C.get_collection("lights_cameras"))
    tune_lighting(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(C.ROOT, "configs", "world.json"))
    ap.add_argument("--out", default=C.ASSETS)
    ap.add_argument("--no-glb", action="store_true")
    a = ap.parse_args(C.script_args())
    w = world_spec.load_world(a.spec)
    os.makedirs(a.out, exist_ok=True)
    C.clear_scene()
    build(w)
    world_spec.write_layout(w, os.path.join(a.out, "world_layout.json"))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.out, "world.blend"))
    if not a.no_glb:
        bpy.ops.export_scene.gltf(filepath=os.path.join(a.out, "world.glb"), export_format="GLB", export_lights=True)
    print(f"[build_world] {len(bpy.data.objects)} objects -> {a.out}")


if __name__ == "__main__":
    main()
