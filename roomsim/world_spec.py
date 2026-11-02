"""Pure-Python description of the house + garden + street world (configs/world.json).

The house interior comes from the referenced room.json. Like room_spec, this is shared
by Blender (visuals) and the torch env (physics), so the colliders match what is drawn.
"""
import json
import math
import os

from . import room_spec


def is_world(path):
    with open(path) as f:
        return "room" in json.load(f)


def load_world(path):
    with open(path) as f:
        w = json.load(f)
    room_path = os.path.join(os.path.dirname(os.path.abspath(path)), w["room"])
    w["_room"] = room_spec.load_spec(room_path)
    validate(w)
    return w


def validate(w):
    x0, y0, x1, y1 = w["bounds"]
    r = w["_room"]
    if not (x0 < -r["wall_thickness"] and y0 < -r["wall_thickness"]
            and x1 > r["width"] + r["wall_thickness"] and y1 > r["depth"] + r["wall_thickness"]):
        raise ValueError("world bounds must enclose the house")
    for f in w.get("objects", []):
        for c in room_spec.obb_corners(f):
            if not (x0 - 1e-3 <= c[0] <= x1 + 1e-3 and y0 - 1e-3 <= c[1] <= y1 + 1e-3):
                raise ValueError(f"object {f['name']} pokes outside the world bounds: corner {c}")
    for fe in w.get("fences", []):
        length = math.dist(fe["from"], fe["to"])
        for g in fe.get("gates", []):
            if g["offset"] < 0 or g["offset"] + g["width"] > length + 1e-6:
                raise ValueError(f"gate {g['name']} does not fit on fence {fe['name']}")


FENCE_T = 0.08


def fence_pieces(fe):
    """Solid runs of a fence (gates cut out): list of (name, (ax, ay), (bx, by))."""
    (ax, ay), (bx, by) = fe["from"], fe["to"]
    length = math.dist(fe["from"], fe["to"])
    ux, uy = (bx - ax) / length, (by - ay) / length
    cuts = sorted((g["offset"], g["offset"] + g["width"]) for g in fe.get("gates", []))
    out, u = [], 0.0
    for i, (a, b) in enumerate(cuts + [(length, length)]):
        if a > u + 1e-6:
            out.append((f"{fe['name']}_{i}", (ax + ux * u, ay + uy * u), (ax + ux * a, ay + uy * a)))
        u = b
    return out


def fence_colliders(w):
    out = []
    for fe in w.get("fences", []):
        for name, (ax, ay), (bx, by) in fence_pieces(fe):
            length = math.dist((ax, ay), (bx, by))
            out.append(dict(name=name, cx=(ax + bx) / 2, cy=(ay + by) / 2, hx=length / 2, hy=FENCE_T / 2,
                            theta=math.atan2(by - ay, bx - ax), height=fe.get("height", 1.8)))
    return out


def gates(w):
    """Gate openings: dict(name, x, y) centre points (also used as route landmarks / POIs)."""
    out = []
    for fe in w.get("fences", []):
        (ax, ay), (bx, by) = fe["from"], fe["to"]
        length = math.dist(fe["from"], fe["to"])
        ux, uy = (bx - ax) / length, (by - ay) / length
        for g in fe.get("gates", []):
            m = g["offset"] + g["width"] / 2
            out.append(dict(name=g["name"], x=ax + ux * m, y=ay + uy * m, width=g["width"],
                            theta=math.atan2(uy, ux), fence=fe["name"]))
    return out


def object_colliders(w):
    out = []
    for f in w.get("objects", []):
        if not f.get("collide", True):
            continue
        out.append(dict(name=f["name"], cx=f["center"][0], cy=f["center"][1], hx=f["size"][0] / 2, hy=f["size"][1] / 2,
                        theta=math.radians(f.get("rot", 0.0)), height=f["size"][2]))
    return out


def colliders(w):
    room = w["_room"]
    return room_spec.wall_colliders(room) + room_spec.colliders(room) + fence_colliders(w) + object_colliders(w)


def pois(w):
    out = list(room_spec.pois(w["_room"]))
    for f in w.get("objects", []):
        if "poi" in f:
            x, y = room_spec.local_to_world(f, *f["poi"])
            out.append(dict(name=f["name"], x=x, y=y))
    out.extend(w.get("extra_pois", []))
    return out


def layout(w):
    room = w["_room"]
    x0, y0, x1, y1 = w["bounds"]
    return dict(kind="world", name=w["name"], width=x1 - x0, depth=y1 - y0, height=room["height"], bounds=w["bounds"],
                colliders=colliders(w), pois=pois(w), openings=room.get("openings", []), gates=gates(w),
                indoor=[[0.0, 0.0, room["width"], room["depth"]]], road=w.get("road", []),
                crosswalks=w.get("crosswalks", []), sidewalks=w.get("sidewalks", []),
                nav=w.get("nav", {}), traffic=w.get("traffic"))


def write_layout(w, path):
    with open(path, "w") as f:
        json.dump(layout(w), f, indent=2)
