"""Pure-Python room description shared by Blender (visuals) and the torch env (physics).

Both sides call these functions on the same configs/room.json, so the colliders the
agents feel are exactly the furniture Blender draws.
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SPEC = os.path.join(ROOT, "configs", "room.json")


def load_spec(path=None):
    with open(path or DEFAULT_SPEC) as f:
        spec = json.load(f)
    validate(spec)
    return spec


def validate(spec):
    W, D = spec["width"], spec["depth"]
    for o in spec.get("openings", []):
        length = W if o["wall"] in ("south", "north") else D
        if o["offset"] < 0 or o["offset"] + o["width"] > length + 1e-6:
            raise ValueError(f"opening {o['name']} does not fit on wall {o['wall']}")
        if o["sill"] + o["height"] > spec["height"] + 1e-6:
            raise ValueError(f"opening {o['name']} taller than ceiling")
    for f in spec.get("furniture", []):
        for c in obb_corners(f):
            if not (-1e-3 <= c[0] <= W + 1e-3 and -1e-3 <= c[1] <= D + 1e-3):
                raise ValueError(f"furniture {f['name']} pokes outside the room: corner {c}")


def local_to_world(f, dx, dy):
    th = math.radians(f.get("rot", 0.0))
    c, s = math.cos(th), math.sin(th)
    cx, cy = f["center"]
    return cx + c * dx - s * dy, cy + s * dx + c * dy


def obb_corners(f):
    hx, hy = f["size"][0] / 2, f["size"][1] / 2
    return [local_to_world(f, sx * hx, sy * hy) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


FRAME_W = 0.05  # door/window frame width (Blender draws it; the open leaf hangs inside it)


def wall_point(spec, wall, u, n):
    """(u along the wall from its start corner, n = distance into the room from the inner face) -> world xy."""
    W, D = spec["width"], spec["depth"]
    return {"south": (u, n), "north": (u, D - n), "west": (n, u), "east": (W - n, u)}[wall]


def door_leaf(spec, o):
    """Footprint of an open door leaf (swung 90 deg in or out at its hinge) or None if the door is closed."""
    if o.get("type") != "door" or not o.get("open"):
        return None
    t, lt = spec["wall_thickness"], 0.04
    a, b = o["offset"] + FRAME_W, o["offset"] + o["width"] - FRAME_W
    length = b - a
    u = a + lt / 2 if o.get("hinge", "left") == "left" else b - lt / 2
    n = length / 2 if o["open"] == "in" else -(t + length / 2)
    cx, cy = wall_point(spec, o["wall"], u, n)
    along_x = o["wall"] in ("south", "north")
    return dict(name=f"{o['name']}_leaf", cx=cx, cy=cy, hx=lt / 2 if along_x else length / 2,
                hy=length / 2 if along_x else lt / 2, theta=0.0, height=o["height"] - FRAME_W, length=length,
                hinge_u=u, inward=o["open"] == "in")


def wall_colliders(spec):
    """The room's walls as solid boxes at walking height (door openings are gaps; windows are solid)."""
    W, D, t = spec["width"], spec["depth"], spec["wall_thickness"]
    out = []
    for wall in ("south", "north", "west", "east"):
        horiz = wall in ("south", "north")
        lo, hi = (-t, W + t) if horiz else (0.0, D)
        cuts = sorted((o["offset"], o["offset"] + o["width"]) for o in spec.get("openings", [])
                      if o["wall"] == wall and o["type"] == "door" and o["sill"] <= 0.05)
        u = lo
        for i, (a, b) in enumerate(cuts + [(hi, hi)]):
            if a > u + 1e-6:
                (x0, y0), (x1, y1) = wall_point(spec, wall, u, -t), wall_point(spec, wall, a, 0.0)
                out.append(dict(name=f"wall_{wall}_{i}", cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, hx=abs(x1 - x0) / 2,
                                hy=abs(y1 - y0) / 2, theta=0.0, height=spec["height"]))
            u = b
    return out


def colliders(spec):
    """Oriented 2D boxes: dict(name, cx, cy, hx, hy, theta[rad], height)."""
    out = [leaf for leaf in (door_leaf(spec, o) for o in spec.get("openings", [])) if leaf]
    for f in spec.get("furniture", []):
        if not f.get("collide", True):
            continue
        out.append(dict(name=f["name"], cx=f["center"][0], cy=f["center"][1],
                        hx=f["size"][0] / 2, hy=f["size"][1] / 2,
                        theta=math.radians(f.get("rot", 0.0)), height=f["size"][2]))
    return out


def opening_center_inward(spec, o, inset=0.6):
    """Point `inset` metres inside the room in front of an opening's centre."""
    W, D = spec["width"], spec["depth"]
    m = o["offset"] + o["width"] / 2
    return {"south": (m, inset), "north": (m, D - inset),
            "west": (inset, m), "east": (W - inset, m)}[o["wall"]]


def pois(spec):
    """Named goal points (front of furniture, doors, extras)."""
    out = []
    for f in spec.get("furniture", []):
        if "poi" in f:
            x, y = local_to_world(f, *f["poi"])
            out.append(dict(name=f["name"], x=x, y=y))
    for o in spec.get("openings", []):
        if o["type"] == "door":
            x, y = opening_center_inward(spec, o)
            out.append(dict(name=o["name"], x=x, y=y))
    out.extend(spec.get("extra_pois", []))
    return out


def layout(spec):
    return dict(kind="room", name=spec["name"], width=spec["width"], depth=spec["depth"], height=spec["height"],
                bounds=[0.0, 0.0, spec["width"], spec["depth"]],
                colliders=colliders(spec), pois=pois(spec), openings=spec.get("openings", []))


def write_layout(spec, path):
    with open(path, "w") as f:
        json.dump(layout(spec), f, indent=2)
