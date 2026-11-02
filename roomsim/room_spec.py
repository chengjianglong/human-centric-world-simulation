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


def colliders(spec):
    """Oriented 2D boxes: dict(name, cx, cy, hx, hy, theta[rad], height)."""
    out = []
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
    return dict(name=spec["name"], width=spec["width"], depth=spec["depth"], height=spec["height"],
                colliders=colliders(spec), pois=pois(spec), openings=spec.get("openings", []))


def write_layout(spec, path):
    with open(path, "w") as f:
        json.dump(layout(spec), f, indent=2)
