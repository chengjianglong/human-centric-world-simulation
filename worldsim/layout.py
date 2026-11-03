"""Room / world layout as torch tensors (built from the same JSON specs Blender uses)."""
from dataclasses import dataclass, field

import torch

from . import room_spec, world_spec


@dataclass
class TorchLayout:
    width: float
    depth: float
    box_c: torch.Tensor   # [K, 2] collider centres
    box_h: torch.Tensor   # [K, 2] half extents in box frame
    box_cs: torch.Tensor  # [K, 2] (cos, sin) of box rotation
    pois: torch.Tensor    # [P, 2] named goal points
    poi_names: list
    box_names: list
    kind: str = "room"
    bounds: tuple = None          # (xmin, ymin, xmax, ymax) of the walkable area
    road: torch.Tensor = None     # [R, 4] rectangles
    crosswalks: torch.Tensor = None
    indoor: torch.Tensor = None
    traffic: dict = None
    nav_cfg: dict = field(default_factory=dict)
    raw: dict = None
    _nav: object = None

    def nav(self, radius=0.25):
        """Geodesic route planner (built on first use; worlds only)."""
        if self.kind != "world":
            return None
        if self._nav is None:
            from .nav import NavGrid
            self._nav = NavGrid(self, radius, self.nav_cfg.get("cell", 0.25), self.nav_cfg.get("road_cost", 4.0))
        return self._nav

    @property
    def num_boxes(self):
        return self.box_c.shape[0]


def load_layout(spec_path=None, device="cpu"):
    path = spec_path or room_spec.DEFAULT_SPEC
    L = world_spec.layout(world_spec.load_world(path)) if world_spec.is_world(path) else room_spec.layout(room_spec.load_spec(path))
    cols = L["colliders"]
    f = dict(device=device, dtype=torch.float32)
    th = torch.tensor([c["theta"] for c in cols], **f)
    rects = lambda k: torch.tensor(L.get(k, []), **f).reshape(-1, 4)  # noqa: E731
    return TorchLayout(
        kind=L["kind"], bounds=tuple(float(b) for b in L["bounds"]), road=rects("road"), crosswalks=rects("crosswalks"),
        indoor=rects("indoor"), traffic=L.get("traffic"), nav_cfg=L.get("nav") or {}, raw=L,
        width=float(L["width"]), depth=float(L["depth"]),
        box_c=torch.tensor([[c["cx"], c["cy"]] for c in cols], **f).reshape(-1, 2),
        box_h=torch.tensor([[c["hx"], c["hy"]] for c in cols], **f).reshape(-1, 2),
        box_cs=torch.stack([th.cos(), th.sin()], -1).reshape(-1, 2),
        pois=torch.tensor([[p["x"], p["y"]] for p in L["pois"]], **f).reshape(-1, 2),
        poi_names=[p["name"] for p in L["pois"]], box_names=[c["name"] for c in cols],
    )


def to_box_frame(p, layout):
    """p [..., 2] world points -> [..., K, 2] coordinates in each box frame."""
    d = p.unsqueeze(-2) - layout.box_c
    c, s = layout.box_cs[:, 0], layout.box_cs[:, 1]
    return torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def rotate_to_box(v, layout):
    """Direction vectors v [..., 2] -> [..., K, 2] in each box frame."""
    c, s = layout.box_cs[:, 0], layout.box_cs[:, 1]
    vx, vy = v[..., 0:1], v[..., 1:2]
    return torch.stack([c * vx + s * vy, -s * vx + c * vy], -1)


def box_signed_distance(p, layout):
    """Signed distance from points [..., 2] to every box -> [..., K] (negative inside)."""
    q = to_box_frame(p, layout).abs() - layout.box_h
    outside = q.clamp(min=0).norm(dim=-1)
    inside = q.max(dim=-1).values.clamp(max=0)
    return outside + inside


def push_out_of_boxes(p, radius, layout):
    """Resolve disc-vs-OBB penetration. p [..., 2] -> (new p, collided bool [...])."""
    if layout.num_boxes == 0:
        return p, torch.zeros(p.shape[:-1], dtype=torch.bool, device=p.device)
    q = to_box_frame(p, layout)                        # [..., K, 2]
    h = layout.box_h
    closest = torch.maximum(torch.minimum(q, h), -h)
    d = q - closest
    dist = d.norm(dim=-1)                              # [..., K]
    inside = dist < 1e-6
    # inside a box: push out through the nearest face
    pen_axis = h - q.abs()                             # distance to each face pair
    use_x = pen_axis[..., 0] < pen_axis[..., 1]
    sgn = torch.where(q >= 0, 1.0, -1.0)  # sign() would be 0 exactly on an axis
    zero = torch.zeros_like(q[..., 0])
    face_n = torch.where(use_x.unsqueeze(-1), torch.stack([sgn[..., 0], zero], -1), torch.stack([zero, sgn[..., 1]], -1))
    n = torch.where(inside.unsqueeze(-1), face_n, d / dist.clamp(min=1e-6).unsqueeze(-1))
    depth = torch.where(inside, torch.minimum(pen_axis[..., 0], pen_axis[..., 1]) + radius, radius - dist)
    hit = depth > 0
    push_local = n * depth.clamp(min=0).unsqueeze(-1) * hit.unsqueeze(-1)
    c, s = layout.box_cs[:, 0], layout.box_cs[:, 1]
    push_world = torch.stack([c * push_local[..., 0] - s * push_local[..., 1],
                              s * push_local[..., 0] + c * push_local[..., 1]], -1)
    return p + push_world.sum(-2), hit.any(-1)


def in_rects(p, rects):
    """p [..., 2] inside any of the axis-aligned rectangles [R, 4] (xmin, ymin, xmax, ymax) -> bool [...]."""
    if rects is None or rects.shape[0] == 0:
        return torch.zeros(p.shape[:-1], dtype=torch.bool, device=p.device)
    x, y = p[..., 0:1], p[..., 1:2]
    return ((x >= rects[:, 0]) & (x <= rects[:, 2]) & (y >= rects[:, 1]) & (y <= rects[:, 3])).any(-1)


def push_out_of_aabbs(p, radius, c, h):
    """Discs p [B, N, 2] vs per-world axis-aligned boxes (centres c [B, C, 2], half sizes h [B, C, 2]).
    Returns (new p, hit [B, N])."""
    if c.shape[1] == 0:
        return p, torch.zeros(p.shape[:-1], dtype=torch.bool, device=p.device)
    q = p.unsqueeze(2) - c.unsqueeze(1)                                  # [B, N, C, 2]
    hh = h.unsqueeze(1)
    closest = torch.maximum(torch.minimum(q, hh), -hh)
    d = q - closest
    dist = d.norm(dim=-1)
    inside = dist < 1e-6
    pen = hh - q.abs()
    use_x = pen[..., 0] < pen[..., 1]
    sgn = torch.where(q >= 0, 1.0, -1.0)
    zero = torch.zeros_like(q[..., 0])
    face_n = torch.where(use_x.unsqueeze(-1), torch.stack([sgn[..., 0], zero], -1), torch.stack([zero, sgn[..., 1]], -1))
    n = torch.where(inside.unsqueeze(-1), face_n, d / dist.clamp(min=1e-6).unsqueeze(-1))
    depth = torch.where(inside, torch.minimum(pen[..., 0], pen[..., 1]) + radius, radius - dist)
    hit = depth > 0
    return p + (n * (depth.clamp(min=0) * hit).unsqueeze(-1)).sum(2), hit.any(-1)
