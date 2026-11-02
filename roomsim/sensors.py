"""Batched egocentric sensors: 2D lidar (walls, furniture, other humans) and neighbour features."""
import math

import torch

from .layout import rotate_to_box, to_box_frame

INF = 1e6


def ray_dirs(yaw, n_rays):
    """yaw [B, N] -> unit directions [B, N, R, 2], evenly covering 360 deg starting straight ahead."""
    offs = torch.arange(n_rays, device=yaw.device, dtype=yaw.dtype) * (2 * math.pi / n_rays)
    a = yaw.unsqueeze(-1) + offs
    return torch.stack([a.cos(), a.sin()], -1)


def lidar(pos, yaw, layout, radius, n_rays=16, max_range=5.0, active=None):
    """Distances [B, N, R] normalised to [0, 1] (1 = nothing within max_range).

    pos [B, N, 2], yaw [B, N]; active [B, N] bool marks which slots hold a human.
    """
    B, N, _ = pos.shape
    d = ray_dirs(yaw, n_rays)                         # [B, N, R, 2]
    o = pos.unsqueeze(2)                              # [B, N, 1, 2]
    dx, dy = d[..., 0], d[..., 1]
    eps = 1e-6
    # room walls (interior rectangle)
    tx = torch.where(dx > eps, (layout.width - o[..., 0]) / dx.clamp(min=eps),
                     torch.where(dx < -eps, -o[..., 0] / dx.clamp(max=-eps), torch.full_like(dx, INF)))
    ty = torch.where(dy > eps, (layout.depth - o[..., 1]) / dy.clamp(min=eps),
                     torch.where(dy < -eps, -o[..., 1] / dy.clamp(max=-eps), torch.full_like(dy, INF)))
    t = torch.minimum(tx, ty)
    # oriented furniture boxes: slab test in each box frame
    if layout.num_boxes:
        q = to_box_frame(pos, layout).unsqueeze(2)    # [B, N, 1, K, 2]
        u = rotate_to_box(d, layout)                  # [B, N, R, K, 2]
        u = torch.where(u.abs() < eps, torch.full_like(u, eps), u)
        t1 = (-layout.box_h - q) / u
        t2 = (layout.box_h - q) / u
        tmin = torch.minimum(t1, t2).max(-1).values
        tmax = torch.maximum(t1, t2).min(-1).values
        hit = (tmax >= tmin.clamp(min=0)) & (tmax > 0)
        tb = torch.where(hit, tmin.clamp(min=0), torch.full_like(tmin, INF))
        t = torch.minimum(t, tb.min(-1).values)
    # other humans as discs
    if N > 1:
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)     # [B, i, j, 2] = p_j - p_i
        b = torch.einsum("bnrk,bnmk->bnrm", d, rel)   # projection onto ray
        c2 = (rel * rel).sum(-1).unsqueeze(2) - radius ** 2  # |o + t d - p_j|^2 = r^2
        disc = b * b - c2
        th = b - disc.clamp(min=0).sqrt()
        ok = (disc > 0) & (th > 0)
        eye = torch.eye(N, dtype=torch.bool, device=pos.device).view(1, N, 1, N)
        ok = ok & ~eye
        if active is not None:
            ok = ok & active.view(B, 1, 1, N)
        ta = torch.where(ok, th, torch.full_like(th, INF))
        t = torch.minimum(t, ta.min(-1).values)
    return (t.clamp(max=max_range) / max_range)


def neighbours(pos, vel, yaw, k=3, sense_range=4.0, active=None):
    """Nearest-k other humans in the body frame -> [B, N, k, 5] (dx, dy, dvx, dvy, present)."""
    B, N, _ = pos.shape
    out = torch.zeros(B, N, k, 5, device=pos.device, dtype=pos.dtype)
    if N == 1:
        return out
    rel = pos.unsqueeze(1) - pos.unsqueeze(2)         # [B, i, j, 2]
    rv = vel.unsqueeze(1) - vel.unsqueeze(2)
    dist = rel.norm(dim=-1)
    invalid = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0) | (dist > sense_range)
    if active is not None:
        invalid = invalid | ~active.unsqueeze(1)
    dist = dist.masked_fill(invalid, INF)
    kk = min(k, N - 1)
    dsort, idx = dist.topk(kk, dim=-1, largest=False)
    g = lambda x: torch.gather(x, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2))  # noqa: E731
    r, v = g(rel), g(rv)
    c, s = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)
    rb = torch.stack([c * r[..., 0] + s * r[..., 1], -s * r[..., 0] + c * r[..., 1]], -1)
    vb = torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)
    present = (dsort < INF / 2).to(pos.dtype).unsqueeze(-1)
    out[:, :, :kk] = torch.cat([rb / sense_range, vb / 2.0, present], -1) * present
    return out
