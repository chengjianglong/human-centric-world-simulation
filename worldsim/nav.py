"""Geodesic route planning on a coarse occupancy grid, all on the GPU.

The world is rasterised into `cell`-sized squares; a cell is free when a human disc at its
centre clears every static collider. Shortest weighted path lengths from every free cell to
every cell are found by parallel wavefront relaxation (8-connected, no corner cutting).
Road cells outside the crossings cost `road_cost` x more, so planned routes use the zebra
crossing unless the detour is large.

Per step the env only needs cheap gathers:
  potential(p, g) -- geodesic distance from p to goal g (continuous in p; used for reward shaping)
  waypoint(p, g)  -- a point ~1 m ahead on the shortest route (the observation's steering cue)
"""
import math

import torch
import torch.nn.functional as F

from .layout import box_signed_distance, in_rects

INF = float("inf")
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # (dy, dx)


class NavGrid:
    def __init__(self, layout, radius=0.25, cell=0.25, road_cost=4.0, clearance=-0.02, chunk=1024):
        dev = layout.box_c.device
        self.device = dev
        x0, y0, x1, y1 = layout.bounds
        self.cell, self.x0, self.y0 = cell, x0, y0
        self.W, self.H = int(math.ceil((x1 - x0) / cell)), int(math.ceil((y1 - y0) / cell))
        ys = y0 + (torch.arange(self.H, device=dev) + 0.5) * cell
        xs = x0 + (torch.arange(self.W, device=dev) + 0.5) * cell
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        self.centers = torch.stack([cx, cy], -1).reshape(-1, 2)          # [H*W, 2]
        r = radius + clearance
        c = self.centers
        free = (c[:, 0] > x0 + r) & (c[:, 0] < x1 - r) & (c[:, 1] > y0 + r) & (c[:, 1] < y1 - r)
        if layout.num_boxes:
            free &= (box_signed_distance(c, layout) > r).all(-1)
        weight = torch.ones(self.H * self.W, device=dev)
        if layout.road.shape[0]:
            on_road = in_rects(c, layout.road) & ~in_rects(c, layout.crosswalks)
            weight = torch.where(on_road, torch.full_like(weight, road_cost), weight)
        self.weight = weight
        self.free = free
        self._edges()
        # keep only the largest connected region (drops pockets behind furniture)
        seed = int(torch.nonzero(free)[len(torch.nonzero(free)) // 2])
        if len(layout.pois):
            pc = self.cell_of(layout.pois)
            ok = free[pc]
            if ok.any():
                seed = int(pc[ok][0])
        reach = torch.isfinite(self._relax(torch.tensor([seed], device=dev))[0])
        self.free = free & reach
        self._edges()
        self.free_idx = torch.nonzero(self.free).flatten()               # [F] cell ids of free cells
        self.slot = torch.full((self.H * self.W,), -1, dtype=torch.long, device=dev)
        self.slot[self.free_idx] = torch.arange(len(self.free_idx), device=dev)
        # nearest free cell for every cell (for goals/agents sitting on a blocked cell)
        fc = self.centers[self.free_idx]
        near = torch.cat([torch.cdist(self.centers[i:i + 4096], fc).argmin(1) for i in range(0, self.H * self.W, 4096)])
        self.nearest_free = self.free_idx[near]                          # [H*W] -> free cell id
        # all-pairs: D[slot(goal cell), cell] = weighted geodesic distance (inf for blocked cells)
        self.D = torch.cat([self._relax(self.free_idx[i:i + chunk]) for i in range(0, len(self.free_idx), chunk)])
        nb = []
        for dy, dx in [(0, 0)] + OFFS:
            nb.append(torch.tensor([dy, dx], device=dev))
        self.nb9 = torch.stack(nb)                                        # [9, 2]

    # ------------------------------------------------------------------ construction
    def _edges(self):
        H, W = self.H, self.W
        free = self.free.view(H, W)
        w = self.weight.view(H, W)
        pf = F.pad(free.float(), (1, 1, 1, 1)).bool()
        pw = F.pad(w, (1, 1, 1, 1), value=1.0)
        self.edge_cost = []
        for dy, dx in OFFS:
            nbf = pf[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            ok = free & nbf
            if dy and dx:  # diagonal: both orthogonal neighbours must be free (no corner cutting)
                ok &= pf[1 + dy:1 + dy + H, 1:1 + W] & pf[1:1 + H, 1 + dx:1 + dx + W]
            nbw = pw[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            cost = math.hypot(dy, dx) * self.cell * (w + nbw) / 2
            self.edge_cost.append(torch.where(ok, cost, torch.full_like(cost, INF)))

    @torch.no_grad()
    def _relax(self, sources):
        """Distances from each source cell to every cell -> [S, H*W] (wavefront relaxation)."""
        H, W, S = self.H, self.W, len(sources)
        D = torch.full((S, H, W), INF, device=self.device)
        D.view(S, -1)[torch.arange(S, device=self.device), sources] = 0.0
        it = 0
        while True:
            P = F.pad(D, (1, 1, 1, 1), value=INF)
            best = D
            for (dy, dx), ec in zip(OFFS, self.edge_cost):
                best = torch.minimum(best, P[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] + ec)
            it += 1
            if it % 16 == 0 and torch.equal(best, D):
                break
            D = best
            if it > 4 * (H + W) * 4:
                break
        return D.view(S, -1)

    # ------------------------------------------------------------------ queries
    def cell_xy(self, p):
        ix = ((p[..., 0] - self.x0) / self.cell).floor().long().clamp(0, self.W - 1)
        iy = ((p[..., 1] - self.y0) / self.cell).floor().long().clamp(0, self.H - 1)
        return iy, ix

    def cell_of(self, p):
        iy, ix = self.cell_xy(p)
        return iy * self.W + ix

    def is_free(self, p):
        return self.free[self.cell_of(p)]

    def goal_slot(self, g):
        """Goal points [...,2] -> row index into D (nearest free cell)."""
        return self.slot[self.nearest_free[self.cell_of(g)]]

    def _neigh(self, p):
        iy, ix = self.cell_xy(p)
        ny = iy.unsqueeze(-1) + self.nb9[:, 0]
        nx = ix.unsqueeze(-1) + self.nb9[:, 1]
        inside = (ny >= 0) & (ny < self.H) & (nx >= 0) & (nx < self.W)
        return (ny.clamp(0, self.H - 1) * self.W + nx.clamp(0, self.W - 1)), inside

    def potential(self, p, g, gs):
        """Geodesic distance [...] from p to goal g (gs = goal_slot(g)); returns (phi, best neighbour cell)."""
        cells, inside = self._neigh(p)                                    # [..., 9]
        d = self.D[gs.unsqueeze(-1), cells]
        d = torch.where(inside, d, torch.full_like(d, INF))
        cost = d + (self.centers[cells] - p.unsqueeze(-2)).norm(dim=-1)
        best, arg = cost.min(-1)
        best_cell = torch.gather(cells, -1, arg.unsqueeze(-1)).squeeze(-1)
        # fall back to the nearest free cell if the whole neighbourhood is blocked
        nf = self.nearest_free[self.cell_of(p)]
        fb = self.D[gs, nf] + (self.centers[nf] - p).norm(dim=-1)
        bad = ~torch.isfinite(best)
        best = torch.where(bad, fb, best)
        best_cell = torch.where(bad, nf, best_cell)
        gcell = self.free_idx[gs]
        best = best + (g - self.centers[gcell]).norm(dim=-1)
        # close to the goal: straight-line distance
        iy, ix = self.cell_xy(p)
        gy, gx = gcell // self.W, gcell % self.W
        near = ((iy - gy).abs() <= 1) & ((ix - gx).abs() <= 1)
        return torch.where(near, (g - p).norm(dim=-1), best), best_cell, near

    def waypoint(self, p, g, gs, hops=4):
        """A point ~hops cells ahead along the shortest route to g -> ([..., 2], geodesic distance)."""
        phi, cell, near = self.potential(p, g, gs)
        for _ in range(hops):
            iy, ix = cell // self.W, cell % self.W
            ny = iy.unsqueeze(-1) + self.nb9[1:, 0]
            nx = ix.unsqueeze(-1) + self.nb9[1:, 1]
            inside = (ny >= 0) & (ny < self.H) & (nx >= 0) & (nx < self.W)
            nc = ny.clamp(0, self.H - 1) * self.W + nx.clamp(0, self.W - 1)
            d = self.D[gs.unsqueeze(-1), nc] + (self.centers[nc] - self.centers[cell].unsqueeze(-2)).norm(dim=-1)
            d = torch.where(inside, d, torch.full_like(d, INF))
            here = self.D[gs, cell]
            arg = d.argmin(-1)
            step = torch.gather(nc, -1, arg.unsqueeze(-1)).squeeze(-1)
            cell = torch.where(self.D[gs, step] < here, step, cell)
        wp = self.centers[cell]
        gcell = self.free_idx[gs]
        wp = torch.where((near | (cell == gcell)).unsqueeze(-1), g, wp)
        return wp, phi
