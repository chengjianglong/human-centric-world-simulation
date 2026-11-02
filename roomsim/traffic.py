"""Batched street traffic: cars on straight lanes with IDM car-following and pedestrian yielding.

Each world has C cars spread over the lanes (slot i drives lane i % n_lanes). A lane is a loop
over x_range, so a car leaving one end re-enters at the other with a fresh desired speed and
driver type. Drivers treat a person in or next to their lane ahead of them as a stopped
leader (Intelligent Driver Model), but braking is limited, and "distracted" drivers only
notice people within a few metres -- stepping out in front of them is dangerous.
"""
import torch


class Traffic:
    def __init__(self, cfg, B, C, device, gen):
        self.c, self.B, self.C, self.device, self.gen = cfg, B, C, torch.device(device), gen
        lanes = cfg["lanes"]
        self.n_lanes = len(lanes)
        lane = torch.arange(C, device=self.device) % max(self.n_lanes, 1)
        self.lane = lane
        self.lane_y = torch.tensor([ln["y"] for ln in lanes], device=self.device)[lane] if C else torch.zeros(0, device=self.device)
        self.dir = torch.tensor([float(ln["dir"]) for ln in lanes], device=self.device)[lane] if C else torch.zeros(0, device=self.device)
        self.x0, self.x1 = cfg["x_range"]
        self.ring = self.x1 - self.x0
        L, Wd = cfg["car_size"][0], cfg["car_size"][1]
        self.half = torch.tensor([L / 2, Wd / 2], device=self.device)
        z = lambda: torch.zeros(B, C, device=self.device)  # noqa: E731
        self.s, self.v, self.v0, self.view = z(), z(), z(), z()
        self.same_lane = (lane.view(-1, 1) == lane.view(1, -1)) & ~torch.eye(C, dtype=torch.bool, device=self.device)
        self.reset(torch.ones(B, dtype=torch.bool, device=self.device))

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _driver(self, n):
        lo, hi = self.c["speed"]
        v0 = lo + (hi - lo) * self._rand(n)
        distracted = self._rand(n) < self.c["distracted_prob"]
        view = torch.where(distracted, torch.full_like(v0, self.c["distracted_view"]), torch.full_like(v0, self.c["attentive_view"]))
        return v0, view

    def reset(self, worlds):
        wb = worlds.nonzero(as_tuple=True)[0]
        if wb.numel() == 0 or self.C == 0:
            return
        n = wb.numel()
        # evenly spaced per lane with jitter
        k = torch.zeros(self.C, device=self.device)
        for ln in range(self.n_lanes):
            idx = (self.lane == ln).nonzero().flatten()
            k[idx] = torch.arange(len(idx), device=self.device, dtype=torch.float32) / max(len(idx), 1)
        s = (k.view(1, -1) + 0.6 / max(self.C, 1) * self._rand(n, self.C) + self._rand(n, 1)) * self.ring
        self.s[wb] = torch.remainder(s, self.ring)
        v0, view = self._driver(n * self.C)
        self.v0[wb], self.view[wb] = v0.view(n, self.C), view.view(n, self.C)
        self.v[wb] = self.v0[wb] * (0.7 + 0.3 * self._rand(n, self.C))

    # ------------------------------------------------------------------ geometry
    def xy(self, s=None):
        s = self.s if s is None else s
        x = torch.where(self.dir > 0, self.x0 + s, self.x1 - s)
        return torch.stack([x, self.lane_y.expand_as(x)], -1)           # [B, C, 2]

    def velocity(self):
        return torch.stack([self.v * self.dir, torch.zeros_like(self.v)], -1)

    def yaw(self):
        return torch.where(self.dir > 0, torch.zeros_like(self.v), torch.full_like(self.v, torch.pi))

    # ------------------------------------------------------------------ dynamics
    @torch.no_grad()
    def step(self, dt, ped):
        """Advance one step. ped [B, N, 2] = people. Returns (old centres, new centres) for swept collision."""
        if self.C == 0:
            e = torch.zeros(self.B, 0, 2, device=self.device)
            return e, e
        c = self.c
        old = self.xy()
        # leader car in the same lane (ring distance ahead, bumper to bumper)
        ds = torch.remainder(self.s.unsqueeze(1) - self.s.unsqueeze(2), self.ring)   # [B, i, j] = s_j - s_i
        gap_car = torch.where(self.same_lane, ds - 2 * self.half[0], torch.full_like(ds, 1e9))
        g_car, j = gap_car.min(-1)
        v_lead = torch.gather(self.v, 1, j)
        # people ahead of the bumper, near the lane, within this driver's view
        front = old[..., 0] + self.dir * self.half[0]                                   # [B, C]
        ahead = (ped[..., 0].unsqueeze(1) - front.unsqueeze(-1)) * self.dir.view(1, -1, 1)  # [B, C, N]
        lateral = (ped[..., 1].unsqueeze(1) - self.lane_y.view(1, -1, 1)).abs()
        seen = (ahead > -0.5) & (ahead < self.view.unsqueeze(-1)) & (lateral < self.half[1] + c["ped_lateral"])
        g_ped = torch.where(seen, ahead - c["ped_gap"], torch.full_like(ahead, 1e9)).min(-1).values
        ped_first = g_ped < g_car
        gap = torch.where(ped_first, g_ped, g_car).clamp(min=0.05)
        dv = self.v - torch.where(ped_first, torch.zeros_like(v_lead), v_lead)
        a, b = c["accel"], c["comfort_decel"]
        s0 = torch.where(ped_first, torch.full_like(gap, 0.5), torch.full_like(gap, c["min_gap"]))
        s_star = s0 + (self.v * c["headway"] + self.v * dv / (2 * (a * b) ** 0.5)).clamp(min=0)
        acc = a * (1 - (self.v / self.v0.clamp(min=0.1)) ** 4 - (s_star / gap) ** 2)
        acc = acc.clamp(-c["max_decel"], a)
        self.v = (self.v + acc * dt).clamp(min=0.0)
        s = self.s + self.v * dt
        wrapped = s >= self.ring
        self.s = torch.remainder(s, self.ring)
        if wrapped.any():
            n = int(wrapped.sum())
            v0, view = self._driver(n)
            self.v0[wrapped], self.view[wrapped] = v0, view
        new = self.xy()
        # a wrapped car teleports: use its new position only
        old = torch.where(wrapped.unsqueeze(-1), new, old)
        return old, new

    def swept_boxes(self, old, new):
        """Axis-aligned boxes covering each car's motion this step -> (centres, half sizes) [B, C, 2]."""
        c = (old + new) / 2
        h = self.half.expand_as(c) + torch.stack([(new[..., 0] - old[..., 0]).abs() / 2, torch.zeros_like(c[..., 1])], -1)
        return c, h
