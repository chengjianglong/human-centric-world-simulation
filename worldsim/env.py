"""VecRoomEnv: B parallel copies of the room (or the house + garden + street world), each with N
digital humans, fully batched on GPU.

Each human is a disc (radius ~0.25 m) with unicycle dynamics (speed v, turn rate w), a gait
phase that drives the Blender walk animation, and a goal (a named point of interest such as
"sofa", "bus_stop" or "garden_bench", or a random free spot). Agents get dense progress reward,
a bonus on arrival, and penalties for bumping into furniture/walls/each other, invading personal
space and jerky control.

World mode (layout.kind == "world") adds:
  * geodesic routing: progress is measured along the shortest walkable route (through doors,
    the garden gate, preferring the zebra crossing), and the observation includes a waypoint ~1 m
    ahead on that route;
  * street traffic (worldsim.traffic): cars that yield -- mostly -- to people; being hit by a car is
    heavily penalised, and walking on the road outside the crossing costs a little every step.
"""
import math

import torch

from . import gait
from .layout import box_signed_distance, in_rects, push_out_of_aabbs, push_out_of_boxes
from .sensors import car_features, lidar, neighbours

DEFAULTS = dict(
    dt=0.05, radius=0.25, max_speed=1.5, min_speed=-0.3, accel=2.5, max_turn=2.5, turn_accel=10.0,
    n_rays=16, lidar_range=5.0, k_neighbours=3,
    goal_radius=0.35, goal_timeout=300, episode_len=1000, poi_prob=0.6, goal_min_dist=1.0,
    social_dist=0.8,
    r_progress=5.0, r_goal=5.0, r_hit=0.5, r_agent_hit=1.0, r_social=0.5, r_smooth=0.02, r_time=0.01, r_backward=0.1,
    # world mode
    n_cars=4, k_cars=2, r_car=10.0, r_jaywalk=0.05,
)
OBS_DIM = 16 + 4 + 4 + 4 + 3 * 5  # room mode: lidar, goal, self, pose, neighbours (for default n_rays/k)


def obs_dim(cfg=None, kind="room"):
    c = {**DEFAULTS, **(cfg or {})}
    base = c["n_rays"] + 4 + 4 + 4 + c["k_neighbours"] * 5
    if kind == "world":  # + waypoint (3) + flags (indoor, on road, on crossing) + cars
        base += 3 + 3 + c["k_cars"] * 5
    return base


class VecRoomEnv:
    def __init__(self, layout, num_worlds, num_agents, device="cuda", cfg=None, seed=0):
        self.L = layout
        self.B, self.N = num_worlds, num_agents
        self.device = torch.device(device)
        self.c = {**DEFAULTS, **(cfg or {})}
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.world = layout.kind == "world"
        self.obs_dim = obs_dim(self.c, layout.kind)
        x0, y0, x1, y1 = layout.bounds if layout.bounds else (0.0, 0.0, layout.width, layout.depth)
        self.lo_b = torch.tensor([x0, y0], device=self.device)
        self.hi_b = torch.tensor([x1, y1], device=self.device)
        self.nav = layout.nav(self.c["radius"]) if self.world else None
        self.traffic = None
        if self.world and layout.traffic and self.c["n_cars"] > 0:
            from .traffic import Traffic
            self.traffic = Traffic(layout.traffic, self.B, int(self.c["n_cars"]), self.device, self.gen)
        B, N, dev = self.B, self.N, self.device
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        self.pos, self.goal, self.prev_a = z(B, N, 2), z(B, N, 2), z(B, N, 2)
        self.yaw, self.v, self.w, self.phase = z(B, N), z(B, N), z(B, N), z(B, N)
        self.goal_t = torch.zeros(B, N, dtype=torch.long, device=dev)
        self.goal_idx = torch.full((B, N), -1, dtype=torch.long, device=dev)
        self.goal_slot = torch.zeros(B, N, dtype=torch.long, device=dev)
        self.t = torch.zeros(B, dtype=torch.long, device=dev)
        self.prev_dist = z(B, N)
        self.reset_stats()
        self.reset(torch.ones(B, dtype=torch.bool, device=dev))

    # ------------------------------------------------------------------ sampling
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _free_points(self, count, others=None, m=64, clearance=0.08):
        """Sample `count` collision-free points; `others` [count, K, 2] (nan = ignore) must stay > 2r apart."""
        r = self.c["radius"] + clearance
        lo, hi = self.lo_b + r, self.hi_b - r
        cand = lo + (hi - lo) * self._rand(count, m, 2)
        ok = torch.ones(count, m, dtype=torch.bool, device=self.device)
        if self.L.num_boxes:
            ok &= (box_signed_distance(cand, self.L) > r).all(-1)
        if self.world:  # reachable, and not standing in the road
            ok &= self.nav.is_free(cand) & ~in_rects(cand, self.L.road)
        if others is not None and others.shape[1] > 0:
            d = (cand.unsqueeze(2) - others.unsqueeze(1)).norm(dim=-1)  # [count, m, K]
            ok &= torch.nan_to_num(d, nan=1e9).min(-1).values > 2 * self.c["radius"] + 0.1
        idx = ok.float().argmax(1)  # first valid (0 if none -- vanishingly rare)
        return cand[torch.arange(count, device=self.device), idx]

    def _dist_to_goal(self, pos, goal, slot):
        if self.world:
            return self.nav.potential(pos, goal, slot)[0]
        return (goal - pos).norm(dim=-1)

    def _sample_goals(self, mask):
        """New goals for agents where mask [B, N] is True."""
        b, n = mask.nonzero(as_tuple=True)
        if b.numel() == 0:
            return
        cnt, m = b.numel(), 16
        cand = self._free_points(cnt * m, m=8 if not self.world else 16).view(cnt, m, 2)
        idx = torch.full((cnt, m), -1, dtype=torch.long, device=self.device)
        P = self.L.pois.shape[0]
        if P:
            use_poi = self._rand(cnt, m) < self.c["poi_prob"]
            pick = (self._rand(cnt, m) * P).long().clamp(max=P - 1)
            cand = torch.where(use_poi.unsqueeze(-1), self.L.pois[pick], cand)
            idx = torch.where(use_poi, pick, idx)
        far = (cand - self.pos[b, n].unsqueeze(1)).norm(dim=-1) >= self.c["goal_min_dist"]
        sel = far.float().argmax(1)
        ar = torch.arange(cnt, device=self.device)
        self.goal[b, n] = cand[ar, sel]
        self.goal_idx[b, n] = idx[ar, sel]
        self.goal_t[b, n] = 0
        if self.world:
            self.goal_slot[b, n] = self.nav.goal_slot(self.goal[b, n])
        self.prev_dist[b, n] = self._dist_to_goal(self.pos[b, n], self.goal[b, n], self.goal_slot[b, n])

    def reset(self, worlds):
        """Respawn every human in the worlds selected by bool mask [B]."""
        wb = worlds.nonzero(as_tuple=True)[0]
        if wb.numel() == 0:
            return
        cnt = wb.numel()
        placed = torch.full((cnt, self.N, 2), float("nan"), device=self.device)
        for n in range(self.N):
            p = self._free_points(cnt, placed[:, :n])
            placed[:, n] = p
        self.pos[wb] = placed
        self.yaw[wb] = (self._rand(cnt, self.N) * 2 - 1) * math.pi
        for x in (self.v, self.w, self.prev_a):
            x[wb] = 0
        self.phase[wb] = self._rand(cnt, self.N) * 2 * math.pi
        self.t[wb] = 0
        if self.traffic is not None:
            self.traffic.reset(worlds)
        m = torch.zeros(self.B, self.N, dtype=torch.bool, device=self.device)
        m[wb] = True
        self._sample_goals(m)

    # ------------------------------------------------------------------ stats
    def reset_stats(self):
        # kept as device tensors so step() never forces a GPU sync
        self.stats = {k: torch.zeros((), device=self.device, dtype=torch.float64)
                      for k in ("goals", "failed", "hit", "agent_hit", "car_hit", "jaywalk", "agent_steps", "path")}

    def summary(self):
        s = {k: float(v) for k, v in self.stats.items()}
        mins = s["agent_steps"] * self.c["dt"] / 60.0
        done = s["goals"] + s["failed"]
        steps = max(s["agent_steps"], 1.0)
        out = dict(
            success=s["goals"] / done if done else 0.0,
            goals_per_min=s["goals"] / mins if mins else 0.0,
            collision_rate=(s["hit"] + s["agent_hit"] + s["car_hit"]) / steps,
            agent_collision_rate=s["agent_hit"] / steps,
            mean_speed=s["path"] / max(s["agent_steps"] * self.c["dt"], 1e-6),
        )
        if self.world:
            out.update(car_hit_rate=s["car_hit"] / steps, jaywalk_rate=s["jaywalk"] / steps)
        return out

    # ------------------------------------------------------------------ observation
    def _car_boxes(self):
        if self.traffic is None:
            return None
        cc = self.traffic.xy()
        return cc, self.traffic.half.expand_as(cc)

    def observe(self):
        c, L = self.c, self.L
        cy, sy = self.yaw.cos(), self.yaw.sin()
        cars = self._car_boxes()
        lid = lidar(self.pos, self.yaw, L, c["radius"], c["n_rays"], c["lidar_range"], cars=cars)
        g = self.goal - self.pos
        body = lambda v: torch.stack([cy * v[..., 0] + sy * v[..., 1], -sy * v[..., 0] + cy * v[..., 1]], -1)  # noqa: E731
        dist = g.norm(dim=-1, keepdim=True)
        goal = torch.cat([body(g) / 5.0, (dist / 5.0).clamp(max=1.0), (self.goal_t.float() / c["goal_timeout"]).unsqueeze(-1)], -1)
        me = torch.stack([self.v / c["max_speed"], self.w / c["max_turn"]], -1)
        me = torch.cat([me, self.prev_a], -1)
        span = self.hi_b - self.lo_b
        rel = (self.pos - self.lo_b) / span * 2 - 1
        pose = torch.stack([rel[..., 0], rel[..., 1], cy, sy], -1)
        vel = torch.stack([self.v * cy, self.v * sy], -1)
        nb = neighbours(self.pos, vel, self.yaw, c["k_neighbours"]).flatten(-2)
        if not self.world:
            return torch.cat([lid, goal, me, pose, nb], -1)
        wp, geo = self.nav.waypoint(self.pos, self.goal, self.goal_slot)
        wv = wp - self.pos
        wv = wv / wv.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        route = torch.cat([body(wv), (geo / 25.0).clamp(max=1.0).unsqueeze(-1)], -1)
        flags = torch.stack([in_rects(self.pos, L.indoor), in_rects(self.pos, L.road),
                             in_rects(self.pos, L.crosswalks)], -1).float()
        if self.traffic is not None:
            cc, ch = cars
            cf = car_features(self.pos, vel, self.yaw, cc, ch, self.traffic.velocity(), c["k_cars"]).flatten(-2)
        else:
            cf = torch.zeros(self.B, self.N, c["k_cars"] * 5, device=self.device)
        return torch.cat([lid, goal, me, pose, nb, route, flags, cf], -1)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, action):
        """action [B, N, 2] in [-1, 1] (forward accel, turn accel) -> obs, reward [B, N], done [B], info."""
        c, dt, L = self.c, self.c["dt"], self.L
        a = action.clamp(-1, 1)
        self.v = (self.v + a[..., 0] * c["accel"] * dt).clamp(c["min_speed"], c["max_speed"])
        self.w = (self.w + a[..., 1] * c["turn_accel"] * dt).clamp(-c["max_turn"], c["max_turn"])
        self.yaw = torch.remainder(self.yaw + self.w * dt + math.pi, 2 * math.pi) - math.pi
        old = self.pos
        pos = old + self.v.unsqueeze(-1) * torch.stack([self.yaw.cos(), self.yaw.sin()], -1) * dt

        # human-human separation (disc overlap, split the push)
        r = c["radius"]
        agent_hit = torch.zeros(self.B, self.N, dtype=torch.bool, device=self.device)
        social = torch.zeros(self.B, self.N, device=self.device)
        if self.N > 1:
            rel = pos.unsqueeze(2) - pos.unsqueeze(1)          # [B, i, j, 2] = p_i - p_j
            d = rel.norm(dim=-1) + torch.eye(self.N, device=self.device) * 1e3
            overlap = (2 * r - d).clamp(min=0)
            agent_hit = (overlap > 0).any(-1)
            push = (rel / d.clamp(min=1e-4).unsqueeze(-1) * (overlap / 2).unsqueeze(-1)).sum(2)
            pos = pos + push
            social = (c["social_dist"] - d).clamp(min=0).sum(-1)
        # cars (moved first; their swept boxes push people aside)
        car_hit = torch.zeros(self.B, self.N, dtype=torch.bool, device=self.device)
        if self.traffic is not None:
            c_old, c_new = self.traffic.step(dt, pos)
            sc, sh = self.traffic.swept_boxes(c_old, c_new)
            pos, car_hit = push_out_of_aabbs(pos, r, sc, sh)
        # furniture, walls, fences, outer boundary
        pos, hit = push_out_of_boxes(pos, r, L)
        lo, hi = self.lo_b + r, self.hi_b - r
        wall = ((pos < lo) | (pos > hi)).any(-1)
        pos = torch.maximum(torch.minimum(pos, hi), lo)
        hit = hit | wall
        self.v = torch.where(hit | agent_hit, self.v * 0.3, self.v)
        self.v = torch.where(car_hit, torch.zeros_like(self.v), self.v)
        moved = (pos - old).norm(dim=-1)
        self.pos = pos
        self.phase = torch.remainder(self.phase + 2 * math.pi * (0.5 + 0.42 * self.v.abs().clamp(max=gait.MAX_SPEED)) * dt, 2 * math.pi)

        # goals
        dist = self._dist_to_goal(self.pos, self.goal, self.goal_slot)
        progress = self.prev_dist - dist
        reached = (self.goal - self.pos).norm(dim=-1) < c["goal_radius"]
        self.goal_t += 1
        failed = (~reached) & (self.goal_t >= c["goal_timeout"])
        rew = (c["r_progress"] * progress + c["r_goal"] * reached.float()
               - c["r_hit"] * hit.float() - c["r_agent_hit"] * agent_hit.float()
               - c["r_social"] * social - c["r_smooth"] * ((a - self.prev_a) ** 2).sum(-1)
               - c["r_time"] - c["r_backward"] * (-self.v).clamp(min=0))
        jay = torch.zeros_like(reached)
        if self.world:
            jay = in_rects(self.pos, L.road) & ~in_rects(self.pos, L.crosswalks)
            rew = rew - c["r_car"] * car_hit.float() - c["r_jaywalk"] * jay.float()
        self.prev_a = a
        self.prev_dist = dist
        self._sample_goals(reached | failed)

        s = self.stats
        s["goals"] += reached.sum()
        s["failed"] += failed.sum()
        s["hit"] += hit.sum()
        s["agent_hit"] += agent_hit.sum()
        s["car_hit"] += car_hit.sum()
        s["jaywalk"] += jay.sum()
        s["agent_steps"] += self.B * self.N
        s["path"] += moved.sum()

        self.t += 1
        done = self.t >= c["episode_len"]
        info = dict(reached=reached, failed=failed, hit=hit, agent_hit=agent_hit, car_hit=car_hit)
        if done.any():
            self.reset(done)
        return self.observe(), rew, done, info

    def snapshot(self):
        snap = dict(pos=self.pos.clone(), yaw=self.yaw.clone(), v=self.v.clone(), phase=self.phase.clone(),
                    goal=self.goal.clone(), goal_idx=self.goal_idx.clone())
        if self.traffic is not None:
            snap.update(car_pos=self.traffic.xy().clone(), car_yaw=self.traffic.yaw().clone(),
                        car_v=self.traffic.v.clone())
        return snap
