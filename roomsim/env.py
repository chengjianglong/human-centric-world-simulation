"""VecRoomEnv: B parallel copies of the room, each with N digital humans, fully batched on GPU.

Each human is a disc (radius ~0.25 m) with unicycle dynamics (speed v, turn rate w), a gait
phase that drives the Blender walk animation, and a goal (a named point of interest such as
"sofa" or "door", or a random free spot). Agents get dense progress reward, a bonus on
arrival, and penalties for bumping into furniture/walls/each other, invading personal
space and jerky control.
"""
import math

import torch

from . import gait
from .layout import box_signed_distance, push_out_of_boxes
from .sensors import lidar, neighbours

DEFAULTS = dict(
    dt=0.05, radius=0.25, max_speed=1.5, min_speed=-0.3, accel=2.5, max_turn=2.5, turn_accel=10.0,
    n_rays=16, lidar_range=5.0, k_neighbours=3,
    goal_radius=0.35, goal_timeout=300, episode_len=1000, poi_prob=0.6, goal_min_dist=1.0,
    social_dist=0.8,
    r_progress=5.0, r_goal=5.0, r_hit=0.5, r_agent_hit=1.0, r_social=0.5, r_smooth=0.02, r_time=0.01, r_backward=0.1,
)
OBS_DIM = 16 + 4 + 4 + 4 + 3 * 5  # lidar, goal, self, pose, neighbours (for default n_rays/k)


def obs_dim(cfg=None):
    c = {**DEFAULTS, **(cfg or {})}
    return c["n_rays"] + 4 + 4 + 4 + c["k_neighbours"] * 5


class VecRoomEnv:
    def __init__(self, layout, num_worlds, num_agents, device="cuda", cfg=None, seed=0):
        self.L = layout
        self.B, self.N = num_worlds, num_agents
        self.device = torch.device(device)
        self.c = {**DEFAULTS, **(cfg or {})}
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.obs_dim = obs_dim(self.c)
        B, N, dev = self.B, self.N, self.device
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        self.pos, self.goal, self.prev_a = z(B, N, 2), z(B, N, 2), z(B, N, 2)
        self.yaw, self.v, self.w, self.phase = z(B, N), z(B, N), z(B, N), z(B, N)
        self.goal_t = torch.zeros(B, N, dtype=torch.long, device=dev)
        self.goal_idx = torch.full((B, N), -1, dtype=torch.long, device=dev)
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
        lo = torch.tensor([r, r], device=self.device)
        hi = torch.tensor([self.L.width - r, self.L.depth - r], device=self.device)
        cand = lo + (hi - lo) * self._rand(count, m, 2)
        ok = torch.ones(count, m, dtype=torch.bool, device=self.device)
        if self.L.num_boxes:
            ok &= (box_signed_distance(cand, self.L) > r).all(-1)
        if others is not None and others.shape[1] > 0:
            d = (cand.unsqueeze(2) - others.unsqueeze(1)).norm(dim=-1)  # [count, m, K]
            ok &= torch.nan_to_num(d, nan=1e9).min(-1).values > 2 * self.c["radius"] + 0.1
        idx = ok.float().argmax(1)  # first valid (0 if none -- vanishingly rare)
        return cand[torch.arange(count, device=self.device), idx]

    def _sample_goals(self, mask):
        """New goals for agents where mask [B, N] is True."""
        b, n = mask.nonzero(as_tuple=True)
        if b.numel() == 0:
            return
        cnt, m = b.numel(), 16
        cand = self._free_points(cnt * m, m=8).view(cnt, m, 2)
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
        self.prev_dist[b, n] = (self.goal[b, n] - self.pos[b, n]).norm(dim=-1)

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
        m = torch.zeros(self.B, self.N, dtype=torch.bool, device=self.device)
        m[wb] = True
        self._sample_goals(m)

    # ------------------------------------------------------------------ stats
    def reset_stats(self):
        # kept as device tensors so step() never forces a GPU sync
        self.stats = {k: torch.zeros((), device=self.device, dtype=torch.float64)
                      for k in ("goals", "failed", "hit", "agent_hit", "agent_steps", "path")}

    def summary(self):
        s = {k: float(v) for k, v in self.stats.items()}
        mins = s["agent_steps"] * self.c["dt"] / 60.0
        done = s["goals"] + s["failed"]
        return dict(
            success=s["goals"] / done if done else 0.0,
            goals_per_min=s["goals"] / mins if mins else 0.0,
            collision_rate=(s["hit"] + s["agent_hit"]) / max(s["agent_steps"], 1.0),
            agent_collision_rate=s["agent_hit"] / max(s["agent_steps"], 1.0),
            mean_speed=s["path"] / max(s["agent_steps"] * self.c["dt"], 1e-6),
        )

    # ------------------------------------------------------------------ observation
    def observe(self):
        c, L = self.c, self.L
        cy, sy = self.yaw.cos(), self.yaw.sin()
        lid = lidar(self.pos, self.yaw, L, c["radius"], c["n_rays"], c["lidar_range"])
        g = self.goal - self.pos
        gb = torch.stack([cy * g[..., 0] + sy * g[..., 1], -sy * g[..., 0] + cy * g[..., 1]], -1)
        dist = g.norm(dim=-1, keepdim=True)
        goal = torch.cat([gb / 5.0, (dist / 5.0).clamp(max=1.0), (self.goal_t.float() / c["goal_timeout"]).unsqueeze(-1)], -1)
        me = torch.stack([self.v / c["max_speed"], self.w / c["max_turn"]], -1)
        me = torch.cat([me, self.prev_a], -1)
        pose = torch.stack([self.pos[..., 0] / L.width * 2 - 1, self.pos[..., 1] / L.depth * 2 - 1, cy, sy], -1)
        vel = torch.stack([self.v * cy, self.v * sy], -1)
        nb = neighbours(self.pos, vel, self.yaw, c["k_neighbours"]).flatten(-2)
        return torch.cat([lid, goal, me, pose, nb], -1)

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
        # furniture and walls
        pos, hit = push_out_of_boxes(pos, r, L)
        lo = torch.tensor([r, r], device=self.device)
        hi = torch.tensor([L.width - r, L.depth - r], device=self.device)
        wall = ((pos < lo) | (pos > hi)).any(-1)
        pos = torch.maximum(torch.minimum(pos, hi), lo)
        hit = hit | wall
        self.v = torch.where(hit | agent_hit, self.v * 0.3, self.v)
        moved = (pos - old).norm(dim=-1)
        self.pos = pos
        self.phase = torch.remainder(self.phase + 2 * math.pi * (0.5 + 0.42 * self.v.abs().clamp(max=gait.MAX_SPEED)) * dt, 2 * math.pi)

        # goals
        dist = (self.goal - self.pos).norm(dim=-1)
        progress = self.prev_dist - dist
        reached = dist < c["goal_radius"]
        self.goal_t += 1
        failed = (~reached) & (self.goal_t >= c["goal_timeout"])
        rew = (c["r_progress"] * progress + c["r_goal"] * reached.float()
               - c["r_hit"] * hit.float() - c["r_agent_hit"] * agent_hit.float()
               - c["r_social"] * social - c["r_smooth"] * ((a - self.prev_a) ** 2).sum(-1)
               - c["r_time"] - c["r_backward"] * (-self.v).clamp(min=0))
        self.prev_a = a
        self.prev_dist = dist
        self._sample_goals(reached | failed)

        s = self.stats
        s["goals"] += reached.sum()
        s["failed"] += failed.sum()
        s["hit"] += hit.sum()
        s["agent_hit"] += agent_hit.sum()
        s["agent_steps"] += self.B * self.N
        s["path"] += moved.sum()

        self.t += 1
        done = self.t >= c["episode_len"]
        info = dict(reached=reached, failed=failed, hit=hit, agent_hit=agent_hit)
        if done.any():
            self.reset(done)
        return self.observe(), rew, done, info

    def snapshot(self):
        return dict(pos=self.pos.clone(), yaw=self.yaw.clone(), v=self.v.clone(), phase=self.phase.clone(),
                    goal=self.goal.clone(), goal_idx=self.goal_idx.clone())
