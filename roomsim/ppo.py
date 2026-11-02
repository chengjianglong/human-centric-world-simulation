"""PPO (clipped surrogate + GAE) for parameter-shared multi-human control, with optional
frozen self-play opponents controlling some of the humans in each room."""
import time

import torch
import torch.nn as nn

DEFAULT_HP = dict(lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.003, vf_coef=0.5, epochs=4,
                  minibatches=4, rollout_len=64, max_grad_norm=1.0, target_kl=0.03, selfplay_frac=0.5)


class PPOTrainer:
    def __init__(self, net, env, hp=None, opponent=None):
        self.net, self.env = net, env
        self.hp = {**DEFAULT_HP, **(hp or {})}
        self.opt = torch.optim.Adam(net.parameters(), lr=self.hp["lr"], eps=1e-5)
        self.dev = env.device
        self.set_opponent(opponent)
        self.obs = env.observe()
        self.ep_ret = torch.zeros(env.B, env.N, device=self.dev)
        self.updates = 0

    def set_opponent(self, opponent):
        """Frozen policy that drives slots 1..N-1 in a `selfplay_frac` share of the rooms."""
        B, N = self.env.B, self.env.N
        self.opponent = opponent
        self.learner = torch.ones(B, N, dtype=torch.bool, device=self.dev)
        if opponent is not None and N > 1:
            k = int(B * self.hp["selfplay_frac"])
            self.learner[:k, 1:] = False
            opponent.eval()

    def set_lr(self, lr):
        self.hp["lr"] = lr
        for g in self.opt.param_groups:
            g["lr"] = lr

    @torch.no_grad()
    def collect(self):
        env, T = self.env, self.hp["rollout_len"]
        B, N, D = env.B, env.N, env.obs_dim
        buf = dict(obs=torch.empty(T, B, N, D, device=self.dev), act=torch.empty(T, B, N, 2, device=self.dev),
                   logp=torch.empty(T, B, N, device=self.dev), val=torch.empty(T, B, N, device=self.dev),
                   rew=torch.empty(T, B, N, device=self.dev), done=torch.empty(T, B, N, device=self.dev))
        lm = self.learner.unsqueeze(-1)
        for t in range(T):
            o = self.obs
            a, lp, v = self.net.act(o)
            if self.opponent is not None and not bool(self.learner.all()):
                oa, _, _ = self.opponent.act(o)
                a = torch.where(lm, a, oa)
            o2, r, d, _ = env.step(a)
            buf["obs"][t], buf["act"][t], buf["logp"][t], buf["val"][t] = o, a, lp, v
            buf["rew"][t] = r
            buf["done"][t] = d.float().unsqueeze(-1).expand(B, N)
            self.obs = o2
        self.net.norm.update(buf["obs"][:, self.learner])
        buf["last_val"] = self.net.value(self.obs)
        return buf

    @torch.no_grad()
    def gae(self, buf):
        g, lam, T = self.hp["gamma"], self.hp["lam"], self.hp["rollout_len"]
        adv = torch.zeros_like(buf["rew"])
        last = torch.zeros_like(buf["rew"][0])
        for t in reversed(range(T)):
            nv = buf["last_val"] if t == T - 1 else buf["val"][t + 1]
            nonterm = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + g * nv * nonterm - buf["val"][t]
            last = delta + g * lam * nonterm * last
            adv[t] = last
        return adv, adv + buf["val"]

    def update(self, buf):
        hp = self.hp
        adv, ret = self.gae(buf)
        m = self.learner.unsqueeze(0).expand(hp["rollout_len"], -1, -1)
        obs, act, logp_old = buf["obs"][m], buf["act"][m], buf["logp"][m]
        adv, ret = adv[m], ret[m]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        mb = n // hp["minibatches"]
        logs = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0, kl=0.0, clipfrac=0.0)
        cnt = 0
        for _ in range(hp["epochs"]):
            perm = torch.randperm(n, device=self.dev)
            for i in range(hp["minibatches"]):
                idx = perm[i * mb:(i + 1) * mb]
                lp, ent, v = self.net.evaluate(obs[idx], act[idx])
                ratio = (lp - logp_old[idx]).exp()
                a = adv[idx]
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * a).mean()
                v_loss = 0.5 * ((v - ret[idx]) ** 2).mean()
                loss = pi_loss + hp["vf_coef"] * v_loss - hp["ent_coef"] * ent.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), hp["max_grad_norm"])
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (lp - logp_old[idx])).mean()
                    logs["pi_loss"] += pi_loss.item()
                    logs["v_loss"] += v_loss.item()
                    logs["entropy"] += ent.mean().item()
                    logs["kl"] += kl.item()
                    logs["clipfrac"] += ((ratio - 1).abs() > hp["clip"]).float().mean().item()
                cnt += 1
            if logs["kl"] / cnt > hp["target_kl"]:
                break
        self.updates += 1
        out = {k: v / cnt for k, v in logs.items()}
        out["mean_reward"] = buf["rew"][m].mean().item()
        return out

    def train(self, num_updates, log_fn=None):
        last = {}
        for _ in range(num_updates):
            t0 = time.time()
            buf = self.collect()
            last = self.update(buf)
            last["sps"] = buf["rew"].numel() / (time.time() - t0)
            if log_fn:
                log_fn(self.updates, last)
        return last


@torch.no_grad()
def evaluate_policy(net, env_factory, steps=600, deterministic=True):
    """Run a fresh env (fixed seed via env_factory) with every human driven by `net`."""
    env = env_factory()
    net.eval()
    obs = env.observe()
    env.reset_stats()
    for _ in range(steps):
        a, _, _ = net.act(obs, deterministic=deterministic)
        obs, _, _, _ = env.step(a)
    net.train()
    m = env.summary()
    m["score"] = score(m)
    return m


def score(m):
    """Single fitness number for population-based training.

    Collisions are weighted heavily (1% of steps in contact costs 0.8 goals/min) so selection
    favours polite walkers over fast ones that brush past furniture and people. Being hit by a car
    (world mode) costs far more on top: 0.1% of steps under a car = -0.3."""
    return m["goals_per_min"] * m["success"] - 80.0 * m["collision_rate"] - 300.0 * m.get("car_hit_rate", 0.0)
