"""Actor-critic MLP with built-in observation normalisation (so checkpoints are self-contained)."""
import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, dim, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)


def _mlp(i, h, o, layers):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=2, hidden=256, layers=3, init_log_std=-0.5):
        super().__init__()
        self.cfg = dict(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, layers=layers)
        self.norm = RunningNorm(obs_dim)
        self.actor = _mlp(obs_dim, hidden, act_dim, layers)
        self.critic = _mlp(obs_dim, hidden, 1, layers)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, obs):
        mu = torch.tanh(self.actor(self.norm(obs)))
        return torch.distributions.Normal(mu, self.log_std.clamp(-3, 0.5).exp().expand_as(mu))

    def value(self, obs):
        return self.critic(self.norm(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)

    def evaluate(self, obs, act):
        d = self.dist(obs)
        return d.log_prob(act).sum(-1), d.entropy().sum(-1), self.value(obs)

    # ---------------------------------------------------------------- checkpoint helpers
    def checkpoint(self, **extra):
        return dict(cfg=self.cfg, state=self.state_dict(), **extra)

    @staticmethod
    def from_checkpoint(ck, device="cpu"):
        if isinstance(ck, str):
            ck = torch.load(ck, map_location=device, weights_only=False)
        net = ActorCritic(**ck["cfg"]).to(device)
        net.load_state_dict(ck["state"])
        return net
