"""Policy distillation: a fresh student imitates an ensemble of elite teachers (DAgger-style).

This is the "recursive" step of self-improvement: each new generation of students is
bootstrapped from the best policies of all previous generations, then improved further by RL.
"""
import torch

from .policy import ActorCritic


def distill(student: ActorCritic, teacher_paths, env, steps=300, lr=1e-3, log_fn=None):
    dev = env.device
    teachers = [ActorCritic.from_checkpoint(p, dev).eval() for p in teacher_paths]
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    obs = env.observe()
    last = {}
    for i in range(steps):
        beta = max(0.0, 1.0 - i / (0.6 * steps))  # teacher drives early, student later
        with torch.no_grad():
            dists = [t.dist(obs) for t in teachers]
            t_mu = torch.stack([d.mean for d in dists]).mean(0)
            t_std = torch.stack([d.stddev for d in dists]).mean(0)
            t_val = torch.stack([t.value(obs) for t in teachers]).mean(0)
            student.norm.update(obs)
        d = student.dist(obs)
        loss_pi = ((d.mean - t_mu) ** 2).sum(-1).mean()
        loss_std = ((d.stddev.log() - t_std.log()) ** 2).sum(-1).mean()
        loss_v = ((student.value(obs) - t_val) ** 2).mean() / (t_val.var() + 1.0)
        loss = loss_pi + 0.1 * loss_std + 0.5 * loss_v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            use_t = torch.rand(env.B, env.N, 1, device=dev) < beta
            a = torch.where(use_t, t_mu + t_std * torch.randn_like(t_mu), d.sample())
            obs, _, _, _ = env.step(a)
        last = dict(distill_pi=loss_pi.item(), distill_v=loss_v.item())
        if log_fn and i % 50 == 0:
            log_fn(i, last)
    return last
