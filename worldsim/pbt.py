"""Population-based training: exploit (copy the winners) + explore (mutate hyper-parameters
and network width). Pure-python bookkeeping, run by the coordinator process."""
import copy
import math
import random

MUTABLE = dict(lr=(1e-5, 3e-3), ent_coef=(0.0, 0.03), clip=(0.05, 0.4), gamma=(0.95, 0.997), lam=(0.8, 0.99))


def init_member(i, base_hp, hidden, rng: random.Random, jitter=True):
    hp = dict(base_hp)
    if jitter and i > 0:  # diverse starting points; member 0 keeps the defaults
        hp["lr"] = hp["lr"] * math.exp(rng.uniform(-0.7, 0.7))
        hp["ent_coef"] = hp["ent_coef"] * math.exp(rng.uniform(-0.7, 0.7))
    return dict(id=i, hp=hp, hidden=hidden, ckpt=None, distill_from=None, lineage=[f"g0:init{i}"])


def mutate(hp, hidden, rng: random.Random, cfg):
    hp = dict(hp)
    lo, hi = cfg.get("perturb", [0.8, 1.25])
    for k, (a, b) in MUTABLE.items():
        if k not in hp:
            continue
        if k in ("gamma", "lam"):  # perturb distance to 1 instead
            v = 1 - (1 - hp[k]) * rng.choice([lo, hi])
        else:
            v = hp[k] * rng.choice([lo, hi]) if hp[k] > 0 else rng.uniform(a, b) * 0.1
        hp[k] = min(max(v, a), b)
    new_hidden = hidden
    if rng.random() < cfg.get("width_mutation_prob", 0.15):
        choices = [w for w in cfg.get("width_choices", [128, 256, 384]) if w != hidden]
        new_hidden = rng.choice(choices) if choices else hidden
    return hp, new_hidden


def exploit_explore(members, results, gen, rng: random.Random, cfg):
    """members: list of member dicts (with updated ckpt); results: {id: metrics incl. score}.
    Returns new member list; losers inherit a winner's checkpoint and mutated hyper-params."""
    ranked = sorted(members, key=lambda m: results[m["id"]]["score"], reverse=True)
    k = max(1, int(round(len(ranked) * cfg.get("exploit_frac", 0.25))))
    winners, losers = ranked[:k], ranked[-k:]
    out = {m["id"]: copy.deepcopy(m) for m in members}
    for m in losers:
        if m in winners:
            continue
        w = rng.choice(winners)
        hp, hidden = mutate(w["hp"], w["hidden"], rng, cfg)
        n = out[m["id"]]
        n["hp"], n["hidden"] = hp, hidden
        n["ckpt"] = w["ckpt"]
        # a width change can't copy weights: distill the parent into the new architecture
        n["distill_from"] = [w["ckpt"]] if hidden != w["hidden"] else None
        n["lineage"] = w["lineage"][-20:] + [f"g{gen}:exploit<-{w['id']}" + (f",w{hidden}" if hidden != w["hidden"] else "")]
    return [out[m["id"]] for m in members], [m["id"] for m in winners], [m["id"] for m in losers if m not in winners]
