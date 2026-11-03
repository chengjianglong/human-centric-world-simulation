"""Benchmark a checkpoint and export a rollout for Blender.

    python evaluate.py --ckpt runs/main/best.pt --out renders/rollout.npz [--agents 4] [--steps 600] [--bvh]
"""
import argparse
import json
import os

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec", help="room/world JSON (default: the one the checkpoint was trained on)")
    ap.add_argument("--cars", type=int, help="cars on the street (world mode; default: the checkpoint's stage)")
    ap.add_argument("--agents", type=int, help="humans in the room (default: the checkpoint's curriculum stage)")
    ap.add_argument("--steps", type=int, default=600, help="rollout length (dt=0.05 -> 600 = 30 s)")
    ap.add_argument("--worlds", type=int, default=64, help="rooms for the benchmark")
    ap.add_argument("--bench-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default=os.path.join(ROOT, "renders", "rollout.npz"))
    ap.add_argument("--bvh", action="store_true", help="also write one .bvh per human next to --out")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from worldsim.env import VecRoomEnv
    from worldsim.export import rollout, save_npz, write_bvh
    from worldsim.layout import load_layout
    from worldsim.policy import ActorCritic
    from worldsim.ppo import evaluate_policy

    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    net = ActorCritic.from_checkpoint(ck, a.device)
    n = a.agents or ck.get("n_agents", 4)
    env_cfg = dict(ck.get("env_cfg", {}))
    if a.cars is not None:
        env_cfg["n_cars"] = a.cars
    spec = a.spec or os.path.join(ROOT, ck.get("spec", "configs/room.json"))
    layout = load_layout(spec, device=a.device)
    m = evaluate_policy(net, lambda: VecRoomEnv(layout, a.worlds, n, a.device, env_cfg, seed=a.seed), a.bench_steps)
    print(json.dumps({"ckpt": a.ckpt, "spec": os.path.relpath(spec, ROOT), "agents": n, "trained_gen": ck.get("gen"), **{k: round(v, 4) for k, v in m.items()}}, indent=1))
    data = rollout(net, layout, n, a.steps, a.seed, a.device, env_cfg)
    save_npz(a.out, data)
    print(f"rollout: {data['pos'].shape[0]} frames x {n} humans, {len(data['reached'])} goals reached -> {a.out}")
    if a.bvh:
        for i in range(n):
            p = os.path.splitext(a.out)[0] + f"_human{i}.bvh"
            write_bvh(p, data, i)
            print("bvh:", p)


if __name__ == "__main__":
    main()
