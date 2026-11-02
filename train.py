"""Recursive self-improvement training across GPUs.

    python train.py --config configs/default.yaml            # full run
    python train.py --smoke --generations 2                   # quick check
    python train.py --resume runs/<id>                        # continue a run

A coordinator process runs population-based training; one worker per GPU trains/evaluates
its share of the population each generation (PPO with self-play opponents, optional
distillation from the elite archive), and returns metrics + checkpoint paths.
"""
import argparse
import json
import os
import random
import shutil
import time

import torch
import torch.multiprocessing as mp
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))


# ===================================================================== worker (one per GPU)
def worker(gpu, cmd_q, res_q, cfg, run_dir):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from roomsim.distill import distill
    from roomsim.env import VecRoomEnv
    from roomsim.layout import load_layout
    from roomsim.policy import ActorCritic
    from roomsim.ppo import DEFAULT_HP, PPOTrainer, evaluate_policy

    dev = f"cuda:{gpu}"
    torch.cuda.set_device(dev)
    layout = load_layout(os.path.join(ROOT, cfg["spec"]), device=dev)
    while True:
        msg = cmd_q.get()
        if msg is None:
            return
        gen, stage = msg["gen"], msg["stage"]
        env_cfg = {**cfg.get("env", {}), "goal_min_dist": stage["goal_min_dist"]}
        for task in msg["tasks"]:
            t0 = time.time()
            mid = task["id"]
            torch.manual_seed(task["seed"])
            env = VecRoomEnv(layout, cfg["num_worlds"], stage["n_agents"], dev, env_cfg, seed=task["seed"])
            hp = {**DEFAULT_HP, **cfg["ppo"], **task["hp"]}
            info = {}
            init = task["ckpt"]
            ck = torch.load(init, map_location=dev, weights_only=False) if init else None
            if task["distill_from"]:
                net = ActorCritic(env.obs_dim, hidden=task["hidden"], layers=cfg["policy"]["layers"]).to(dev)
                info.update(distill(net, task["distill_from"], env, cfg["pbt"]["distill_steps"]))
                ck = None
            elif ck is not None and ck["cfg"]["hidden"] == task["hidden"]:
                net = ActorCritic.from_checkpoint(ck, dev)
            else:
                net = ActorCritic(env.obs_dim, hidden=task["hidden"], layers=cfg["policy"]["layers"]).to(dev)
                ck = None
            opp = ActorCritic.from_checkpoint(task["opponent"], dev) if task["opponent"] else None
            tr = PPOTrainer(net, env, hp, opponent=opp)
            if ck is not None and "opt" in ck:
                try:
                    tr.opt.load_state_dict(ck["opt"])
                except ValueError:
                    pass
            tr.set_lr(hp["lr"])
            logs = tr.train(cfg["updates_per_generation"])
            ev_seed = 10_000 + gen
            fac = lambda: VecRoomEnv(layout, cfg["eval"]["worlds"], stage["n_agents"], dev, env_cfg, seed=ev_seed)  # noqa: E731
            metrics = evaluate_policy(net, fac, cfg["eval"]["steps"])
            path = os.path.join(run_dir, f"gen_{gen:03d}", f"member_{mid}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(net.checkpoint(opt=tr.opt.state_dict(), hp=hp, gen=gen, member=mid, metrics=metrics,
                                      stage=msg["stage_idx"], env_cfg=env_cfg, n_agents=stage["n_agents"]), path)
            res_q.put(dict(id=mid, gpu=gpu, path=path, metrics=metrics, train=logs, extra=info,
                           seconds=time.time() - t0, mem_mb=torch.cuda.max_memory_allocated(dev) / 1e6))
            del env, tr, net, opp
            torch.cuda.empty_cache()


# ===================================================================== coordinator
def load_cfg(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        cfg.update(population=4, updates_per_generation=3, num_worlds=128)
        cfg["eval"] = dict(worlds=16, steps=200)
        cfg["pbt"].update(distill_every=2, distill_steps=30)
    if args.generations is not None:
        cfg["generations"] = args.generations
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--generations", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume")
    ap.add_argument("--name")
    args = ap.parse_args()

    from roomsim import pbt
    from roomsim.selfplay import Archive, choose_opponent

    if args.resume:
        run_dir = args.resume
        with open(os.path.join(run_dir, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        if args.generations is not None:
            cfg["generations"] = args.generations
        with open(os.path.join(run_dir, "state.json")) as f:
            state = json.load(f)
    else:
        cfg = load_cfg(args)
        run_dir = os.path.join(ROOT, "runs", args.name or time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if args.smoke else ""))
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)
        rng0 = random.Random(cfg["seed"])
        state = dict(gen=0, stage=0, archive=[], best=None,
                     members=[pbt.init_member(i, cfg["ppo"], cfg["policy"]["hidden"], rng0) for i in range(cfg["population"])])

    if args.resume:  # re-score stored elites with the current fitness function (it may have changed)
        from roomsim.ppo import score
        for e in state["archive"] + ([state["best"]] if state["best"] else []):
            if os.path.exists(e["path"]):
                e["score"] = score(torch.load(e["path"], map_location="cpu", weights_only=False)["metrics"])
    rng = random.Random(cfg["seed"] * 7919 + state["gen"])
    archive = Archive(cfg["pbt"]["archive_size"], state["archive"])
    members = state["members"]
    gpus = [g for g in cfg["gpus"] if g < torch.cuda.device_count()] or [0]
    print(f"[train] run_dir={run_dir} gpus={gpus} population={len(members)}")

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(run_dir, "tb"))

    ctx = mp.get_context("spawn")
    res_q = ctx.Queue()
    cmd_qs = [ctx.Queue() for _ in gpus]
    procs = [ctx.Process(target=worker, args=(g, q, res_q, cfg, run_dir), daemon=True) for g, q in zip(gpus, cmd_qs)]
    for p in procs:
        p.start()
    stages = cfg["curriculum"]["stages"]
    try:
        for gen in range(state["gen"], cfg["generations"]):
            t0 = time.time()
            stage = stages[state["stage"]]
            current = [m["ckpt"] for m in members]
            per_gpu = [[] for _ in gpus]
            for m in members:
                per_gpu[m["id"] % len(gpus)].append(dict(
                    id=m["id"], hp=m["hp"], hidden=m["hidden"], ckpt=m["ckpt"], distill_from=m["distill_from"],
                    opponent=choose_opponent(rng, archive, current, cfg["selfplay"]["archive_prob"]),
                    seed=cfg["seed"] * 100_003 + gen * 1000 + m["id"]))
            for q, tasks in zip(cmd_qs, per_gpu):
                q.put(dict(gen=gen, stage=stage, stage_idx=state["stage"], tasks=tasks))
            results = {}
            while len(results) < len(members):
                r = res_q.get(timeout=6 * 3600)
                results[r["id"]] = r
                for p in procs:
                    if not p.is_alive():
                        raise RuntimeError(f"worker {p.pid} died")
            # ---- bookkeeping
            for m in members:
                m["ckpt"], m["distill_from"] = results[m["id"]]["path"], None
            scores = {i: r["metrics"] for i, r in results.items()}
            best_id = max(scores, key=lambda i: scores[i]["score"])
            bm = scores[best_id]
            archive.add([dict(path=results[i]["path"], score=scores[i]["score"], gen=gen, member=i,
                              stage=state["stage"], hidden=next(m["hidden"] for m in members if m["id"] == i))
                         for i in scores])
            better = (state["best"] is None or state["stage"] > state["best"]["stage"]
                      or (state["stage"] == state["best"]["stage"] and bm["score"] > state["best"]["score"]))
            if better:
                state["best"] = dict(path=results[best_id]["path"], score=bm["score"], gen=gen, member=best_id,
                                     stage=state["stage"], metrics=bm)
                shutil.copy(results[best_id]["path"], os.path.join(run_dir, "best.pt"))
            for i, r in results.items():
                for k, v in r["metrics"].items():
                    tb.add_scalar(f"member_{i}/{k}", v, gen)
                tb.add_scalar(f"member_{i}/lr", next(m["hp"]["lr"] for m in members if m["id"] == i), gen)
            for k, v in bm.items():
                tb.add_scalar(f"best/{k}", v, gen)
            tb.add_scalar("curriculum/stage", state["stage"], gen)
            tb.add_scalar("curriculum/n_agents", stage["n_agents"], gen)

            # ---- curriculum: harder room once the best member is good enough
            th = cfg["curriculum"]["threshold"]
            advanced = False
            if (bm["success"] >= th["success"] and bm["collision_rate"] <= th["collision_rate"]
                    and bm["goals_per_min"] >= th.get("goals_per_min", 0.0) and state["stage"] < len(stages) - 1):
                state["stage"] += 1
                advanced = True

            # ---- recursive self-improvement: exploit/explore + periodic distillation from elites
            members, winners, losers = pbt.exploit_explore(members, scores, gen + 1, rng, cfg["pbt"])
            distilled = None
            if cfg["pbt"]["distill_every"] and (gen + 1) % cfg["pbt"]["distill_every"] == 0 and archive.entries:
                worst = min(members, key=lambda m: scores[m["id"]]["score"])
                worst["distill_from"] = [e["path"] for e in archive.top(cfg["pbt"]["distill_teachers"])]
                worst["lineage"] = worst["lineage"][-20:] + [f"g{gen + 1}:distill<-archive"]
                distilled = worst["id"]

            line = dict(gen=gen, seconds=round(time.time() - t0, 1), stage=state["stage"] - int(advanced), n_agents=stage["n_agents"],
                        best_member=best_id, best=bm, winners=winners, losers=losers, distilled=distilled,
                        advanced=advanced, mem_mb={r["gpu"]: round(r["mem_mb"]) for r in results.values()},
                        scores={i: round(s["score"], 3) for i, s in scores.items()})
            with open(os.path.join(run_dir, "history.jsonl"), "a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"[gen {gen:03d}] {line['seconds']:6.1f}s stage={line['stage']} N={stage['n_agents']} "
                  f"best=m{best_id} score={bm['score']:.2f} success={bm['success']:.2f} goals/min={bm['goals_per_min']:.1f} "
                  f"coll={bm['collision_rate']:.3f} | exploit {losers}<-{winners}"
                  + (f" | distill m{distilled}" if distilled is not None else "") + (" | CURRICULUM UP" if advanced else ""),
                  flush=True)
            state.update(gen=gen + 1, members=members, archive=archive.to_list())
            with open(os.path.join(run_dir, "state.json"), "w") as f:
                json.dump(state, f, indent=1)
    finally:
        for q in cmd_qs:
            q.put(None)
        for p in procs:
            p.join(timeout=30)
        tb.close()
    print(f"[train] done. best: {state['best']['path'] if state['best'] else None} -> {os.path.join(run_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
