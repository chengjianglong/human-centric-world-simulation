import math
import random

import numpy as np
import pytest
import torch

from roomsim import gait, pbt, room_spec
from roomsim.env import VecRoomEnv
from roomsim.export import rollout, write_bvh
from roomsim.layout import box_signed_distance, load_layout, push_out_of_boxes
from roomsim.policy import ActorCritic
from roomsim.ppo import PPOTrainer
from roomsim.sensors import lidar
from roomsim.selfplay import Archive, choose_opponent

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def L():
    return load_layout(device=DEV)


def test_spec_valid_and_pois_free(L):
    spec = room_spec.load_spec()
    pois = torch.tensor([[p["x"], p["y"]] for p in room_spec.pois(spec)], device=DEV)
    # every goal point is reachable: not inside furniture and away from walls
    assert (box_signed_distance(pois, L) > 0.1).all()
    assert ((pois > 0.25) & (pois < torch.tensor([L.width - 0.25, L.depth - 0.25], device=DEV))).all()


def test_push_out_of_rotated_box(L):
    # a point at the bookshelf centre (rotated 90 deg) must be pushed clear of it
    i = L.box_names.index("bookshelf")
    p = L.box_c[i].clone().view(1, 2)
    q, hit = push_out_of_boxes(p, 0.25, L)
    assert hit.all()
    assert box_signed_distance(q, L)[0, i] >= 0.25 - 1e-4


def test_lidar_wall_and_furniture_distance(L):
    # facing -y (south): at x=1.5 the ray reaches the wall; at x=2.7 it hits the TV stand (y in 0.1..0.5)
    pos = torch.tensor([[[1.5, 1.2], [2.7, 1.2]]], device=DEV)
    yaw = torch.full((1, 2), -math.pi / 2, device=DEV)
    d = lidar(pos, yaw, L, 0.25, n_rays=16, max_range=5.0)[0, :, 0] * 5.0
    assert abs(d[0].item() - 1.2) < 1e-3
    assert abs(d[1].item() - 0.7) < 1e-3


def test_env_no_penetration_and_determinism(L):
    def run(seed):
        env = VecRoomEnv(L, 64, 4, DEV, seed=seed)
        g = torch.Generator(device=DEV).manual_seed(0)
        for _ in range(100):
            env.step(torch.rand(64, 4, 2, generator=g, device=DEV) * 2 - 1)
        return env
    a, b = run(3), run(3)
    assert torch.equal(a.pos, b.pos)
    r = a.c["radius"]
    assert (box_signed_distance(a.pos, L) > r - 1e-3).all()
    assert (a.pos >= r - 1e-4).all() and (a.pos[..., 0] <= L.width - r + 1e-4).all()
    assert torch.isfinite(a.observe()).all()


def test_ppo_improves_reward(L):
    torch.manual_seed(0)
    env = VecRoomEnv(L, 256, 2, DEV, seed=1)
    net = ActorCritic(env.obs_dim, hidden=128).to(DEV)
    tr = PPOTrainer(net, env, dict(rollout_len=32))
    first = tr.train(2)["mean_reward"]
    last = tr.train(20)["mean_reward"]
    assert last > first


def test_selfplay_masks_opponents(L):
    env = VecRoomEnv(L, 8, 3, DEV, seed=0)
    net = ActorCritic(env.obs_dim, hidden=64).to(DEV)
    opp = ActorCritic(env.obs_dim, hidden=64).to(DEV)
    tr = PPOTrainer(net, env, dict(rollout_len=4, selfplay_frac=0.5), opponent=opp)
    assert tr.learner.sum().item() == 8 * 3 - 4 * 2
    tr.update(tr.collect())


def test_pbt_exploit_explore():
    rng = random.Random(0)
    members = [pbt.init_member(i, dict(lr=3e-4, ent_coef=0.003, clip=0.2, gamma=0.99, lam=0.95), 256, rng) for i in range(8)]
    for m in members:
        m["ckpt"] = f"m{m['id']}.pt"
    res = {i: dict(score=float(i)) for i in range(8)}
    new, winners, losers = pbt.exploit_explore(members, res, 1, rng, dict(exploit_frac=0.25, width_mutation_prob=1.0))
    assert set(winners) == {7, 6} and set(losers) == {0, 1}
    for i in losers:
        assert new[i]["ckpt"] in ("m6.pt", "m7.pt")
        assert new[i]["distill_from"] == [new[i]["ckpt"]]  # width mutated -> distill
        assert 1e-5 <= new[i]["hp"]["lr"] <= 3e-3
    assert new[3] == members[3]


def test_archive_and_opponents():
    ar = Archive(2)
    ar.add([dict(path="a", score=1, stage=0), dict(path="b", score=5, stage=0), dict(path="c", score=0, stage=1)])
    assert [e["path"] for e in ar.entries] == ["c", "b"]
    assert choose_opponent(random.Random(0), Archive(2), [None, None]) is None


def test_gait_symmetry():
    j = gait.joint_angles(np.array([0.3]), np.array([1.3]))[0]
    k = gait.joint_angles(np.array([0.3 + math.pi]), np.array([1.3]))[0]
    names = gait.JOINTS
    assert abs(j[names.index("hip_L")] - k[names.index("hip_R")]) < 1e-9
    still = gait.joint_angles(np.array([1.0]), np.array([0.0]))[0]
    assert abs(still[names.index("hip_L")]) < 1e-9


def test_export_schema(L, tmp_path):
    net = ActorCritic(VecRoomEnv(L, 1, 1, DEV).obs_dim, hidden=64).to(DEV)
    d = rollout(net, L, 3, steps=20, device=DEV)
    assert d["pos"].shape == (21, 3, 2) and d["joints"].shape == (21, 3, len(gait.JOINTS))
    p = tmp_path / "h.bvh"
    write_bvh(p, d, 0)
    txt = p.read_text()
    assert "ROOT pelvis" in txt and "Frames: 21" in txt
    # each motion row has 6 root channels + 3 per other joint
    row = txt.strip().splitlines()[-1].split()
    assert len(row) == 6 + 3 * 13
