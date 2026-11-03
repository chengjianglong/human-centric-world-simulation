"""World mode: house + garden + street (configs/world.json)."""
import os

import pytest
import torch

from worldsim import world_spec
from worldsim.env import VecRoomEnv
from worldsim.export import rollout
from worldsim.layout import box_signed_distance, in_rects, load_layout
from worldsim.policy import ActorCritic

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SPEC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "world.json")


@pytest.fixture(scope="module")
def W():
    return load_layout(SPEC, device=DEV)


def _pt(W, name):
    i = W.poi_names.index(name)
    return W.pois[i:i + 1]


def test_world_spec_valid(W):
    w = world_spec.load_world(SPEC)
    names = {c["name"] for c in world_spec.colliders(w)}
    assert {"wall_south_0", "wall_north_1", "garden_bench", "bus_stop", "fence_side_w_1"} <= names
    # outdoor goal points are clear of obstacles and off the road
    for n in ("bus_stop", "garden_bench", "mailbox", "shed", "front_porch", "garden_lawn"):
        p = _pt(W, n)
        assert (box_signed_distance(p, W) > 0.2).all(), n
        assert not in_rects(p, W.road).any(), n


def test_routes_go_through_doors_and_gate(W):
    nav = W.nav()
    sofa, bus, bench = _pt(W, "sofa"), _pt(W, "bus_stop"), _pt(W, "garden_bench")
    phi, _, _ = nav.potential(sofa, bus, nav.goal_slot(bus))
    assert torch.isfinite(phi).all()
    assert phi.item() > (sofa - bus).norm().item() + 2.0   # detours via the front door + crossing
    # from the street side yard to the garden bench: through the west gate (no other way round)
    side = torch.tensor([[-2.4, 0.6]], device=DEV)
    phi2, _, _ = nav.potential(side, bench, nav.goal_slot(bench))
    assert 9.0 < phi2.item() < 16.0
    # the waypoint from the living room towards the bus stop heads for the front door (south-west)
    wp, _ = nav.waypoint(torch.tensor([[2.5, 1.5]], device=DEV), bus, nav.goal_slot(bus))
    assert wp[0, 1] < 1.5 and wp[0, 0] < 2.5


def test_crossing_preferred(W):
    nav = W.nav()
    a = torch.tensor([[2.0, -5.0]], device=DEV)          # near sidewalk, in front of the crossing
    b = torch.tensor([[2.0, -14.0]], device=DEV)
    c = torch.tensor([[8.5, -5.0]], device=DEV)          # far from the crossing
    d_ab = nav.potential(a, b, nav.goal_slot(b))[0].item()
    d_cb = nav.potential(c, b, nav.goal_slot(b))[0].item()
    assert d_ab < 9.5                                   # straight over the zebra crossing (~9 m)
    assert d_cb > d_ab                                  # crossing elsewhere costs road_cost


def test_world_env_step_and_no_penetration(W):
    env = VecRoomEnv(W, 32, 3, DEV, dict(n_cars=4), seed=0)
    assert env.observe().shape == (32, 3, env.obs_dim) and env.obs_dim == 59
    g = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(150):
        obs, r, d, info = env.step(torch.rand(32, 3, 2, generator=g, device=DEV) * 2 - 1)
    assert torch.isfinite(obs).all() and torch.isfinite(r).all()
    assert (box_signed_distance(env.pos, W) > env.c["radius"] - 1e-3).all()
    s = env.summary()
    assert "car_hit_rate" in s and "jaywalk_rate" in s


def test_cars_yield_to_pedestrian(W):
    env = VecRoomEnv(W, 2, 1, DEV, dict(n_cars=2), seed=0)
    tr = env.traffic
    tr.view[:] = 35.0                                    # attentive drivers
    tr.v[:] = 10.0
    tr.v0[:] = 10.0
    # put eastbound car 1 at x = -20 heading east; a person stands in its lane at x = 0
    tr.s[:, 1] = -20.0 - tr.x0
    tr.s[:, 0] = 0.0
    ped = torch.tensor([[[0.0, -11.25]], [[0.0, -11.25]]], device=DEV)
    for _ in range(100):
        tr.step(0.05, ped)
    front = tr.xy()[:, 1, 0] + tr.half[0]
    assert (front < -0.4).all()                          # stopped short of the person
    assert (tr.v[:, 1] < 0.5).all()


def test_car_hits_are_detected(W):
    env = VecRoomEnv(W, 1, 1, DEV, dict(n_cars=2), seed=0)
    tr = env.traffic
    tr.view[:] = 0.5                                     # a driver who never sees you
    tr.v[:] = 10.0
    tr.v0[:] = 10.0
    tr.s[:, 1] = -8.0 - tr.x0
    env.pos[:] = torch.tensor([0.0, -11.25], device=DEV)
    env.v[:] = 0
    hit = False
    for _ in range(40):
        _, _, _, info = env.step(torch.zeros(1, 1, 2, device=DEV))
        hit |= bool(info["car_hit"].any())
    assert hit and env.summary()["car_hit_rate"] > 0


def test_world_export_has_cars(W):
    env = VecRoomEnv(W, 1, 1, DEV, dict(n_cars=3))
    net = ActorCritic(env.obs_dim, hidden=64).to(DEV)
    d = rollout(net, W, 2, steps=10, device=DEV, env_cfg=dict(n_cars=3))
    assert d["car_pos"].shape == (11, 3, 2) and d["car_yaw"].shape == (11, 3)
    assert str(d["kind"]) == "world" and d["bounds"].shape == (4,)
