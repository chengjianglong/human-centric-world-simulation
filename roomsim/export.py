"""Roll out a trained policy in one room and export trajectories for Blender (.npz) and BVH."""
import math
import os

import numpy as np
import torch

from . import gait
from .env import VecRoomEnv
from .policy import ActorCritic
from .skeleton import BONE_DRIVE, BONES


@torch.no_grad()
def rollout(net: ActorCritic, layout, n_agents, steps=600, seed=0, device="cuda", env_cfg=None, deterministic=True):
    env = VecRoomEnv(layout, 1, n_agents, device, env_cfg, seed=seed)
    net.eval()
    frames = [env.snapshot()]
    events = []
    obs = env.observe()
    for t in range(steps):
        a, _, _ = net.act(obs, deterministic=deterministic)
        obs, _, _, info = env.step(a)
        frames.append(env.snapshot())
        for n in info["reached"][0].nonzero().flatten().tolist():
            events.append((t + 1, n, "reached"))
        if (info["hit"] | info["agent_hit"])[0].any():
            events.append((t + 1, -1, "contact"))
    cat = lambda k: torch.stack([f[k][0] for f in frames]).cpu().numpy()  # noqa: E731
    pos, yaw, v, phase = cat("pos"), cat("yaw"), cat("v"), cat("phase")
    # smooth the gait phase/speed for animation (sim speed can change abruptly on contact)
    joints = gait.joint_angles(phase, v).astype(np.float32)
    return dict(pos=pos.astype(np.float32), yaw=yaw.astype(np.float32), speed=v.astype(np.float32),
                phase=phase.astype(np.float32), goal=cat("goal").astype(np.float32), goal_idx=cat("goal_idx"),
                joints=joints, joint_names=np.array(gait.JOINTS), dt=np.float32(env.c["dt"]),
                radius=np.float32(env.c["radius"]), poi_names=np.array(layout.poi_names),
                reached=np.array([e for e in events if e[2] == "reached"], dtype=object),
                summary=np.array(env.summary(), dtype=object))


def save_npz(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **data)


# ---------------------------------------------------------------- BVH
def _bvh(p):  # world (x, y, z-up) -> BVH (x, y-up, z) with the character's front (-y) as +z
    return (p[0], p[2], -p[1])


def write_bvh(path, data, agent):
    """Write one human's motion as a BVH file (metres, Y-up, 1/dt fps)."""
    bones = {b[0]: b for b in BONES}
    children = {}
    for name, parent, *_ in BONES:
        children.setdefault(parent, []).append(name)
    lines, order = ["HIERARCHY"], []

    def emit(name, depth):
        _, parent, head, tail, _ = bones[name]
        ind = "  " * depth
        off = head if parent is None else tuple(h - ph for h, ph in zip(head, bones[parent][2]))
        if parent is None:
            off = (0.0, 0.0, 0.0)
        lines.append(f"{ind}{'ROOT' if parent is None else 'JOINT'} {name}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}  OFFSET {' '.join(f'{x:.4f}' for x in _bvh(off))}")
        if parent is None:
            lines.append(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation")
        else:
            lines.append(f"{ind}  CHANNELS 3 Yrotation Xrotation Zrotation")
        order.append(name)
        kids = children.get(name, [])
        for k in kids:
            emit(k, depth + 1)
        if not kids:
            end = tuple(t - h for t, h in zip(tail, head))
            lines.append(f"{ind}  End Site")
            lines.append(f"{ind}  {{")
            lines.append(f"{ind}    OFFSET {' '.join(f'{x:.4f}' for x in _bvh(end))}")
            lines.append(f"{ind}  }}")
        lines.append(f"{ind}}}")

    emit("pelvis", 0)
    T = data["pos"].shape[0]
    J = {n: i for i, n in enumerate(data["joint_names"].tolist())}
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {float(data['dt']):.6f}"]
    pelvis_h = bones["pelvis"][2][2]
    for t in range(T):
        ang = data["joints"][t, agent]
        x, y = data["pos"][t, agent]
        root_y = math.degrees(float(data["yaw"][t, agent]) + math.pi / 2)
        row = [x, pelvis_h + ang[J["pelvis_bob"]], -y, root_y, 0.0, 0.0]
        for name in order[1:]:
            ry = rx = 0.0
            if name in BONE_DRIVE:
                joint, sign = BONE_DRIVE[name]
                rx = math.degrees(sign * ang[J[joint]])
            elif name == "spine":
                ry, rx = math.degrees(ang[J["spine_twist"]]), math.degrees(0.04)
            row += [ry, rx, 0.0]
        lines.append(" ".join(f"{v:.4f}" for v in row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
