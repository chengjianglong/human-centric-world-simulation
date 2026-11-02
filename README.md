# human-centric-world-simulation: digital humans learning to walk around a 3D room

Several digital humans share a 3D room built in Blender. Their walking policies are trained
in PyTorch with PPO across two GPUs, and a *recursive self-improvement* loop improves them
generation by generation (population-based training, self-play against elite past policies,
distillation of the best policies into new students, and an automatic curriculum). The
learned trajectories go back into Blender and are rendered as videos or saved as `.blend`
and `.bvh` files.

```
configs/room.json ──► blender/build_room.py ──► assets/room.blend / room.glb   (visuals)
        │
        └──────────► roomsim/room_spec.py ──► roomsim/env.py (GPU physics: same colliders)
                                                     │
train.py: PBT coordinator ── worker cuda:0 ──┐       ▼
                         └── worker cuda:1 ──┴─► PPO + self-play + distillation ─► runs/<id>/best.pt
                                                                                    │
evaluate.py ─► renders/rollout.npz (+ .bvh) ─► blender/render_rollout.py ─► .mp4 + animated .blend
```

## Make it *your* room
Everything about the room comes from one file, **`configs/room.json`**. The physics the
agents learn in and the Blender visuals are both generated from it, so they always match.
Measure your room and edit:
- `width`, `depth`, `height`: interior size in metres. The origin is the inside south-west corner,
  x runs along the south wall, y runs into the room.
- `openings`: doors and windows, as wall + offset from the wall's start corner + width/height/sill.
- `furniture`: per piece, `type` (sofa, table, chair, tv_stand, bookshelf, plant, lamp, rug,
  or any other name for a plain box), the footprint `center`, `size` = [width, depth, height],
  `rot` (0 = front faces south/-y), `color`, and an optional `poi` goal point in front of it.
- `materials`, `lights`: floor, wall and ceiling colours, plus the sun direction.

Then run `blender -b -P blender/build_room.py` (it validates that everything fits) and retrain.
For photorealism, pass a phone LiDAR scan of the real room (Polycam, Scaniverse, etc.,
exported as `.glb` or `.obj`, aligned to the same origin) via `render_rollout.py --room-scan scan.glb`.
The scan replaces the procedural visuals while the measured `room.json` stays the physics.

## Quick start
```bash
./scripts/install_blender.sh                              # portable Blender 4.2 LTS -> tools/blender
tools/blender/blender -b -P blender/build_room.py          # assets/room.blend, room.glb, room_layout.json
tools/blender/blender -b -P blender/build_human.py         # assets/human.blend/.glb (simple mannequin rig)
python3 scripts/fetch_rocketbox.py                         # photo-realistic avatars + mocap (~540 MB, MIT)

.venv/bin/python -m pytest -q tests                       # 10 tests
.venv/bin/python train.py --smoke --generations 2         # ~1 min check on both GPUs
.venv/bin/python train.py --name main                     # full run (see configs/default.yaml)
.venv/bin/tensorboard --logdir runs/main/tb               # curves per member, best, curriculum stage
.venv/bin/python train.py --resume runs/main --generations 100   # continue later

.venv/bin/python evaluate.py --ckpt runs/main/best.pt --agents 4 --out renders/rollout.npz --bvh
tools/blender/blender -b -P blender/render_rollout.py -- --rollout renders/rollout.npz --camera corner
#   --camera corner|top|follow  --engine cycles|eevee  --res 1920x1080  --frames 300  --still 120
#   --humans rocketbox|mannequin  --avatars Female_Adult_01 Male_Adult_05 ...   (default: rocketbox if downloaded)
```
Each render also saves an animated `.blend` next to the video, so you can open it in Blender and scrub the timeline.

## How it works
**Environment (`roomsim/env.py`)**: B rooms × N humans simulated together as tensors on the GPU.
Each human is a 0.25 m disc with unicycle dynamics (acceleration and turn-rate control,
max 1.5 m/s). Collisions with rotated furniture boxes, walls and other humans are
resolved every step. Observations (43 numbers):
16-ray 360° lidar, goal in the body frame, own velocity, room pose, and the 3 nearest humans.
Goals are named points (sofa, TV, bookshelf, door, ...) or random free spots. The reward
pays for progress and arrivals, and penalises bumps, personal-space intrusion (< 0.8 m)
and jerky control.

**Policy and learning (`policy.py`, `ppo.py`)**: an actor-critic MLP with observation normalisation built in.
PPO with GAE, shared by every human a member controls. In half of the rooms, the other
humans are a *frozen opponent* (`selfplay.py`) drawn from the elite archive or the current population.

**Recursive self-improvement (`train.py`, `pbt.py`, `distill.py`)**: each generation:
1. Each of the 8 members trains with PPO (4 per GPU, one worker process per GPU).
2. Each member is scored on a fixed-seed benchmark (`score = goals/min × success − 80 × collision rate`).
3. Exploit/explore: the bottom 25% copy the top 25%'s weights and mutate lr, entropy, clip,
   γ and λ. Sometimes the network width changes too, and the parent is then distilled into the new size.
4. The best policies enter an elite archive. They become self-play opponents and teachers.
5. Every 5 generations, the worst member is reborn as a student distilled from the top-3
   elites (DAgger-style), and then keeps improving with RL.
6. Curriculum: once the best member reaches ≥90% success, ≤4% collisions and ≥6 goals/min,
   the room gets busier (2 → 3 → 4 → 6 humans, farther goals).

Everything is checkpointed (`runs/<id>/gen_XXX/member_k.pt`, `state.json`, `history.jsonl`) and resumable.

**Photo-realistic digital humans (`blender/rocketbox.py`)**: [Microsoft Rocketbox](https://github.com/microsoft/Microsoft-Rocketbox)
avatars are rigged, textured, scan-quality characters under the MIT license. Six are fetched by default.
List more with `fetch_rocketbox.py --list` and add them with `--avatars`. They are animated with Rocketbox's own **motion capture**:
- The trained policy drives each human's root (position and heading). The limbs come from mocap clips layered in Blender's NLA:
  `idle < walk_slow < walk_neutral < walk_fast`. The blend weights follow the policy's speed (0.8 / 1.0 / 1.5 m/s clips).
- All walk clips share one gait phase that advances by *distance travelled ÷ stride length*, so feet stay planted.
  Clips are made in-place and phase-aligned automatically from the foot oscillation. Walking backwards plays the cycle in reverse.
- Clip and avatar skeletons have different rest poses (arms-down vs T-pose). Each bone is retargeted with
  `q_avatar = (L_avatar⁻¹ · L_clip) · q_clip`, and root height is scaled to the avatar's body.
- Materials are rebuilt from the textures (colour, normal and specular maps, skin subsurface, alpha hair cards). Use Cycles for realism.

`--humans mannequin` uses the simple procedural rig instead (`build_human.py`, `skeleton.py`, `gait.py`).
`evaluate.py --bvh` writes that skeleton's motion as BVH for retargeting onto other characters.

## Layout
```
configs/   room.json (the room), default.yaml (training)
blender/   build_room.py, build_human.py, rocketbox.py, render_rollout.py, common.py
roomsim/   room_spec, layout, sensors, env, gait, skeleton, policy, ppo, pbt, selfplay, distill, export
train.py   evaluate.py   tests/   scripts/install_blender.sh, scripts/fetch_rocketbox.py
```

## Notes
- Hardware: 2× RTX A2000 (6 GB). Each worker peaks at about 0.6 GB, and a generation takes about 3 min
  (8 members × 40 updates × 1024 rooms). GPU utilisation is about 30% because the small
  simulation kernels are launch-bound. Raising `num_worlds` gives more samples nearly for
  free, and `torch.compile`/CUDA graphs on `env.step` is the next speed-up.
- `download.blender.org` blocks scripted downloads, so the installer falls back to official mirrors and checks SHA-256.
