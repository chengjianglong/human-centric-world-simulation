#!/usr/bin/env bash
# Re-create the README demo media in docs/media from trained checkpoints.
#   scripts/make_demo_media.sh [world_ckpt] [room_ckpt]
# Needs: tools/blender, Rocketbox avatars, `pip install imageio-ffmpeg matplotlib` in .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
WORLD_CKPT=${1:-runs/world/best.pt}
ROOM_CKPT=${2:-runs/main/best.pt}
PY=.venv/bin/python
BL=tools/blender/blender
FF=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
OUT=docs/media
TMP=renders/demo
mkdir -p "$OUT" "$TMP"

gif() {  # gif <video> <out.gif> <start s> <length s> <width>
  "$FF" -v error -y -ss "$3" -t "$4" -i "$1" -vf \
    "fps=12,scale=$5:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "$2"
}
mp4() {  # mp4 <video> <out.mp4>: small web-friendly copy
  "$FF" -v error -y -i "$1" -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -movflags +faststart -an "$2"
}

# --- world: one rollout, four cameras -------------------------------------------------------
$PY evaluate.py --ckpt "$WORLD_CKPT" --steps 400 --out $TMP/world.npz
for cam in cutaway street aerial garden; do
  res=960x540; [ $cam = cutaway ] && res=1280x720   # the hero clip gets full HD-ish, the gallery clips less
  $BL -b -P blender/render_rollout.py -- --rollout $TMP/world.npz --camera $cam --engine eevee --res $res --out $TMP/$cam.mp4
  mp4 $TMP/$cam.mp4 $OUT/world_$cam.mp4
done
$BL -b -P blender/render_rollout.py -- --rollout $TMP/world.npz --camera top --engine eevee --still 200 --out $TMP/top.mp4
"$FF" -v error -y -i $TMP/top_f200.png -q:v 3 $OUT/world_top.jpg
gif $TMP/cutaway.mp4 $OUT/world_cutaway.gif 7 10 720
gif $TMP/street.mp4  $OUT/world_street.gif 11 8 480
gif $TMP/garden.mp4  $OUT/world_garden.gif 2 8 480
gif $TMP/aerial.mp4  $OUT/world_aerial.gif 8 8 480

# --- room: Rocketbox avatars, Cycles ----------------------------------------------------------
$PY evaluate.py --ckpt "$ROOM_CKPT" --steps 300 --out $TMP/room.npz
$BL -b -P blender/render_rollout.py -- --rollout $TMP/room.npz --camera corner --engine cycles --out $TMP/room.mp4
mp4 $TMP/room.mp4 $OUT/room_corner.mp4
gif $TMP/room.mp4 $OUT/room.gif 2 8 480

# --- training curves --------------------------------------------------------------------------
$PY scripts/plot_training.py --log "runs/main.log:Room mode" --log "runs/world.log:World mode" --out $OUT/training.png
ls -lh $OUT
