#!/usr/bin/env bash
# Download portable Blender 4.2 LTS into tools/blender (no root required).
# download.blender.org sits behind a bot challenge, so official mirrors are tried too.
set -euo pipefail
VER="${BLENDER_VERSION:-4.2.3}"
MAJ="${VER%.*}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/tools"
FILE="blender-${VER}-linux-x64.tar.xz"
MIRRORS=(
  "https://download.blender.org/release"
  "https://mirrors.ocf.berkeley.edu/blender/release"
  "https://ftp.halifax.rwth-aachen.de/blender/release"
)
mkdir -p "$DEST"
if [ -x "$DEST/blender/blender" ]; then echo "Blender already installed: $DEST/blender"; exit 0; fi
ok=0
for m in "${MIRRORS[@]}"; do
  base="$m/Blender${MAJ}"
  echo "Trying $base/$FILE"
  if curl -fL --retry 3 -o "$DEST/$FILE" "$base/$FILE" && curl -fsL -o "$DEST/sha256" "$base/blender-${VER}.sha256"; then
    (cd "$DEST" && grep " $FILE\$" sha256 | sha256sum -c -) && ok=1 && break
  fi
  echo "  failed, trying next mirror"
done
[ "$ok" = 1 ] || { echo "Could not download/verify Blender ${VER}"; exit 1; }
tar -xJf "$DEST/$FILE" -C "$DEST"
mv "$DEST/blender-${VER}-linux-x64" "$DEST/blender"
rm -f "$DEST/$FILE" "$DEST/sha256"
"$DEST/blender/blender" --version | head -1
