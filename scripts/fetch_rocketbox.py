#!/usr/bin/env python3
"""Download selected Microsoft Rocketbox avatars + mocap clips (MIT license) into assets/rocketbox.

    python scripts/fetch_rocketbox.py                       # default cast + walk/idle clips
    python scripts/fetch_rocketbox.py --avatars Male_Adult_02 Female_Adult_07

Only the files needed are fetched (FBX + textures per avatar, a handful of animation FBX),
not the 4 GB repository. Source: https://github.com/microsoft/Microsoft-Rocketbox
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

REPO = "microsoft/Microsoft-Rocketbox"
API = f"https://api.github.com/repos/{REPO}/contents/"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "assets", "rocketbox")

DEFAULT_AVATARS = ["Male_Adult_01", "Female_Adult_01", "Male_Adult_05", "Female_Adult_04",
                   "Male_Adult_08", "Female_Adult_06"]
# in-place idles come from the *_static set; walks (with root motion, made in-place in Blender) from *_xy
CLIPS = {
    "static": ["m_idle_neutral_01", "f_idle_neutral_01"],
    "xy": ["m_walk_slow_01", "m_walk_neutral_01", "m_walk_fast_01",
           "f_walk_slow_01", "f_walk_neutral_01", "f_walk_fast_01"],
}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "human-centric-world-simulation-fetch"})
    if "api.github.com" in url and os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def listdir(path):
    return json.loads(_get(API + urllib.parse.quote(path)))


def download(path, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = _get(RAW + urllib.parse.quote(path))
    if data[:20].startswith(b"version https://git-lfs"):
        sys.exit(f"{path} is a Git LFS pointer; download the repo with git-lfs instead")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return True


def avatar_dir(name):
    group = "Children" if "Child" in name else "Professions" if not name.split("_")[1] == "Adult" else "Adults"
    return f"Assets/Avatars/{group}/{name}"


def fetch_avatar(name):
    base = avatar_dir(name)
    out = os.path.join(DEST, "avatars", name)
    files = [(f"{base}/Export/{name}.fbx", os.path.join(out, "Export", f"{name}.fbx"))]
    for e in listdir(f"{base}/Textures"):
        files.append((e["path"], os.path.join(out, "Textures", e["name"])))
    n = sum(download(p, d) for p, d in files)
    print(f"  {name}: {len(files)} files ({n} new)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatars", nargs="*", default=DEFAULT_AVATARS)
    ap.add_argument("--list", action="store_true", help="list available adult avatars and exit")
    a = ap.parse_args()
    if a.list:
        print("\n".join(e["name"] for e in listdir("Assets/Avatars/Adults")))
        return
    os.makedirs(DEST, exist_ok=True)
    download("LICENSE.md", os.path.join(DEST, "LICENSE.md"))
    print("avatars:")
    for name in a.avatars:
        fetch_avatar(name)
    print("animations:")
    for kind, names in CLIPS.items():
        for c in names:
            p = f"Assets/Animations/all_animations_max_motextr_{kind}/{c}.max.fbx"
            new = download(p, os.path.join(DEST, "animations", f"{c}.fbx"))
            print(f"  {c} ({kind}){' new' if new else ''}")
    print(f"done -> {DEST}")


if __name__ == "__main__":
    main()
