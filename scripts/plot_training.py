"""Plot PBT training curves from one or more train.py logs as small multiples.

    python scripts/plot_training.py --log "runs/main.log:Room mode" --log "runs/world.log:World mode" \
        --out docs/media/training.png
"""
import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GEN = re.compile(r"^\[gen (\d+)\].*?N=(\d+).*?success=([\d.]+) goals/min=([\d.]+) coll=([\d.]+)"
                 r"(?: cars=(\d+) car_hit=([\d.]+))?")

PANELS = [  # key, title, y-format
    ("success", "Goal success rate", "{:.0%}"),
    ("goals", "Goals reached per minute", "{:.0f}"),
    ("coll", "Collision rate (per step)", "{:.0%}"),
    ("car_hit", "Car-hit rate (per step)", "{:.2%}"),
    ("people", "People in the scene", "{:.0f}"),
    ("cars", "Cars on the street", "{:.0f}"),
]


def parse(path):
    rows = []
    for line in open(path):
        m = GEN.match(line)
        if m:
            g, n, s, gm, c, cars, hit = m.groups()
            rows.append(dict(gen=int(g), people=int(n), success=float(s), goals=float(gm), coll=float(c),
                             cars=int(cars) if cars else None, car_hit=float(hit) if hit else None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True, help="path[:label]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = []
    for spec in a.log:
        path, _, label = spec.partition(":")
        runs.append((label or path, parse(path)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), dpi=150, facecolor=SURFACE)
    for ax, (key, title, fmt) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        for (label, rows), color in zip(runs, SERIES):
            pts = [(r["gen"], r[key]) for r in rows if r[key] is not None]
            if not pts:
                continue
            x, y = zip(*pts)
            if key in ("people", "cars"):
                ax.step(x, y, where="post", color=color, lw=2, label=label)
            else:
                ax.plot(x, y, color=color, lw=2, label=label, solid_capstyle="round")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, f=fmt: f.format(v)))
        if key == "success":
            ax.set_ylim(0.5, 1.02)
        elif key in ("coll", "car_hit"):
            ax.set_ylim(bottom=0)
        ax.set_xlabel("generation", fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=10,
               labelcolor=INK, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Population-based training: best member per generation", x=0.01, ha="left",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, facecolor=SURFACE)
    print(f"[plot] {sum(len(r) for _, r in runs)} generations -> {a.out}")


if __name__ == "__main__":
    main()
