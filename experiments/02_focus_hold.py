"""Challenge: keep a 600-tile snake scan in focus while the slide drifts and gets bumped.

The strategies only see images and only move z. Ground truth (``sim.tissue_z``) is set
by the experiment and used for scoring, never by the strategies.

    python experiments/02_focus_hold.py --out results          # full run, about 3 to 4 min
    python experiments/02_focus_hold.py --out results --quick  # 120 tiles, for testing
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidescan_twin import open_scope, sharpness, snake_grid, tissue_fraction  # noqa: E402

INK, GREY, RED, BLUE = "#222222", "#B0B0B0", "#C0392B", "#2F6DB5"
STRATEGIES = {
    "none": "no correction",
    "af": "13-plane AF when sharpness drops",
    "nudge": "estimate |dz| from sharpness, probe for sign, nudge",
    "oracle": "trained estimator stand-in (true dz + noise), nudge",
}


def run_experiment(n_tiles=600, bump_every=80, seed=3):
    core, sim = open_scope("mixed", seed=1)
    half = sim._dof / 2                       # "in focus" = within half a depth of field
    path = snake_grid()
    rng = np.random.default_rng(seed)
    # ground truth: slow drift (µm per tile) plus a 10-16 µm bump every `bump_every` tiles
    drift = np.cumsum(rng.normal(0.03, 0.15, 4 * n_tiles))
    for k in range(bump_every, 4 * n_tiles, bump_every):
        drift[k:] += rng.choice([-1, 1]) * rng.uniform(10, 16)

    # calibration measured in the twin: normalised sharpness vs |dz| on a few tissue tiles
    zs = np.arange(0, 31, 1.0); cal = []
    for xy in path[:40:8]:
        core.setXYPosition(*xy); sim.tissue_z = 0
        if tissue_fraction(core.snap()) < .3:
            continue
        c = []
        for z in zs:
            core.setPosition(z); c.append(sharpness(core.snap()))
        cal.append(np.array(c) / c[0])
    cal = np.median(cal, 0); core.setPosition(0)

    def abs_dz_from_ratio(ratio):
        mono = np.minimum.accumulate(cal)
        return float(np.interp(-ratio, -mono, zs))

    def run(mode):
        z, t, snaps, errors, frames, ref = 0.0, 0, 0, [], [], []
        for i in range(n_tiles):
            sim.tissue_z = drift[t]
            core.setXYPosition(*path[i % len(path)]); core.setPosition(z)
            im = core.snap(); snaps += 1; t += 1
            err = z - sim.tissue_z; errors.append(err)
            if mode == "oracle":
                if i % 5 == 0: frames.append((im, err, i))
                # stand-in for a trained single-shot estimator (the role Pinkard et al. 2019 fill)
                z -= np.clip(0.7 * (err + rng.normal(0, 1.0)), -8, 8)
                continue
            if mode == "none" or tissue_fraction(im) < .3:   # only act on tissue
                continue
            s = sharpness(im); ref.append(s)
            r = s / np.median(ref[-30:]) if len(ref) > 5 else 1
            if mode == "af" and r < .7:
                sweep = []
                for dz in np.arange(-18, 19, 3):                # 13-plane focal sweep
                    core.setPosition(z + dz); sweep.append((sharpness(core.snap()), z + dz)); snaps += 1; t += 1
                    sim.tissue_z = drift[t]
                top = max(v for v, _ in sweep)
                z = float(np.mean([p for v, p in sweep if v >= .97 * top])); ref.clear()   # centre of the sharp plateau
            elif mode == "nudge" and r < .8:
                mag = abs_dz_from_ratio(min(r, 1.0))
                core.setPosition(z + 4); s2 = sharpness(core.snap()); snaps += 1; t += 1   # probe for direction
                if abs(s2 - s) < .03 * s:                                               # ambiguous: probe other side
                    core.setPosition(z - 4); s3 = sharpness(core.snap()); snaps += 1; t += 1
                    sign = 1 if s2 > s3 else -1
                else:
                    sign = 1 if s2 > s else -1
                z = z + sign * min(max(mag, 2), 15); ref.clear()
        errors = np.array(errors)
        return errors, dict(in_focus=float((np.abs(errors) <= half).mean()), extra_snaps=(snaps - n_tiles) / n_tiles), frames

    results = {m: run(m) for m in STRATEGIES}
    return results, half, n_tiles


def plot(results, half, n, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhspan(-half, half, color="#EEEEEE", zorder=0)
    styles = {"none": (GREY, "-", 1.2), "af": (INK, "-", 1.2), "nudge": (RED, "-", 1.4), "oracle": (BLUE, "--", 1.4)}
    for m, (c, ls, lw) in styles.items():
        e, s, _ = results[m]
        ax.plot(e, color=c, ls=ls, lw=lw, label=f"{STRATEGIES[m]}: {s['in_focus']*100:.0f}% in focus, +{s['extra_snaps']*100:.0f}% snaps")
    ax.set_xlabel("tile (snake scan)"); ax.set_ylabel("focus error (µm)"); ax.set_ylim(-40, 40)
    ax.text(n * .01, half + 1, f"in focus (±{half:g} µm, half DOF at 10x)", fontsize=11, color="#777")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.17), fontsize=11)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "focus_hold.png", dpi=200); plt.close(fig)


def movie(results, half, n, out):
    writer = imageio.get_writer(out / "focus_movie.mp4", fps=12, codec="libx264", quality=7, macro_block_size=8)
    base, errs = results["none"][0], results["oracle"][0]
    for im, err, i in results["oracle"][2]:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
        ax[0].imshow(im); ax[0].axis("off")
        ax[0].set_title(f"tile {i}   error {err:+.1f} µm", color=RED if abs(err) > half else INK)
        ax[1].axhspan(-half, half, color="#EEEEEE"); ax[1].plot(base[:i + 1], color=GREY, lw=1); ax[1].plot(errs[:i + 1], color=BLUE, lw=1.4)
        ax[1].set_xlim(0, n); ax[1].set_ylim(-30, 30); ax[1].set_xlabel("tile"); ax[1].set_ylabel("focus error (µm)")
        ax[1].set_title("estimate defocus, nudge z\n(estimator = stand-in for a trained network)", fontsize=12)
        fig.tight_layout(); fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("results")); p.add_argument("--quick", action="store_true")
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    res, half, n = run_experiment(n_tiles=120 if a.quick else 600, bump_every=40 if a.quick else 80)
    plot(res, half, n, a.out); movie(res, half, n, a.out)
    scores = {m: v[1] for m, v in res.items()}
    json.dump(scores, open(a.out / "focus_hold.json", "w"), indent=1); print(json.dumps(scores, indent=1))
