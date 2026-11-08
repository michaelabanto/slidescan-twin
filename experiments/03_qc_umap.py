"""Challenge: find the bad tiles in a scan without a human looking at every one.

Five tissue types, every 20th tissue tile defocused (0.5, 1, 2 or 4 DOF beyond the
in-focus band), about 2% of tiles with the backend's own debris overlay. Off-tissue
tiles are gated out first. UMAP is for looking, an isolation forest flags.

    python experiments/03_qc_umap.py --out results          # 1125 tiles, about 3 min
    python experiments/03_qc_umap.py --out results --quick  # 2 tissue types, coarse grid
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidescan_twin import open_scope, qc_features, snake_grid, tissue_fraction  # noqa: E402

INK, GREY, RED = "#222222", "#B0B0B0", "#C0392B"
LEVELS = [0.5, 1, 2, 4]
TISSUES = [("glandular", 1), ("epithelium", 2), ("connective", 3), ("adipose", 4), ("mixed", 5)]


def collect(tissues, step, every=20, frac_debris=.02, seed=5):
    from virtual_microscope.pipeline.debris_overlay import add_debris

    rng = np.random.default_rng(seed)
    X, lab, thumbs, world, pos = [], [], [], [], []
    k = 0
    for w, (tissue, sd) in enumerate(tissues):
        core, sim = open_scope(tissue, sd); sim.tissue_z = 0
        for xy in snake_grid(step=step, lim=840):
            core.setXYPosition(*xy); core.setPosition(0); im = core.snap(); l = "clean"
            if tissue_fraction(im) < .3:
                l = "background"
            elif k % every == 0:
                lv = LEVELS[(k // every) % 4]
                core.setPosition(sim._dof / 2 + lv * sim._dof); im = core.snap(); l = f"{lv} DOF"
            elif rng.random() < frac_debris:
                im = add_debris(im, density=0.0004, size_range=(3, 15), defocus_fraction=.3, rng_seed=int(rng.integers(1e6))); l = "debris"
            k += 1
            X.append(qc_features(im)); lab.append(l); thumbs.append(cv2.resize(im, (128, 128))); world.append(w); pos.append(xy)
    return np.array(X), np.array(lab), thumbs, np.array(world), pos


def score(X, lab):
    import umap
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(X)
    emb = umap.UMAP(n_neighbors=15, min_dist=.1, random_state=1).fit_transform(X)
    bad = np.isin(lab, [f"{l} DOF" for l in LEVELS] + ["debris"]); tis = lab != "background"
    iso = IsolationForest(contamination=max(bad.sum(), 1) / tis.sum(), random_state=0).fit(X[tis])
    flag = np.zeros(len(lab), bool); flag[tis] = iso.predict(X[tis]) == -1
    cats = [f"{l} DOF" for l in LEVELS] + ["debris"]
    recall = {c: float(flag[lab == c].mean()) if (lab == c).any() else None for c in cats}
    return emb, flag, recall, float(flag[lab == "clean"].mean())


def plot_umap(emb, lab, recall, fpr, out):
    cats = [f"{l} DOF" for l in LEVELS] + ["debris"]; shades = ["#F5B7B1", "#E6776B", "#C0392B", "#7B241C"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    a = ax[0]
    a.scatter(*emb[lab == "clean"].T, s=5, c=GREY, lw=0, label="clean tissue")
    a.scatter(*emb[lab == "background"].T, s=5, c="#7FB3D5", lw=0, label="off tissue (gated out)")
    for c, sh in zip(cats[:4], shades): a.scatter(*emb[lab == c].T, s=26, c=sh, lw=0, label=c)
    a.scatter(*emb[lab == "debris"].T, s=40, c=INK, marker="x", label="debris")
    a.set_xticks([]); a.set_yticks([]); a.legend(frameon=False, fontsize=10); a.set_title(f"UMAP of {len(lab)} virtual-microscope tiles")
    for s in a.spines.values(): s.set_visible(False)
    b = ax[1]; vals = [100 * (recall[c] or 0) for c in cats]
    bars = b.bar(cats, vals, color=shades + [INK])
    for bb, v in zip(bars, vals): b.text(bb.get_x() + bb.get_width() / 2, v + 2, f"{v:.0f}%", ha="center")
    b.set_ylim(0, 110); b.set_ylabel("flagged (%)"); b.set_title(f"Outlier model on tissue tiles (false alarms {fpr*100:.1f}%)", fontsize=12)
    for sp in ("top", "right"): b.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "qc_umap.png", dpi=200); plt.close(fig)


def plot_montage(lab, flag, thumbs, out, n=6):
    groups = [("caught: defocus", flag & np.char.endswith(lab, "DOF") & (lab != "0.5 DOF")), ("caught: debris", flag & (lab == "debris")),
              ("missed: 0.5 DOF", ~flag & (lab == "0.5 DOF")), ("false alarm", flag & (lab == "clean")), ("gated: off tissue", lab == "background")]
    fig, ax = plt.subplots(len(groups), n, figsize=(n * 1.6, len(groups) * 1.75))
    for r, (name, mask) in enumerate(groups):
        idx = np.where(mask)[0]
        sel = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)] if len(idx) else []
        for c in range(n):
            a = ax[r, c]; a.axis("off")
            if c < len(sel): a.imshow(thumbs[sel[c]]); a.set_title(lab[sel[c]], fontsize=9, color="#777")
        ax[r, 0].text(-0.08, .5, name, transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=12, weight="bold",
                      color=RED if name.startswith(("missed", "false")) else INK)
    plt.subplots_adjust(left=.2, wspace=.05, hspace=.35); fig.savefig(out / "qc_flagged.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def scan_movie(emb, lab, flag, thumbs, world, pos, out, which):
    idx = np.where(world == which)[0]
    xs = sorted({pos[i][0] for i in idx}); ys = sorted({pos[i][1] for i in idx}); T = 40
    mosaic = np.full((len(ys) * T, len(xs) * T, 3), 255, np.uint8)
    writer = imageio.get_writer(out / "qc_movie.mp4", fps=15, codec="libx264", quality=7, macro_block_size=8)
    frame = None
    for step in range(0, len(idx) + 1, 3):
        for i in idx[max(0, step - 3):step]:
            cx, cy = xs.index(pos[i][0]), ys.index(pos[i][1])
            mosaic[cy * T:(cy + 1) * T, cx * T:(cx + 1) * T] = cv2.resize(thumbs[i][34:94, 34:94], (T, T))  # centre crop = step size
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
        ax[0].imshow(mosaic); ax[0].axis("off"); ax[0].set_title(f"virtual scan: {step} tiles")
        for i in idx[:step]:
            if flag[i]:
                cx, cy = xs.index(pos[i][0]), ys.index(pos[i][1]); ax[0].add_patch(Rectangle((cx * T, cy * T), T, T, fill=False, ec=RED, lw=2))
        seen = idx[:step]; other = np.setdiff1d(np.arange(len(lab)), idx)
        ax[1].scatter(*emb[other].T, s=3, c="#E5E5E5", lw=0)
        ax[1].scatter(*emb[seen][~flag[seen]].T, s=6, c="#9A9A9A", lw=0); ax[1].scatter(*emb[seen][flag[seen]].T, s=30, c=RED, lw=0)
        ax[1].set_xticks([]); ax[1].set_yticks([]); [s.set_visible(False) for s in ax[1].spines.values()]
        ax[1].set_title(f"UMAP: {int(flag[seen].sum())} flagged for review", color=RED)
        fig.tight_layout(); fig.canvas.draw(); frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
        writer.append_data(frame)
    for _ in range(20): writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("results")); p.add_argument("--quick", action="store_true")
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    tissues = TISSUES[-2:] if a.quick else TISSUES
    X, lab, thumbs, world, pos = collect(tissues, step=240 if a.quick else 120)
    emb, flag, recall, fpr = score(X, lab)
    plot_umap(emb, lab, recall, fpr, a.out); plot_montage(lab, flag, thumbs, a.out)
    scan_movie(emb, lab, flag, thumbs, world, pos, a.out, which=len(tissues) - 1)
    res = {"recall": recall, "false_alarms": fpr, "counts": {c: int((lab == c).sum()) for c in sorted(set(lab))}}
    json.dump(res, open(a.out / "qc_umap.json", "w"), indent=1); print(json.dumps(res, indent=1))
