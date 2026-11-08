"""Teaching twin: turn up the camera gain and watch the histogram saturate.

Backend: hemocytometer (counting-chamber grid). Output: an MP4 and a poster frame.
    python experiments/01_exposure_teaching.py --out results
"""
import argparse
import logging
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.disable(logging.CRITICAL)


def main(out: Path, quick: bool):
    from virtual_microscope.backends import load_backend

    core, _ = load_backend("hemocytometer")
    n = 12 if quick else 50
    gains = list(np.linspace(1, 2.6, n)) + [2.6] * (3 if quick else 10) + list(np.linspace(2.6, 1, n // 2))
    out.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out / "exposure_movie.mp4", fps=12, codec="libx264", quality=7, macro_block_size=8)
    worst = (0, None)
    for g in gains:
        core.setProperty("Camera", "Gain", float(g))
        im = core.snap()
        sat = (im >= 254).mean() * 100
        hot = "#C0392B" if sat > 1 else "#222222"
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})
        ax[0].imshow(im, cmap="gray", vmin=0, vmax=255); ax[0].axis("off"); ax[0].set_title(f"virtual camera, gain {g:.2f}")
        ax[1].hist(im.ravel(), bins=128, range=(0, 255), color=hot); ax[1].set_yscale("log")
        ax[1].set_xlim(0, 260); ax[1].set_ylim(1, 3e5); ax[1].axvline(255, color="#C0392B", ls="--", lw=1)
        ax[1].set_xlabel("pixel value"); ax[1].set_ylabel("pixels (log)"); ax[1].set_title(f"histogram: {sat:.0f}% saturated", color=hot)
        fig.tight_layout(); fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
        writer.append_data(frame)
        if sat >= worst[0]: worst = (sat, frame)
    writer.close()
    imageio.imwrite(out / "exposure_poster.jpg", worst[1])
    print(f"saved {out/'exposure_movie.mp4'}  (max saturation {worst[0]:.0f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("results")); p.add_argument("--quick", action="store_true")
    a = p.parse_args(); main(a.out, a.quick)
