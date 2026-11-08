"""Shared helpers: open Lucien Hinderling's virtual microscope and measure tiles.

Everything the experiments do to the "microscope" goes through the same
pymmcore-plus calls a real Micro-Manager scope accepts:
    core.setXYPosition(x, y), core.setPosition(z), core.snap()
The only thing that touches the simulator directly is the *ground truth*
(where the tissue really is, ``sim.tissue_z``), which strategies never read.
"""
import logging

import cv2
import numpy as np

logging.disable(logging.CRITICAL)  # the backend is chatty about Micro-Manager adapters


def open_scope(tissue="mixed", seed=1, world_size=2048, n_nuclei=3000):
    """Histology backend: simulated H&E, 10x, depth of field 6 µm."""
    from virtual_microscope.backends import load_backend

    core, sim = load_backend("histology", world_size=world_size, n_nuclei=n_nuclei, seed=seed, tissue_type=tissue)
    core.setConfig("Channel", "H-and-E")
    return core, sim


def snake_grid(step=200, lim=860):
    """XY positions (µm) in snake order, like a real tile scan."""
    xs = np.arange(-lim, lim + 1, step)
    out = []
    for r, y in enumerate(xs):
        row = [(float(x), float(y)) for x in xs]
        out += row if r % 2 == 0 else row[::-1]
    return out


def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(float)


def tissue_fraction(im, thresh=215):
    """Fraction of pixels darker than background glass."""
    return float((gray(im) < thresh).mean())


def sharpness(im):
    """Normalised focus measure: variance of the Laplacian / image variance."""
    g = gray(cv2.resize(im, (256, 256)))
    return float(cv2.Laplacian(g, cv2.CV_64F).var() / (g.var() + 1e-6))


def qc_features(im):
    """11 numbers per tile: 4 sharpness measures, RGB mean, RGB spread, tissue fraction."""
    g = gray(cv2.resize(im, (256, 256)))
    lap = cv2.Laplacian(g, cv2.CV_64F)
    gx, gy = cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    yy, xx = np.mgrid[:256, :256]
    hf = spec[np.hypot(yy - 128, xx - 128) > 256 / 6].sum() / (spec.sum() + 1e-9)
    rgb = im.reshape(-1, 3).astype(float) / 255
    return [np.log(lap.var() / (g.var() + 1e-6) + 1e-9), np.log(((g[2:] - g[:-2]) ** 2).mean() + 1e-9),
            np.log((gx ** 2 + gy ** 2).mean() + 1e-9), hf, *rgb.mean(0), *rgb.std(0), tissue_fraction(im)]
