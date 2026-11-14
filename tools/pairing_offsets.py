"""Measure the time offset between the VIS and IR frames of a directory pair.

A camera pan moves the whole image, in both modalities at once. So the global
shift between consecutive frames (found by phase correlation) is a signal that
the two directories share. If VIS frame i and IR frame i are the same moment,
the two shift signals correlate best at lag 0. A lag of +1 means VIS frame i
matches IR frame i+1.

    # lag table for every collection in the pairing list, by thirds of each
    uv run python tools/pairing_offsets.py data/raw/wisard-full lags

    # search a time scale as well: VIS frame i matches IR frame s*i + L
    uv run python tools/pairing_offsets.py data/raw/wisard-full scale \
        210327_Airfield_FLIR_VIS_4 210327_Airfield_FLIR_IR_4

Motion signals are cached in outputs/pairing-offsets/. Computing them for all
dual-camera directories takes about a minute on an M4 laptop.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from aerial_search.data.wisard import WISARD_COLLECTIONS, _frames_by_index
from PIL import Image

CACHE = Path("outputs/pairing-offsets")
WIDTH = 160  # pixels; frames are shrunk to this width before phase correlation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lags")
    scale = sub.add_parser("scale")
    scale.add_argument("rgb_dir")
    scale.add_argument("thermal_dir")
    args = parser.parse_args()

    if args.command == "lags":
        dirs = [
            d for c in WISARD_COLLECTIONS.values() for d in (c.rgb_dir, c.thermal_dir)
        ]
        _cache_all(args.source, dirs)
        print("| collection | whole | first third | middle third | last third |")
        print("|---|---|---|---|---|")
        # Lags are measured from equal frame numbers, before any thermal_offset.
        for cid, c in WISARD_COLLECTIONS.items():
            cells = _lags_by_third(_shifts(c.rgb_dir), _shifts(c.thermal_dir))
            print(f"| {cid} | " + " | ".join(cells) + " |")
    else:
        _cache_all(args.source, [args.rgb_dir, args.thermal_dir])
        r, s, lag = _best_scale(args.rgb_dir, args.thermal_dir)
        print(f"best: IR frame = {s:.3f} x VIS frame {lag:+.0f}  (r={r:.2f})")


def _cache_all(source: Path, dirs: list[str]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [source / d for d in dirs if not (CACHE / f"{d}.npz").exists()]
    with ProcessPoolExecutor() as pool:
        list(pool.map(_cache_motion, todo))


def _cache_motion(directory: Path) -> None:
    frames = _frames_by_index(directory)
    index = np.array(sorted(frames))
    dx = np.full(len(index), np.nan)
    dy = np.full(len(index), np.nan)
    previous = None
    for k, i in enumerate(index):
        current = _edges(frames[int(i)])
        if previous is not None and previous.shape == current.shape:
            dx[k], dy[k] = _phase_shift(current, previous)
        previous = current
    np.savez(CACHE / f"{directory.name}.npz", index=index, dx=dx, dy=dy)


def _edges(path: Path) -> np.ndarray:
    image = Image.open(path)
    image.draft("L", (image.width // 8, image.height // 8))
    image = image.convert("L")
    height = round(image.height * WIDTH / image.width)
    pixels = np.asarray(image.resize((WIDTH, height)), dtype=np.float32)
    gy, gx = np.gradient(pixels - pixels.mean())
    return np.hypot(gx, gy)


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Shift of a relative to b, as a fraction of the image size."""
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    cross = np.fft.fft2(a * window) * np.conj(np.fft.fft2(b * window))
    cross /= np.abs(cross) + 1e-9
    y, x = np.unravel_index(np.argmax(np.fft.ifft2(cross).real), a.shape)
    y = y - a.shape[0] if y > a.shape[0] // 2 else y
    x = x - a.shape[1] if x > a.shape[1] // 2 else x
    return x / a.shape[1], y / a.shape[0]


def _shifts(name: str) -> dict[int, tuple[float, float]]:
    """Shift into each frame from the frame before it, where the two are adjacent."""
    z = np.load(CACHE / f"{name}.npz")
    gap = np.r_[0, np.diff(z["index"])]
    return {
        int(i): (float(x), float(y))
        for i, x, y, g in zip(z["index"], z["dx"], z["dy"], gap, strict=True)
        if g == 1 and np.isfinite(x)
    }


def _correlation(rgb, thermal, lag, lo=None, hi=None) -> float:
    matched = [
        (rgb[i], thermal[i + lag])
        for i in rgb
        if i + lag in thermal and (lo is None or lo <= i < hi)
    ]
    if len(matched) < 20:
        return np.nan
    a = np.array([m[0] for m in matched])
    b = np.array([m[1] for m in matched])
    with np.errstate(invalid="ignore", divide="ignore"):
        r = [np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(2)]
    return float(np.nanmean(r))


def _lags_by_third(rgb, thermal, max_lag: int = 10) -> list[str]:
    first, last = min(rgb), max(rgb) + 1
    step = (last - first) / 3
    spans = [(None, None)] + [
        (first + t * step, first + (t + 1) * step) for t in range(3)
    ]
    cells = []
    for lo, hi in spans:
        scores = {
            lag: _correlation(rgb, thermal, lag, lo, hi)
            for lag in range(-max_lag, max_lag + 1)
        }
        lag = max(scores, key=lambda k: np.nan_to_num(scores[k], nan=-1))
        cells.append(f"{lag:+d} (r={scores[lag]:.2f})")
    return cells


def _trajectory(name: str):
    z = np.load(CACHE / f"{name}.npz")
    index = z["index"]
    gap = np.maximum(np.r_[1, np.diff(index)], 1)
    full = np.arange(index[0], index[-1] + 1)
    x = np.cumsum(np.interp(full, index, np.nan_to_num(z["dx"]) / gap))
    y = np.cumsum(np.interp(full, index, np.nan_to_num(z["dy"]) / gap))
    return full, x, y


def _best_scale(rgb_name: str, thermal_name: str) -> tuple[float, float, float]:
    """Search IR frame = s * VIS frame + L over s in [0.25, 4] and |L| < 1200."""
    vi, vx, vy = _trajectory(rgb_name)
    ti, tx, ty = _trajectory(thermal_name)
    i = vi[1:]
    vdx, vdy = np.diff(vx), np.diff(vy)
    best = (-1.0, 1.0, 0.0)
    for s in np.exp(np.linspace(np.log(0.25), np.log(4), 97)):
        for lag in np.arange(-1200, 1200, 1.0):
            t1, t0 = s * i + lag, s * (i - 1) + lag
            ok = (t0 >= ti[0]) & (t1 <= ti[-1])
            if ok.sum() < max(50, 0.5 * len(i)):
                continue
            tdx = np.interp(t1[ok], ti, tx) - np.interp(t0[ok], ti, tx)
            tdy = np.interp(t1[ok], ti, ty) - np.interp(t0[ok], ti, ty)
            with np.errstate(invalid="ignore", divide="ignore"):
                rs = [np.corrcoef(vdx[ok], tdx)[0, 1], np.corrcoef(vdy[ok], tdy)[0, 1]]
            rs = [v for v in rs if np.isfinite(v)]
            r = float(np.mean(rs)) if rs else np.nan
            if np.isfinite(r) and r > best[0]:
                best = (float(r), float(s), float(lag))
    return best


if __name__ == "__main__":
    main()
