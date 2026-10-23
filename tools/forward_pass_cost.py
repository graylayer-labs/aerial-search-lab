"""Measure the cost of one frozen-backbone forward pass on the laptop (#64).

For one model, take N frames sampled with a seed from the pair manifest, run
the vision tower on MPS in half precision, and report per image: milliseconds
for the forward pass (host-to-device copy of the input and copy of the patch
features back to the host included; JPEG decoding and resizing excluded and
reported on their own), the patch grid, the size of the patch features in
16-bit floats, and the size after 2x2 average pooling.

    uv run python tools/forward_pass_cost.py siglip2-base-512 --camera rgb
    uv run python tools/forward_pass_cost.py siglip2-base-naflex --camera thermal

Models: siglip2-base-512, siglip2-large-512, siglip2-base-naflex, dinov2-base.
Weights come from the Hugging Face hub through transformers and are not
modified. Peak memory is the highest MPS driver allocation seen after any
forward pass (weights included); MPS has no true peak counter. No
PYTORCH_ENABLE_MPS_FALLBACK is set, so an op without an MPS kernel raises
instead of falling back to the CPU silently.

Frames are resized to a square for the fixed-resolution models, so a 16:9 frame
is squashed. That changes the picture, not the cost. The NaFlex model keeps the
aspect ratio and the grid is whatever fits the patch budget.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from frozen_head.backbone import SPECS, Spec, load, pool2, weight_checksums

TILE = 2  # a 1024 px frame is a 2x2 grid of 512 px tiles


def sample_frames(data: Path, manifest: Path, camera: str, n: int, seed: int):
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    key = "rgb_image" if camera == "rgb" else "thermal_image"
    chosen = random.Random(seed).sample(rows, n)
    return [data / row[key] for row in chosen]


def to_square_tensor(image: Image.Image, side: int, mean, std) -> torch.Tensor:
    resized = image.convert("RGB").resize((side, side), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    return torch.from_numpy(array).permute(2, 0, 1)  # 3 x side x side


def tiles(frame: torch.Tensor) -> torch.Tensor:
    """3 x (TILE*s) x (TILE*s) to (TILE*TILE) x 3 x s x s, row-major."""
    side = frame.shape[-1] // TILE
    parts = frame.unfold(1, side, side).unfold(2, side, side)  # 3 x T x T x s x s
    return parts.permute(1, 2, 0, 3, 4).reshape(TILE * TILE, 3, side, side)


def stitch(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    """(TILE*TILE) x (grid*grid) x D to (TILE*grid) x (TILE*grid) x D."""
    d = tokens.shape[-1]
    x = tokens.reshape(TILE, TILE, grid, grid, d).permute(0, 2, 1, 3, 4)
    return x.reshape(TILE * grid, TILE * grid, d)


def patch_tokens(spec: Spec, model, pixels: torch.Tensor) -> torch.Tensor:
    out = model(pixel_values=pixels).last_hidden_state
    return out[:, 1:] if spec.kind == "dinov2" else out  # drop the CLS token


def run_fixed(spec, model, device, tensors, tiled: bool):
    """Returns per-image ms, grid shape of the cached features, bytes."""
    grid = spec.tile_px // spec.patch
    times, mem = [], []
    shape = None
    nbytes = pooled_bytes = 0
    for frame in tensors:
        t0 = time.perf_counter()
        batch = (tiles(frame) if tiled else frame[None]).to(device, torch.float16)
        with torch.no_grad():
            tokens = patch_tokens(spec, model, batch)
            fmap = stitch(tokens, grid) if tiled else tokens[0].reshape(grid, grid, -1)
            pooled = pool2(fmap)
        host, host_pooled = fmap.cpu(), pooled.cpu()
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        mem.append(torch.mps.driver_allocated_memory() if device == "mps" else 0)
        shape = tuple(host.shape)
        nbytes, pooled_bytes = host.numel() * 2, host_pooled.numel() * 2
        del batch, tokens, fmap, pooled, host, host_pooled
    return times, mem, shape, nbytes, pooled_bytes


def run_naflex(model, device, inputs):
    times, mem, shapes = [], [], []
    nbytes = pooled_bytes = 0
    for item in inputs:
        t0 = time.perf_counter()
        feed = {
            k: v.to(device, torch.float16 if v.is_floating_point() else v.dtype)
            for k, v in item.items()
        }
        with torch.no_grad():
            tokens = model(**feed).last_hidden_state[0]
            h, w = (int(v) for v in item["spatial_shapes"][0])
            fmap = tokens[: h * w].reshape(h, w, -1)
            # an odd side loses its last row or column to the 2x2 pooling
            pooled = pool2(fmap)
        host, host_pooled = fmap.cpu(), pooled.cpu()
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        mem.append(torch.mps.driver_allocated_memory() if device == "mps" else 0)
        shapes.append((h, w))
        nbytes += host.numel() * 2
        pooled_bytes += host_pooled.numel() * 2
    n = len(inputs)
    return times, mem, shapes, nbytes // n, pooled_bytes // n


def report(label, times, mem, shape, nbytes, pooled_bytes, decode_ms, total):
    ms = 1000 * float(np.mean(times))
    row = {
        "setting": label,
        "ms_per_image": round(ms, 1),
        "ms_median": round(1000 * float(np.median(times)), 1),
        "ms_std": round(1000 * float(np.std(times)), 1),
        "peak_mps_gb": round(max(mem) / 2**30, 2),
        "grid": shape,
        "feature_mb_fp16": round(nbytes / 2**20, 2),
        "pooled_mb_fp16": round(pooled_bytes / 2**20, 2),
        "decode_resize_ms": round(decode_ms, 1),
        "projected_cache_gb": round(nbytes * total / 1e9, 1),
        "projected_pooled_cache_gb": round(pooled_bytes * total / 1e9, 1),
        "projected_minutes": round(ms * total / 60000, 1),
    }
    print(json.dumps(row), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(SPECS))
    parser.add_argument("--camera", choices=["rgb", "thermal"], default="rgb")
    parser.add_argument("--data", type=Path, default=Path("data/raw/wisard-full"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/wisard-full/all_pairs.jsonl"),
    )
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total", type=int, default=29668, help="images to project")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    spec = SPECS[args.model]
    paths = sample_frames(args.data, args.manifest, args.camera, args.n, args.seed)
    print(json.dumps({"model": args.model, "repo": spec.repo, "args": str(vars(args))}))
    print(
        json.dumps({"sha256": weight_checksums(spec.repo, spec.revision)}), flush=True
    )
    model = load(spec, args.device)
    print(json.dumps({"torch": torch.__version__, "device": args.device}))

    if spec.kind == "naflex":
        processor = AutoImageProcessor.from_pretrained(
            spec.repo, revision=spec.revision
        )
        for label, budget in spec.budgets:
            t0 = time.perf_counter()
            inputs = []
            for path in paths:
                with Image.open(path) as image:
                    inputs.append(
                        processor(
                            images=[image.convert("RGB")],
                            return_tensors="pt",
                            max_num_patches=budget,
                        )
                    )
            decode_ms = 1000 * (time.perf_counter() - t0) / len(paths)
            run_naflex(model, args.device, inputs[: args.warmup])
            times, mem, shapes, nb, pb = run_naflex(model, args.device, inputs)
            grid = max(set(shapes), key=shapes.count)
            report(
                f"{args.model} {label}px budget={budget} {args.camera}",
                times, mem, grid, nb, pb, decode_ms, args.total,
            )  # fmt: skip
        return

    if spec.kind == "siglip":
        mean = std = (0.5, 0.5, 0.5)
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    for tiled in (False, True):
        side = spec.tile_px * (TILE if tiled else 1)
        t0 = time.perf_counter()
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(to_square_tensor(image, side, mean, std))
        decode_ms = 1000 * (time.perf_counter() - t0) / len(paths)
        warm = tensors[: args.warmup]
        run_fixed(spec, model, args.device, warm, tiled)
        times, mem, shape, nb, pb = run_fixed(spec, model, args.device, tensors, tiled)
        label = (
            f"{args.model} {side}px {'4 tiles' if tiled else 'single'} {args.camera}"
        )
        report(label, times, mem, shape, nb, pb, decode_ms, args.total)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    print(json.dumps({"process_max_rss_gb_macos": round(rss, 2)}))


if __name__ == "__main__":
    main()
