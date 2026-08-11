#!/usr/bin/env python3
"""Appearance-based loop-closure candidates for the equirect star pipeline.

Computes MegaLoc retrieval descriptors (Berton & Masone, MIT license,
loaded via torch.hub from gmberton/MegaLoc) for every panorama and emits
long-range revisit candidates: pairs far apart in capture order but
similar in appearance. pano_star_infer consumes them (--extra_pairs) as
4-frame mini-stars [i, i+1, j, j+1] whose sequential sub-edges pin the
mini-star scale, so the loop edge transports metric scale across the
revisit — closing the multiplicative scale drift a sequential-only
chain accumulates.

Each equirect contributes FOUR heading descriptors (square crops at
yaw 0/90/180/270, each spanning 180 deg horizontally): revisits happen
at arbitrary headings, and a perspective-trained VPR model reads a
heading-local crop far better than a squashed full equirect. Pair
similarity is the max over the 4x4 crop combinations.

No gluemap-repo dependency: torch.hub downloads the model (cached under
~/.cache/torch/hub; weights via huggingface_hub) on first use.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
INPUT_SIZE = 322  # DINOv2 patch 14: MegaLoc's recommended eval resolution
NUM_HEADINGS = 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def heading_crops(equirect: np.ndarray) -> list[np.ndarray]:
    """Four square heading crops (yaw 0/90/180/270) of an equirect image.

    Each crop is the HxH window centred on the heading — 180 deg of
    horizontal field of view on a W = 2H equirect — resized to the
    model input. Wrap-around is handled by rolling.
    """
    import cv2

    h, w = equirect.shape[:2]
    crops = []
    for k in range(NUM_HEADINGS):
        center = int(round(k * w / NUM_HEADINGS))
        rolled = np.roll(equirect, w // 2 - center, axis=1)
        window = rolled[:, (w - h) // 2 : (w + h) // 2]
        crops.append(
            cv2.resize(
                window, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA
            )
        )
    return crops


def compute_descriptors(image_paths: list[Path], device: str, batch_size: int):
    """(N, NUM_HEADINGS, D) L2-normalized MegaLoc descriptors."""
    import cv2
    import torch

    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    model.eval().to(device)

    batch: list[np.ndarray] = []
    chunks = []

    def flush() -> None:
        if not batch:
            return
        arr = np.stack(batch).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            chunks.append(model(tensor).cpu())
        batch.clear()

    for i, path in enumerate(image_paths):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        batch.extend(heading_crops(rgb))
        if len(batch) >= batch_size:
            flush()
        if i % 50 == 0:
            logger.info("descriptors %d/%d", i, len(image_paths))
    flush()
    desc = torch.cat(chunks)
    desc = torch.nn.functional.normalize(desc, dim=1)
    return desc.reshape(len(image_paths), NUM_HEADINGS, -1)


def pooled_similarity(desc):
    """(N, N) similarity: max over the heading-crop combinations."""
    import torch

    n, k, dim = desc.shape
    flat = desc.reshape(n * k, dim)
    sim = flat @ flat.T  # (n*k, n*k)
    sim = sim.reshape(n, k, n, k)
    return torch.amax(sim, dim=(1, 3))


def select_loop_pairs(
    similarity,
    names: list[str],
    top_k: int,
    min_gap: int,
    min_similarity: float,
    max_pairs: int,
) -> list[tuple[str, str, float]]:
    import torch

    n = len(names)
    candidates: dict[tuple[int, int], float] = {}
    for i in range(n):
        sims = similarity[i].clone()
        lo, hi = max(0, i - min_gap), min(n, i + min_gap + 1)
        sims[lo:hi] = -1.0  # mask the sequential window (already connected)
        vals, idxs = torch.topk(sims, min(top_k, n))
        for val, j in zip(vals.tolist(), idxs.tolist(), strict=True):
            if val < min_similarity:
                continue
            key = (min(i, j), max(i, j))
            candidates[key] = max(candidates.get(key, 0.0), val)
    ranked = sorted(candidates.items(), key=lambda kv: -kv[1])[:max_pairs]
    return [(names[i], names[j], round(s, 4)) for (i, j), s in ranked]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--min_gap",
        type=int,
        default=30,
        help="minimum capture-order distance for a pair to count as a revisit",
    )
    parser.add_argument("--min_similarity", type=float, default=0.3)
    parser.add_argument("--max_pairs", type=int, default=1000)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="crops per forward pass (4 crops per pano)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    names = [p.stem for p in image_paths]
    desc = compute_descriptors(image_paths, args.device, args.batch_size)
    similarity = pooled_similarity(desc)
    pairs = select_loop_pairs(
        similarity,
        names,
        args.top_k,
        args.min_gap,
        args.min_similarity,
        args.max_pairs,
    )
    args.out.write_text(
        json.dumps({"schema": "pano-loop-pairs-v1", "pairs": pairs})
    )
    logger.info("Wrote %d loop-closure pairs to %s", len(pairs), args.out)


if __name__ == "__main__":
    main()
