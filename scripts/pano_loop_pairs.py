#!/usr/bin/env python3
"""Appearance-based loop-closure candidates for the equirect star pipeline.

Computes MegaLoc retrieval descriptors (via ddpy_panovggt.vpr — vendored
MIT model, local safetensors weights, no network at runtime) for every
panorama and emits long-range revisit candidates: pairs far apart in
capture order but similar in appearance. pano_star_infer consumes them
(--extra_pairs) as 4-frame mini-stars [i, i+1, j, j+1] whose sequential
sub-edges pin the mini-star scale, so the loop edge transports metric
scale across the revisit — closing the multiplicative scale drift a
sequential-only chain accumulates. Aliased candidates (identical-looking
but distinct places) are additionally gated there by prior distance.

Each equirect contributes four heading descriptors (square crops at yaw
0/90/180/270); pair similarity is the max over the heading combinations.
Runs in the PanoVGGT environment (needs ddpy_panovggt).
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def default_weights() -> Path | None:
    """checkpoints/megaloc.safetensors of the (editable) ddpy-panovggt repo."""
    import ddpy_panovggt

    repo = Path(ddpy_panovggt.__file__).resolve().parents[2]
    candidate = repo / "checkpoints" / "megaloc.safetensors"
    return candidate if candidate.exists() else None


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
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="megaloc.safetensors (default: the ddpy-panovggt repo's "
        "checkpoints/ next to model.pt)",
    )
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

    from ddpy_panovggt.vpr import (
        compute_pano_descriptors,
        load_vpr_model,
        pooled_similarity,
    )

    weights = args.weights or default_weights()
    if weights is None or not Path(weights).exists():
        raise SystemExit(
            "megaloc.safetensors not found — pass --weights or place it in "
            "the ddpy-panovggt repo's checkpoints/ (hf.co/gberton/MegaLoc)"
        )

    image_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    names = [p.stem for p in image_paths]
    model = load_vpr_model(weights, device=args.device)
    logger.info(
        "MegaLoc loaded from %s; %d panoramas", weights, len(image_paths)
    )
    desc = compute_pano_descriptors(
        model, image_paths, batch_size=args.batch_size, device=args.device
    )
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
