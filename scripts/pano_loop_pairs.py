#!/usr/bin/env python3
"""Appearance-based loop-closure candidates for the equirect star pipeline.

Computes GLUEMAP's SALAD retrieval descriptors for every panorama
(equirects resized to the 322px square input, exactly as
gluemap.controllers.image_retrieval does) and emits long-range revisit
candidates: pairs far apart in capture order but similar in appearance.
pano_star_infer consumes them (--extra_pairs) as 4-frame mini-stars
[i, i+1, j, j+1] whose sequential sub-edges pin the mini-star scale, so
the loop edge transports metric scale across the revisit — closing the
multiplicative scale drift a sequential-only chain accumulates.

Runs in the GLUEMAP environment (needs the repo + dino_salad.ckpt).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPTS_DIR.parent
for path in (str(REPO_DIR), str(REPO_DIR / "thirdparty")):
    if path not in sys.path:
        sys.path.insert(0, path)

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def compute_descriptors(
    image_paths: list[Path], checkpoint: Path, device: str, batch_size: int
):
    import torch

    import thirdparty.path_to_thirdparty  # noqa: F401  (sys.path setup)
    from gluemap.utils.load_fn import load_and_preprocess_images
    from gluemap.utils.model_loader import _import_vpr_model

    vpr_model = _import_vpr_model()
    model = vpr_model(
        backbone_arch="dinov2_vitb14",
        backbone_config={
            "num_trainable_blocks": 4,
            "return_token": True,
            "norm_layer": True,
        },
        agg_arch="SALAD",
        agg_config={
            "num_channels": 768,
            "num_clusters": 64,
            "cluster_dim": 128,
            "token_dim": 256,
        },
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False),
        strict=False,
    )
    model.eval().to(device)

    descriptors = []
    paths = [str(p) for p in image_paths]
    for i in range(0, len(paths), batch_size):
        images, _, _ = load_and_preprocess_images(
            paths[i : i + batch_size],
            image_size=322,
            patch_size=14,
            force_square=True,
        )
        with torch.no_grad():
            descriptors.append(model(images.to(device)).cpu())
        if (i // batch_size) % 10 == 0:
            logger.info(
                "descriptors %d/%d", min(i + batch_size, len(paths)), len(paths)
            )
    desc = torch.cat(descriptors)
    return torch.nn.functional.normalize(desc, dim=1)


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
        "--checkpoint",
        type=Path,
        default=REPO_DIR / "checkpoints" / "dino_salad.ckpt",
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
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    names = [p.stem for p in image_paths]
    desc = compute_descriptors(
        image_paths, args.checkpoint, args.device, args.batch_size
    )
    similarity = desc @ desc.T
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
