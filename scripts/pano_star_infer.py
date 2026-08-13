#!/usr/bin/env python3
"""Star-based PanoVGGT inference for the equirect pipeline (phase 1).

GLUEMAP-style star formation over panoramas: every panorama is the
center of exactly one star whose members are its sequential and
GPS-nearest neighbors. PanoVGGT runs once per star, and each star's
relative poses (center = identity) and radial depth maps are saved for
the global solve (pano_star_solve.py, phase 2).

This is the initializer that uses the feedforward network the way
GLUEMAP intends: as a source of many overlapping local solves fused by
global averaging, instead of a chain of rigid batch Sim(3)s. Every
panorama appears in ~star_size stars, so a single tilted or flipped
prediction loses the vote downstream.

Runs in the PanoVGGT environment (needs ddpy_panovggt + ml_utils).

Outputs (under --out_dir):
    stars/star_%04d.npz   names (S,), extr (S,3,4) center-relative w2c,
                          depth (S,518,1036) float16 radial, 0 = invalid
    stars.json            star membership, ENU per frame, image size
    manifest.json         minimal batching-compatible manifest
                          (frames[].enu_xyz + gps.enu_anchor)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def build_stars(
    num_frames: int,
    enu: np.ndarray,
    star_size: int,
    seq_window: int,
    min_baseline: float = 0.3,
) -> list[list[int]]:
    """One star per frame: sequential neighbors first, GPS-nearest fill.

    Mirrors GLUEMAP's sequential-aware star construction: the capture
    order provides guaranteed-overlapping members, GPS proximity adds
    revisit edges (loop closures) beyond the window.

    Stationary clusters (the operator standing still) are the degenerate
    case: a star whose members all share the center's position has no
    parallax, so the network's relative translations for it are
    unconstrained hallucinations — observed parking two capture-tail
    frames 36 m out. When the capture offers them, the two nearest frames
    at least ``min_baseline`` metres away (by prior) always claim the
    star's last slots.
    """
    from scipy.spatial import cKDTree

    have_gps = np.all(np.isfinite(enu), axis=1)
    tree = cKDTree(enu[have_gps]) if have_gps.sum() >= 2 else None
    gps_rows = np.nonzero(have_gps)[0]

    stars = []
    for i in range(num_frames):
        members = [i]
        for off in range(1, seq_window + 1):
            for j in (i - off, i + off):
                if 0 <= j < num_frames and j not in members:
                    members.append(j)
        if tree is not None and have_gps[i]:
            k = min(len(gps_rows), star_size + 2 * seq_window)
            _, nn = tree.query(enu[i], k=k)
            for row in np.atleast_1d(nn):
                j = int(gps_rows[row])
                if j not in members:
                    members.append(j)
        star = members[:star_size]
        if tree is not None and have_gps[i]:
            _ensure_baseline_members(
                star, i, enu, have_gps, gps_rows, star_size, min_baseline
            )
        stars.append(star)
    return stars


def _ensure_baseline_members(
    star: list[int],
    center: int,
    enu: np.ndarray,
    have_gps: np.ndarray,
    gps_rows: np.ndarray,
    star_size: int,
    min_baseline: float,
    want: int = 2,
) -> None:
    """Give ``star`` at least ``want`` members ≥ ``min_baseline`` from its center.

    No-op for normal walking stars (their sequential neighbors already carry
    baseline); a stationary-cluster star gets its lowest-priority slots
    replaced by the nearest genuinely displaced frames.
    """
    num_baseline = sum(
        1
        for j in star[1:]
        if have_gps[j] and np.linalg.norm(enu[j] - enu[center]) >= min_baseline
    )
    if num_baseline >= want:
        return
    dists = np.linalg.norm(enu[gps_rows] - enu[center], axis=1)
    candidates = [
        int(gps_rows[r])
        for r in np.argsort(dists)
        if dists[r] >= min_baseline and int(gps_rows[r]) not in star
    ]
    slot = len(star) - 1
    for j in candidates:
        if num_baseline >= want:
            break
        if len(star) < star_size:
            star.append(j)
        elif slot > 0:
            star[slot] = j
            slot -= 1
        else:
            break
        num_baseline += 1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, default=None)
    parser.add_argument("--star_size", type=int, default=8)
    parser.add_argument(
        "--seq_window",
        type=int,
        default=3,
        help="sequential neighbors per side before GPS-nearest fill",
    )
    parser.add_argument(
        "--extra_pairs",
        type=Path,
        default=None,
        help="loop_pairs.json from pano_loop_pairs.py; each pair (i, j) "
        "becomes a 4-frame mini-star [i, i+1, j, j+1] whose sequential "
        "sub-edges pin its scale, so the loop edge transports metric "
        "scale across the revisit",
    )
    parser.add_argument(
        "--loop_max_prior_dist",
        type=float,
        default=4.0,
        help="reject loop pairs whose position priors are farther apart "
        "than this (m) — doppelganger guard; only applies when both "
        "ends have priors",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import cv2
    import torch
    from ddpy_panovggt.infer import _load_pano_tensor, forward_pass, load_model
    from ml_utils.geo import gps_fixes_to_enu, load_gps_fixes

    image_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images in {args.images}")
    names = [p.stem for p in image_paths]
    n = len(names)

    # Sidecar-first (exif_overrides.json next to the dataset's images);
    # per-image EXIF is the fallback for frames without an entry.
    fixes = load_gps_fixes(image_paths)
    if any(fix is not None for fix in fixes):
        enu, anchor = gps_fixes_to_enu(fixes)
    else:
        # GPS-less capture (e.g. indoor video walk): stars fall back to
        # sequential neighbors and the solve stays gauge-free until the
        # start/end (or other) alignment.
        enu, anchor = np.full((n, 3), np.nan), None
    logger.info("%d frames, %d with GPS", n, int(np.isfinite(enu[:, 0]).sum()))

    stars = build_stars(n, enu, args.star_size, args.seq_window)

    if args.extra_pairs is not None and args.extra_pairs.exists():
        idx_of = {name: i for i, name in enumerate(names)}
        loop_pairs = json.loads(args.extra_pairs.read_text())["pairs"]
        num_added = num_aliased = 0
        for name_i, name_j, _score in loop_pairs:
            i, j = idx_of.get(name_i), idx_of.get(name_j)
            if i is None or j is None:
                continue
            # Prior gate against doppelganger revisits: self-similar
            # interiors (identical aisles) pass appearance retrieval AND
            # geometric covis scoring, and the resulting false edges drag
            # whole segments onto the wrong pass (560d: 86 aliased pairs
            # around one segment, up to 26 m apart). When both ends carry
            # a position prior, a genuine revisit must be nearby.
            if (
                np.all(np.isfinite(enu[i]))
                and np.all(np.isfinite(enu[j]))
                and float(np.linalg.norm(enu[i] - enu[j]))
                > args.loop_max_prior_dist
            ):
                num_aliased += 1
                continue
            members = [i, min(i + 1, n - 1), j, min(j + 1, n - 1)]
            members = list(dict.fromkeys(members))  # dedupe, keep order
            if len(members) >= 3:
                stars.append(members)
                num_added += 1
        logger.info(
            "Added %d loop-closure mini-stars from %s "
            "(%d rejected by the %.1f m prior gate)",
            num_added,
            args.extra_pairs,
            num_aliased,
            args.loop_max_prior_dist,
        )

    # Preload every pano once (resized to the model resolution).
    logger.info("Preloading %d panoramas ...", n)
    stacked = _load_pano_tensor(str(args.images), [p.name for p in image_paths])
    if stacked.shape[0] != n:
        raise RuntimeError(f"Loaded {stacked.shape[0]}/{n} panoramas; aborting")
    tensors = list(stacked)
    target_h, target_w = tensors[0].shape[-2:]

    masks = None
    if args.mask_dir is not None and args.mask_dir.is_dir():
        masks = []
        for name in names:
            mask_path = args.mask_dir / f"{name}.png"
            mask = (
                cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_path.exists()
                else None
            )
            if mask is not None:
                mask = cv2.resize(mask, (target_w, target_h), cv2.INTER_NEAREST)
            masks.append(mask)

    model = load_model(str(args.checkpoint), device=args.device)
    # load_model only map_locations the checkpoint; the module itself
    # stays on CPU (forward_pass silently follows the model's device).
    model.eval().to(args.device)
    stars_dir = args.out_dir / "stars"
    stars_dir.mkdir(parents=True, exist_ok=True)

    for star_idx, members in enumerate(stars):
        out_path = stars_dir / f"star_{star_idx:04d}.npz"
        if out_path.exists():
            continue
        imgs = torch.stack([tensors[j] for j in members]).to(args.device)
        out = forward_pass(imgs, model)
        c2w = np.asarray(out["camera_poses"], dtype=np.float64)  # (S,4,4)
        depth = np.asarray(out["depth"], dtype=np.float32)  # (S,H,W)
        depth[~np.isfinite(depth)] = 0.0
        if masks is not None:
            for slot, j in enumerate(members):
                if masks[j] is not None:
                    depth[slot][masks[j] == 0] = 0.0

        # Center-relative w2c: view 0 becomes the identity.
        w2c = np.linalg.inv(c2w)
        extr_rel = np.einsum("sij,jk->sik", w2c, c2w[0])[:, :3, :]
        np.savez(
            out_path,
            names=np.array([names[j] for j in members]),
            extr=extr_rel.astype(np.float32),
            depth=depth.astype(np.float16),
        )
        if (star_idx + 1) % 20 == 0 or star_idx + 1 == len(stars):
            logger.info("star %d/%d done", star_idx + 1, len(stars))

    stars_payload = {
        "schema": "pano-stars-v1",
        "image_size": [target_w, target_h],
        "frames": names,
        "enu": {
            name: (enu[i].tolist() if np.all(np.isfinite(enu[i])) else None)
            for i, name in enumerate(names)
        },
        "enu_anchor": (
            {
                "lat_deg": anchor.lat_deg,
                "lon_deg": anchor.lon_deg,
                "alt_m": anchor.alt_m,
            }
            if anchor is not None
            else None
        ),
        "stars": [
            {"center": names[m[0]], "members": [names[j] for j in m]}
            for m in stars
        ],
    }
    (args.out_dir / "stars.json").write_text(json.dumps(stars_payload))

    # Minimal manifest so the downstream stages (re-anchor, GPS targets)
    # work without a batching run.
    manifest = {
        "schema_version": 1,
        "frames": [
            {
                "global_index": i,
                "filename": image_paths[i].name,
                "enu_xyz": (
                    enu[i].tolist() if np.all(np.isfinite(enu[i])) else None
                ),
            }
            for i in range(n)
        ],
        "gps": {
            "enu_anchor": (
                {
                    "lat_deg": anchor.lat_deg,
                    "lon_deg": anchor.lon_deg,
                    "alt_m": anchor.alt_m,
                    "has_altitude": anchor.has_altitude,
                }
                if anchor is not None
                else None
            )
        },
        "batches": [],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest))
    logger.info("Wrote %d stars to %s", len(stars), args.out_dir)


if __name__ == "__main__":
    main()
