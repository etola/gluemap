#!/usr/bin/env python3
"""Star-based global solve for the equirect pipeline (phase 2).

Consumes the per-star PanoVGGT predictions written by
pano_star_infer.py and fuses them with ml_utils.star_solve global mapping
machinery — the way feedforward outputs are meant to be used there:

  1. Covisibility scoring per star edge (round-trip spherical depth
     reprojection, the equirect analogue of GLUEMAP's
     covisibility_extraction).
  2. Robust rotation averaging over all star edges
     (gluemap.estimators.rotation_averaging_pycolmap: pycolmap L1+IRLS,
     Geman-McClure) — every pano's rotation is voted by ~star_size
     independent local solves, so a single tilted or flipped star loses.
  3. MST initialization + similarity averaging with one free scale per
     star (gluemap.math.mst_initialization, gluemap.estimators.
     similarity_averaging).
  4. Robust Sim(3) onto the GPS ENU track for the metric frame.
  5. Per-frame metric radial depth = per-pixel median over every star
     observing the frame (scale-corrected per star).

Output is contract-compatible with batch_registration's registered
directory: poses.npz (c2w + frame_names), dmaps/<name>.npy,
neighbors.json — the downstream stages run unchanged.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (SCRIPTS_DIR, SCRIPTS_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

logger = logging.getLogger(__name__)


def _spherical_dirs(
    us: np.ndarray, vs: np.ndarray, w: int, h: int
) -> np.ndarray:
    """Unit ray directions for equirect pixel indices (ml_utils convention)."""
    yaw = ((us + 0.5) / w * 2 - 1) * np.pi
    pitch = (1 - 2 * (vs + 0.5) / h) * np.pi / 2
    return np.stack(
        [
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch),
            np.cos(yaw) * np.cos(pitch),
        ],
        axis=-1,
    )


def _project_spherical(
    pts: np.ndarray, w: int, h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points -> integer equirect pixel indices."""
    yaw = np.arctan2(pts[:, 0], pts[:, 2])
    pitch = -np.arctan2(pts[:, 1], np.hypot(pts[:, 0], pts[:, 2]))
    u = ((1 + yaw / np.pi) / 2 * w).astype(np.int64) % w
    v = np.clip(((1 - pitch * 2 / np.pi) / 2 * h).astype(np.int64), 0, h - 1)
    return u, v


def _unproject_valid(
    depth: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Grid-sample valid pixels of a radial depth map -> camera-frame points.

    Returns (points (K,3), depths (K,)).
    """
    h, w = depth.shape
    vs, us = np.mgrid[stride // 2 : h : stride, stride // 2 : w : stride]
    us, vs = us.ravel(), vs.ravel()
    d = depth[vs, us]
    keep = np.isfinite(d) & (d > 0)
    us, vs, d = us[keep], vs[keep], d[keep]
    return _spherical_dirs(us, vs, w, h) * d[:, None], d


def _directional_agreement(
    pts_src: np.ndarray,
    extr_src_to_dst: np.ndarray | None,
    depth_dst: np.ndarray,
    tol: float,
) -> float:
    """Fraction of source points whose range agrees with the target depth map.

    ``extr_src_to_dst`` maps source-frame points into the target camera
    frame (3x4); None means the frames coincide.
    """
    if pts_src.shape[0] < 30:
        return 0.0
    pts = pts_src
    if extr_src_to_dst is not None:
        pts = pts_src @ extr_src_to_dst[:3, :3].T + extr_src_to_dst[:3, 3]
    h, w = depth_dst.shape
    u, v = _project_spherical(pts, w, h)
    d_pred = np.linalg.norm(pts, axis=1)
    d_map = depth_dst[v, u]
    valid = np.isfinite(d_map) & (d_map > 0) & (d_pred > 1e-6)
    if valid.sum() < 30:
        return 0.0
    rel = np.abs(d_map[valid] - d_pred[valid]) / d_pred[valid]
    return float((rel < tol).mean())


def compute_star_scores(
    extr: np.ndarray,
    depths: np.ndarray,
    stride: int = 14,
    tol: float = 0.05,
) -> np.ndarray:
    """Per-member covisibility scores of one star (center = 1.0).

    Symmetric round-trip: center depth samples projected into the member
    and member samples projected back, score = min of the two agreement
    ratios (conservative, like batch_registration's neighbor gating).
    """
    num = extr.shape[0]
    scores = np.zeros(num)
    scores[0] = 1.0
    pts_center, _ = _unproject_valid(depths[0].astype(np.float32), stride)
    for j in range(1, num):
        depth_j = depths[j].astype(np.float32)
        fwd = _directional_agreement(pts_center, extr[j], depth_j, tol)
        pts_j, _ = _unproject_valid(depth_j, stride)
        # Member -> center: invert the center-relative extrinsic.
        r = extr[j][:3, :3]
        inv = np.hstack([r.T, (-r.T @ extr[j][:3, 3])[:, None]])
        rev = _directional_agreement(
            pts_j, inv, depths[0].astype(np.float32), tol
        )
        scores[j] = min(fwd, rev)
    return scores


def leveling_rotation(rotations: dict[int, np.ndarray]) -> np.ndarray:
    """Rotation aligning the median camera up-vector to +z (gravity leveling).

    The pano frame is y-down, so the world up-vector of frame i is the
    negated y row of its w2c rotation. Same construction as the
    production pipeline's ground-plane alignment.
    """
    ups = np.stack([-rotations[i][1] for i in sorted(rotations)])
    up = np.median(ups, axis=0)
    up /= max(np.linalg.norm(up), 1e-9)
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, z_axis)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        if up[2] > 0:
            return np.eye(3)
        return Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    angle = np.arccos(np.clip(np.dot(up, z_axis), -1.0, 1.0))
    return Rotation.from_rotvec(axis / axis_norm * angle).as_matrix()


def start_end_alignment(
    rotations: dict[int, np.ndarray],
    centers: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
):
    """Sim(3) to the plan frame from start/end anchor points (GPS-less walks).

    The production walkthrough aligns GPS-less captures with the plan's
    start/end points: gravity leveling pins two rotation DOF, the
    start-to-end segment fixes yaw and metric scale, and the start point
    fixes translation. First/last frames are the anchors.
    """
    from ml_utils.sim3 import Sim3

    u_level = leveling_rotation(rotations)
    c_start = u_level @ centers[0]
    c_end = u_level @ centers[-1]
    seg_model = (c_end - c_start)[:2]
    seg_plan = (end_xyz - start_xyz)[:2]
    len_model = float(np.linalg.norm(seg_model))
    len_plan = float(np.linalg.norm(seg_plan))
    if len_model < 1e-6 or len_plan < 1e-6:
        logger.warning(
            "start/end segment degenerate (model %.3f, plan %.3f); "
            "leveling only",
            len_model,
            len_plan,
        )
        return Sim3(scale=1.0, R=u_level, t=np.zeros(3))
    scale = len_plan / len_model
    yaw = np.arctan2(seg_plan[1], seg_plan[0]) - np.arctan2(
        seg_model[1], seg_model[0]
    )
    r_yaw = Rotation.from_euler("z", yaw).as_matrix()
    rot = r_yaw @ u_level
    t = start_xyz - scale * (rot @ centers[0])
    logger.info(
        "start/end alignment: scale %.4f m/unit, yaw %.1f deg, "
        "segment %.2f m (plan %.2f m)",
        scale,
        np.degrees(yaw),
        len_model * scale,
        len_plan,
    )
    return Sim3(scale=scale, R=rot, t=t)


def gravity_and_gps_alignment(
    rotations: dict[int, np.ndarray],
    centers: np.ndarray,
    gps: np.ndarray,
    have_gps: np.ndarray,
):
    """Sim(3) to metric ENU without inheriting a degenerate tilt.

    A plain Umeyama fit on camera centers leaves rotations about the
    trajectory's thin axes unconstrained (a walk is nearly planar, so
    the fitted rotation can carry an arbitrary tilt that the centers
    absorb but every orientation inherits). Instead: (1) level the model
    with the median camera up-vector (the production pipeline's
    ground-plane alignment), which pins the two in-plane rotation DOF,
    then (2) fit yaw + scale + translation to the GPS track (the
    horizontal-alignment convention of ml_utils.batch_registration).

    Returns an ml_utils Sim3 or None when GPS is unusable.
    """
    from ml_utils.sim3 import Sim3

    u_level = leveling_rotation(rotations)
    if have_gps.sum() < 4:
        return None
    c_lvl = centers[have_gps] @ u_level.T
    g = gps[have_gps]

    # Robust yaw + scale + translation (one IRLS trim pass).
    keep = np.ones(len(g), dtype=bool)
    for _ in range(3):
        src, dst = c_lvl[keep, :2], g[keep, :2]
        mu_s, mu_d = src.mean(0), dst.mean(0)
        s_c, d_c = src - mu_s, dst - mu_d
        cov = d_c.T @ s_c / len(src)
        u_svd, dvals, vt = np.linalg.svd(cov)
        sgn = np.sign(np.linalg.det(u_svd @ vt))
        r2 = u_svd @ np.diag([1.0, sgn]) @ vt
        var = (s_c**2).sum() / len(src)
        scale = float((dvals * np.array([1.0, sgn])).sum() / max(var, 1e-12))
        t_xy = mu_d - scale * (r2 @ mu_s)
        resid = np.linalg.norm(
            (scale * (c_lvl[:, :2] @ r2.T) + t_xy) - g[:, :2], axis=1
        )
        threshold = max(3.0 * np.median(resid[keep]), 0.05)
        new_keep = resid < threshold
        if new_keep.sum() < 4 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= 4 else keep
            break
        keep = new_keep

    t_z = float(np.mean(g[keep, 2] - scale * c_lvl[keep, 2]))
    r_yaw = np.eye(3)
    r_yaw[:2, :2] = r2
    tilt_deg = np.degrees(
        np.arccos(np.clip((np.trace(u_level) - 1.0) / 2.0, -1.0, 1.0))
    )
    logger.info(
        "gravity+GPS alignment: scale %.4f, %d/%d inliers, "
        "xy rmse %.3f m, up tilt corrected %.2f deg",
        scale,
        int(keep.sum()),
        len(g),
        float(np.sqrt(np.mean(resid[keep] ** 2))),
        float(tilt_deg),
    )
    return Sim3(
        scale=scale,
        R=r_yaw @ u_level,
        t=np.array([t_xy[0], t_xy[1], t_z]),
    )


def smooth_gps_deltas(
    centers: np.ndarray,
    gps: np.ndarray,
    have_gps: np.ndarray,
    w_gps: float = 1.0,
    w_smooth: float = 1.0,
    robust_k: float = 10.0,
    iters: int = 5,
) -> np.ndarray:
    """Smooth per-frame translation field pulling centers onto GPS.

    Same construction as batch_registration.refine_per_frame_deltas
    (first-difference smoothness + IRLS with Cauchy influence), applied
    to all three axes: the global rigid alignment cannot remove
    low-frequency drift of the visual chain, and the downstream BA's
    Huber-damped priors only partially pull it. Frames without GPS ride
    the smoothness term.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse import vstack as sparse_vstack
    from scipy.sparse.linalg import lsqr

    n = len(centers)
    rows = np.repeat(np.arange(n - 1), 2)
    cols = np.empty(2 * (n - 1), dtype=np.int64)
    cols[0::2] = np.arange(1, n)
    cols[1::2] = np.arange(n - 1)
    vals = np.tile(np.array([1.0, -1.0]), n - 1)
    smooth = csr_matrix((vals, (rows, cols)), shape=(n - 1, n))

    corr = np.zeros((n, 3))
    weights = have_gps.astype(np.float64)
    target = np.where(have_gps[:, None], gps - centers, 0.0)
    for _ in range(iters):
        wg = np.sqrt(w_gps * weights)
        a_top = csr_matrix((wg, (np.arange(n), np.arange(n))), shape=(n, n))
        amat = sparse_vstack([a_top, np.sqrt(w_smooth) * smooth]).tocsr()
        for ax in range(3):
            rhs = np.concatenate([wg * target[:, ax], np.zeros(n - 1)])
            corr[:, ax] = lsqr(amat, rhs)[0]
        res = np.linalg.norm((centers + corr) - gps, axis=1)
        r = res[have_gps]
        mad = float(np.median(np.abs(r - np.median(r)))) + 1e-9
        scale = robust_k * 1.4826 * mad
        weights = np.zeros(n)
        weights[have_gps] = 1.0 / (
            1.0 + (res[have_gps] / max(scale, 1e-6)) ** 2
        )
    moved = np.linalg.norm(corr, axis=1)
    logger.info(
        "smooth GPS deltas: median %.3f m, max %.3f m",
        float(np.median(moved)),
        float(moved.max()),
    )
    return corr


def pin_unanchored_frames(
    c2w_all: np.ndarray,
    names: list[str],
    stars: list[dict],
    gps: np.ndarray,
    have_gps: np.ndarray,
    min_baseline: float = 0.3,
) -> int:
    """Pin frames observed only through coincident views to their priors.

    A frame whose every star membership pairs it solely with frames standing
    at the same spot (by prior) has no parallax evidence: the network's
    relative translations for identical views are unconstrained, so its
    averaged position is a hallucination the robust layers downstream then
    *protect* (observed: two stationary capture-tail frames parked 36 m out,
    Cauchy-downweighted by the delta refine and Huber-saturated in the BA).
    Such frames take their prior's xy and the z of the nearest anchored
    frame in capture order. Returns the number of frames pinned.
    """
    idx_of = {name: i for i, name in enumerate(names)}
    anchored = np.zeros(len(names), dtype=bool)
    for star in stars:
        rows = [idx_of[n] for n in star["names"] if n in idx_of]
        for a in rows:
            if anchored[a] or not have_gps[a]:
                continue
            for b in rows:
                if (
                    b != a
                    and have_gps[b]
                    and np.linalg.norm(gps[a] - gps[b]) >= min_baseline
                ):
                    anchored[a] = True
                    break
    pinned = 0
    anchored_rows = np.nonzero(anchored)[0]
    for i in range(len(names)):
        if anchored[i] or not have_gps[i]:
            continue
        z = c2w_all[i, 2, 3]
        if anchored_rows.size:
            z = c2w_all[anchored_rows[np.argmin(np.abs(anchored_rows - i))], 2, 3]
        logger.warning(
            "%s has no baseline evidence (coincident star members only); "
            "pinning to its prior",
            names[i],
        )
        c2w_all[i, :2, 3] = gps[i, :2]
        c2w_all[i, 2, 3] = z
        pinned += 1
    if pinned:
        logger.info("Pinned %d unanchored frames to their priors", pinned)
    return pinned


def load_stars(star_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((star_dir / "stars.json").read_text())
    stars = []
    for i in range(len(meta["stars"])):
        data = np.load(star_dir / "stars" / f"star_{i:04d}.npz")
        stars.append(
            {
                "names": [str(x) for x in data["names"]],
                "extr": np.asarray(data["extr"], dtype=np.float64),
                "depth": np.asarray(data["depth"]),  # float16
            }
        )
    return meta, stars


def solve(
    star_dir: Path,
    out_dir: Path,
    min_edge_score: float = 0.15,
    covis_tol: float = 0.05,
    points_per_star: int = 2000,
    start_end_json: Path | None = None,
) -> None:
    import torch
    from ml_utils.star_solve import (
        initialize_mst_structures,
        similarity_averaging,
    )
    from ml_utils.star_solve import (
        rotation_averaging as rotation_averaging_pycolmap,
    )

    meta, stars = load_stars(star_dir)
    names = meta["frames"]
    idx_of = {name: i for i, name in enumerate(names)}

    # --- predictions_dict in GLUEMAP's star contract -----------------
    predictions: dict = {
        "indexes": [],
        "pose_scores": [],
        "extrinsics": [],
        "points3d_virtual": [],
    }
    for star in stars:
        scores = compute_star_scores(star["extr"], star["depth"], tol=covis_tol)
        scores[1:][scores[1:] < min_edge_score] = 0.0
        pts, _ = _unproject_valid(
            star["depth"][0].astype(np.float32), stride=14
        )
        if pts.shape[0] > points_per_star:
            sel = np.random.default_rng(0).choice(
                pts.shape[0], points_per_star, replace=False
            )
            pts = pts[sel]
        predictions["indexes"].append([idx_of[n] for n in star["names"]])
        predictions["pose_scores"].append(
            torch.from_numpy(scores[None].astype(np.float32))
        )
        predictions["extrinsics"].append(
            torch.from_numpy(star["extr"][None].astype(np.float64))
        )
        predictions["points3d_virtual"].append(
            torch.from_numpy(pts[None].astype(np.float64))
        )
    num_edges = int(
        sum((s[0, 1:] > 0).sum().item() for s in predictions["pose_scores"])
    )
    logger.info(
        "%d stars, %d valid edges (min score %.2f)",
        len(stars),
        num_edges,
        min_edge_score,
    )

    # --- GLUEMAP global mapping --------------------------------------
    rotations = rotation_averaging_pycolmap(predictions)
    centers0, scales0 = initialize_mst_structures(predictions, rotations)
    scales_list = [
        np.ones(1, dtype=np.float64) * scales0.get(k, 1.0)
        for k in range(len(stars))
    ]
    centers = similarity_averaging(
        predictions, rotations, centers0, scales_list
    )
    solved_scales = np.array([s[0] for s in scales_list])
    logger.info(
        "similarity averaging: star scales median %.3f (range %.3f..%.3f)",
        float(np.median(solved_scales)),
        float(solved_scales.min()),
        float(solved_scales.max()),
    )

    # --- metric ENU: gravity leveling + horizontal GPS fit ------------
    centers_arr = np.stack([centers[i] for i in range(len(names))])
    gps_arr = np.full((len(names), 3), np.nan)
    for i, name in enumerate(names):
        if meta["enu"].get(name) is not None:
            gps_arr[i] = np.asarray(meta["enu"][name], dtype=np.float64)
    have_gps = np.all(np.isfinite(gps_arr), axis=1)
    sim3 = gravity_and_gps_alignment(rotations, centers_arr, gps_arr, have_gps)
    use_gps_deltas = sim3 is not None
    if sim3 is None and start_end_json is not None and start_end_json.exists():
        points = json.loads(start_end_json.read_text())
        if points.get("start_point") is not None and points.get("end_point"):
            sim3 = start_end_alignment(
                rotations,
                centers_arr,
                np.asarray(points["start_point"], dtype=np.float64),
                np.asarray(points["end_point"], dtype=np.float64),
            )
    if sim3 is None:
        from ml_utils.sim3 import Sim3

        logger.warning(
            "No GPS and no start/end anchors; output is gravity-leveled "
            "but up-to-scale"
        )
        sim3 = Sim3(scale=1.0, R=leveling_rotation(rotations), t=np.zeros(3))
    scale_g = float(sim3.scale)

    c2w_all = np.zeros((len(names), 4, 4))
    for i in range(len(names)):
        c2w = np.eye(4)
        c2w[:3, :3] = rotations[i].T
        c2w[:3, 3] = centers[i]
        c2w_all[i] = sim3.apply_pose(c2w)

    if use_gps_deltas:
        deltas = smooth_gps_deltas(c2w_all[:, :3, 3].copy(), gps_arr, have_gps)
        c2w_all[:, :3, 3] += deltas
        pin_unanchored_frames(c2w_all, names, stars, gps_arr, have_gps)

    # --- per-frame metric depth: median over observing stars ---------
    # global = star_local / s_star (similarity averaging convention),
    # then * scale_g for the ENU frame.
    obs_of: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for star_idx, star in enumerate(stars):
        for slot, name in enumerate(star["names"]):
            obs_of[idx_of[name]].append((star_idx, slot))

    out_dir.mkdir(parents=True, exist_ok=True)
    dmaps_dir = out_dir / "dmaps"
    dmaps_dir.mkdir(exist_ok=True)
    for i, name in enumerate(names):
        layers = []
        for star_idx, slot in obs_of[i]:
            factor = scale_g / max(solved_scales[star_idx], 1e-9)
            layer = stars[star_idx]["depth"][slot].astype(np.float32) * factor
            layer[layer <= 0] = np.nan
            layers.append(layer)
        stack = np.stack(layers)
        with np.errstate(all="ignore"):
            merged = np.nanmedian(stack, axis=0)
        merged[~np.isfinite(merged)] = 0.0
        np.save(dmaps_dir / f"{name}.npy", merged.astype(np.float32))

    np.savez(
        out_dir / "poses.npz",
        c2w=c2w_all,
        frame_names=np.array(names),
    )
    logger.info("Wrote %d poses + merged dmaps to %s", len(names), out_dir)

    from ml_utils.batch_registration import (
        BatchRegistrationConf,
        write_neighbors_json,
    )

    write_neighbors_json(out_dir, BatchRegistrationConf())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--star_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--min_edge_score", type=float, default=0.15)
    parser.add_argument("--covis_tol", type=float, default=0.05)
    parser.add_argument(
        "--start_end_json",
        type=Path,
        default=None,
        help="start_end_points.json for GPS-less plan-frame alignment",
    )
    args = parser.parse_args()
    solve(
        args.star_dir,
        args.out_dir,
        min_edge_score=args.min_edge_score,
        covis_tol=args.covis_tol,
        start_end_json=args.start_end_json,
    )


if __name__ == "__main__":
    main()
