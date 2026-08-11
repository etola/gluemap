#!/usr/bin/env python3
"""Floor-plan alignment for the equirect pipeline (Option B port).

Standalone port of the production floor-plan bundle adjustment
(photogrammetry ``threedn_service/core/workflows/floorplan_ba.py`` +
``floorplan_tracks.py``) onto this pipeline's workfolder contract. The
production stage snaps pano-depth wall points onto the customer floor
plan with a 2D GTSAM SE2 solve; this port works directly in the metric
ENU frame (``metric_scale = 1``) and replaces the GeoTIFF machinery
with a plain numpy affine, so rasterio is not needed.

Scope:
  M1  georeference the plan into ENU, build the wall-SDF, extract
      per-frame wall points from the registered depth maps, and
      report/visualize wall->plan agreement under the current poses.
  M2  (--solve) the SE2 GTSAM optimization: odometry + per-frame
      priors + normal-gated robust SDF factors, with the production
      skip-guard, writing corrected_poses.npz next to the reports.
  M3  structure factors (SIFT track pool + covisibility from the
      refined COLMAP model when --model_dir resolves, virtual tracks
      over neighbors.json otherwise) keeping the walk rigid while the
      SDF snaps it, and --apply writing the passing correction back
      (model rig frames, or poses.npz for the model-less source).
      Orchestrated by pano_panovggt_pipeline's floorplan stage
      (--floorplan_plan/--floorplan_anchors), between refine and
      correct.

Inputs:
  --workfolder      pipeline workfolder (reads registered/ poses+dmaps,
                    falling back to initial_registration/, and
                    reference_lla.json for LLA anchors)
  --plan            floor-plan image (PNG/JPG, walls as dark linework)
  --plan_anchors    JSON georeferencing the plan's four corners in
                    UL, UR, LR, LL order (the production overlay
                    convention):
                      {"corners_lla": [[lon, lat], x4]}   or
                      {"corners_enu": [[e, n], x4]}

Outputs (to --out_dir, default workfolder/floorplan_ba):
  report_floorplan_alignment.png   walls + trajectory over plan edges
  report_floorplan_frame_sdf.png   trajectory colored by mean |SDF|
  floorplan_report.json            per-frame + aggregate |SDF| metrics
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (SCRIPTS_DIR, SCRIPTS_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

logger = logging.getLogger(__name__)

# --- plan raster / SDF (constants mirror the production stage) ------------
EDGE_THRESHOLD = 100.0  # Sobel magnitude; low keeps wall lines continuous
MIN_EDGE_COMPONENT = 8  # drop tiny edge specks (text dots)
PLAN_PIXEL_SIZE = 0.05  # m/px of the ENU-warped plan raster
# --- per-keyframe wall sampling -------------------------------------------
POINTS_PER_FRAME = 80
WALL_NZ_MAX = 0.30  # |normal_z| below this => vertical surface
MIN_WALL_POINTS = 30  # frames with fewer candidate wall pixels are skipped
# --- SE2 solve (constants mirror the production stage) --------------------
ODOM_SIGMA = (0.02, 0.02, np.radians(0.5))
FRAME_PRIOR_SIGMA = (1.0, 1.0, np.radians(10.0))
TRACK_BEARING_SIGMA = np.radians(1.0)
TRACK_RANGE_SIGMA = 0.03
VT_BEARING_SIGMA = np.radians(2.0)
VT_RANGE_SIGMA = 0.05
VT_ROBUST_K = 1.0  # Cauchy scale (whitened units)
SDF_SIGMA = 0.20
SDF_HUBER = 0.20  # m: robustifier knee
SDF_REJECT = 0.50  # m: beyond this a point is treated as unmapped clutter
SDF_GATE_COS = 0.5  # normal gating: recon-normal . plan-normal
MIN_FRAMES = 20
# Skip the correction if it doesn't clearly improve alignment or moves a
# wide swath of frames far. Metrics are still recorded when skipped.
CORRECTION_P90_M = 1.0
# --- robustness: global pre-alignment + out-of-sheet gating ---------------
PREALIGN_TRIM = 1.0  # m: |SDF| clip in the coarse-search objective
PREALIGN_MIN_GAIN = 0.05  # apply only if the trimmed score improves >= 5%
PREALIGN_MAX_POINTS = 30000
# Corridors are translation-invariant along their own axis, so distant
# minima can alias. The search stays within plausible anchor error and a
# movement penalty makes the NEAREST minimum win ties.
PREALIGN_T_RANGE = 2.0  # m
PREALIGN_PENALTY_T = 0.02  # score units per meter of shift
PREALIGN_PENALTY_YAW = 0.3  # score units per radian of yaw
# A frame keeps its SDF factor only while its fraction of wall points
# near drawn ink stays above this share of the capture's median fraction.
OFFSHEET_REL_FRACTION = 0.5
OFFSHEET_MIN_MEDIAN = 0.2  # below this the whole capture mismatches; no gate
# --- structure: track/virtual bearing-range factors (production values) ---
TRACK_MIN_OBS = 3
# Frame-scaled cap keeps BA solve cost proportional to trajectory size.
TRACKS_PER_FRAME = 8
TRACK_CAP_MIN = 4000
TRACK_CAP_MAX = 16000
# Frames below this get priority in coverage-first selection.
COVERAGE_TARGET = 30
COVIS_MIN_SHARED_POINTS = 30  # shared 3D points for two panos to covis
VT_DEPTH_TOL = 0.10  # m: reprojected radial-depth agreement gate
VT_MAX_PER_ANCHOR = 250  # cap virtual landmarks emitted per anchor frame
VT_MIN_OBS = 2  # min observing frames for a virtual track
VT_ANCHOR_PTS = 300  # dense sample of the whole filtered cloud
VT_MIN_HORIZ = 0.30  # m: drop near-nadir (degenerate) points


# ==========================================================================
#  Plan georeferencing (replaces georeference_image + resample_geotiff)
# ==========================================================================


@dataclass(slots=True)
class PlanRaster:
    """North-up ENU plan raster: ``E = e0 + px*col``, ``N = n0 - px*row``."""

    gray: np.ndarray  # (H, W) float64 grayscale
    e0: float  # ENU east of pixel (col=0, row=0)
    n0: float  # ENU north of pixel (col=0, row=0)
    px: float  # meters per pixel

    def enu_to_pixel(
        self, east: np.ndarray, north: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (east - self.e0) / self.px, (self.n0 - north) / self.px


def _corners_to_enu(anchors: dict, reference_lla: dict | None) -> np.ndarray:
    """Anchor JSON -> (4, 2) ENU corners in UL, UR, LR, LL order."""
    if "corners_enu" in anchors:
        corners = np.asarray(anchors["corners_enu"], dtype=np.float64)
    elif "corners_lla" in anchors:
        if reference_lla is None:
            raise ValueError(
                "corners_lla anchors need workfolder/reference_lla.json"
            )
        import pymap3d

        corners = np.array(
            [
                pymap3d.geodetic2enu(
                    lat,
                    lon,
                    0.0,
                    reference_lla["latitude"],
                    reference_lla["longitude"],
                    0.0,
                )[:2]
                for lon, lat in anchors["corners_lla"]
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError("plan anchors need corners_enu or corners_lla")
    if corners.shape != (4, 2):
        raise ValueError(f"expected 4 corner points, got {corners.shape}")
    return corners


def load_plan(
    plan_path: Path, anchors: dict, reference_lla: dict | None
) -> PlanRaster:
    """Warp the plan image into an axis-aligned ENU raster at 0.05 m/px.

    The four pixel corners (UL, UR, LR, LL — the production overlay
    order) are mapped to their ENU positions with a least-squares 2D
    affine, then the image is resampled onto a north-up metric grid so
    the SDF's Euclidean distance transform is isotropic in meters
    (production achieves the same via resample_geotiff).
    """
    import cv2

    bgr = cv2.imread(str(plan_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(plan_path)
    gray = bgr.astype(np.float64).mean(axis=2)
    h, w = gray.shape

    corners_enu = _corners_to_enu(anchors, reference_lla)
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    # Least-squares affine pixel -> ENU from the 4 correspondences.
    amat = np.hstack([corners_px, np.ones((4, 1))])
    coef, _, _, _ = np.linalg.lstsq(amat, corners_enu, rcond=None)
    resid = amat @ coef - corners_enu
    logger.info(
        "plan affine: %.3f m/px east, %.3f m/px north, corner rmse %.3f m",
        float(np.hypot(coef[0, 0], coef[0, 1])),
        float(np.hypot(coef[1, 0], coef[1, 1])),
        float(np.sqrt(np.mean(resid**2))),
    )

    px = PLAN_PIXEL_SIZE
    e_min, n_min = corners_enu.min(axis=0)
    e_max, n_max = corners_enu.max(axis=0)
    out_w = int(np.ceil((e_max - e_min) / px)) + 1
    out_h = int(np.ceil((n_max - n_min) / px)) + 1
    e0, n0 = float(e_min), float(n_max)

    # Destination pixel -> ENU -> source pixel, expressed as the 2x3
    # inverse map cv2.warpAffine expects with WARP_INVERSE_MAP.
    a_fwd = np.vstack([coef.T, [0.0, 0.0, 1.0]])  # pixel -> ENU, 3x3
    a_inv = np.linalg.inv(a_fwd)
    dst_to_enu = np.array([[px, 0.0, e0], [0.0, -px, n0], [0.0, 0.0, 1.0]])
    m = (a_inv @ dst_to_enu)[:2]
    warped = cv2.warpAffine(
        gray,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255.0,  # outside the plan = blank paper, no edges
    )
    logger.info(
        "plan raster: %dx%d px (%.1f x %.1f m) at %.2f m/px",
        out_w,
        out_h,
        out_w * px,
        out_h * px,
        px,
    )
    return PlanRaster(gray=warped, e0=e0, n0=n0, px=px)


# ==========================================================================
#  Wall SDF (port of build_wall_sdf / _sdf_sampler, rasterio-free)
# ==========================================================================


@dataclass(slots=True)
class _Sdf:
    """Metric distance-to-wall field + gradient on the plan raster grid."""

    dist: np.ndarray
    grad_col: np.ndarray
    grad_row: np.ndarray
    plan: PlanRaster


def plan_edges(gray: np.ndarray) -> np.ndarray:
    """Boolean wall-line mask from the plan's grayscale (also for plots).

    Sobel edges are robust to the busy CAD linework customers upload
    (no clean wall layer required); tiny components are dropped.
    """
    edges = (
        np.hypot(ndimage.sobel(gray, 0), ndimage.sobel(gray, 1))
        > EDGE_THRESHOLD
    )
    labels, n = ndimage.label(edges)
    if n:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        edges = np.isin(labels, np.where(sizes >= MIN_EDGE_COMPONENT)[0])
    return edges


def build_wall_sdf(plan: PlanRaster) -> _Sdf:
    """Distance-to-wall field (meters) from the ENU-warped plan raster."""
    edges = plan_edges(plan.gray)
    dist = ndimage.distance_transform_edt(~edges) * plan.px
    grad_row, grad_col = np.gradient(dist)
    return _Sdf(dist, grad_col, grad_row, plan)


def sdf_sampler(sdf: _Sdf):
    """Return ``sample(E, N) -> (dist, dE, dN)`` with bilinear interpolation."""
    plan = sdf.plan
    h, w = sdf.dist.shape
    inv_px = 1.0 / plan.px

    def _bilinear(arr, col, row):
        col = np.clip(col, 0, w - 1.001)
        row = np.clip(row, 0, h - 1.001)
        c0 = np.floor(col).astype(int)
        r0 = np.floor(row).astype(int)
        fc = col - c0
        fr = row - r0
        return (
            arr[r0, c0] * (1 - fc) * (1 - fr)
            + arr[r0, c0 + 1] * fc * (1 - fr)
            + arr[r0 + 1, c0] * (1 - fc) * fr
            + arr[r0 + 1, c0 + 1] * fc * fr
        )

    def sample(east: np.ndarray, north: np.ndarray):
        col, row = plan.enu_to_pixel(east, north)
        value = _bilinear(sdf.dist, col, row)
        gcol = _bilinear(sdf.grad_col, col, row)
        grow = _bilinear(sdf.grad_row, col, row)
        # d(col)/dE = 1/px, d(row)/dN = -1/px (north-up raster).
        return value, gcol * inv_px, -grow * inv_px

    return sample


# ==========================================================================
#  Per-frame wall extraction (port of floorplan_tracks.extract_frame_walls)
# ==========================================================================


@dataclass(slots=True)
class FrameWalls:
    """Per-keyframe data for the alignment/BA, all in meters (ENU)."""

    frame_id: int
    name: str
    x: float
    y: float
    yaw: float
    wall_body: np.ndarray  # (K, 2) wall points in the frame's ground frame
    normal_body: np.ndarray  # (K, 2) unit wall normals, same frame


def _rotate_2d(vecs: np.ndarray, angle: float) -> np.ndarray:
    ca, sa = np.cos(angle), np.sin(angle)
    return np.column_stack(
        [ca * vecs[:, 0] - sa * vecs[:, 1], sa * vecs[:, 0] + ca * vecs[:, 1]]
    )


def _frame_normals(
    points_grid: np.ndarray, camera_center: np.ndarray
) -> np.ndarray:
    """Per-pixel world normals from the (H, W, 3) point grid, camera-facing."""
    du = np.zeros_like(points_grid)
    dv = np.zeros_like(points_grid)
    du[:, :-1] = points_grid[:, 1:] - points_grid[:, :-1]
    du[:, -1] = du[:, -2]
    dv[:-1, :] = points_grid[1:, :] - points_grid[:-1, :]
    dv[-1, :] = dv[-2, :]
    normals = np.cross(du, dv)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True) + 1e-9
    flip = np.sum(normals * (camera_center - points_grid), axis=2) < 0
    normals[flip] *= -1.0
    return normals


def _z_band(centers: np.ndarray) -> tuple[float, float]:
    """Height band (m) keeping wall points, dropping floor/ceiling."""
    z_med = float(np.median(centers[:, 2]))
    return z_med - 1.0, z_med + 0.4


def load_frame_infos(registered_dir: Path) -> list[dict]:
    """poses.npz (c2w) -> the frame-info dicts ml_utils' unprojection uses.

    ``R``/``t`` are cam_from_world (``world = (cam - t) @ R``), matching
    ``depth_map_to_world_points``; ``c`` is the camera center.
    """
    poses = np.load(registered_dir / "poses.npz")
    infos = []
    for i, name in enumerate(str(n) for n in poses["frame_names"]):
        c2w = np.asarray(poses["c2w"][i], dtype=np.float64)
        r_w2c = c2w[:3, :3].T
        center = c2w[:3, 3]
        infos.append(
            {
                "R": r_w2c,
                "t": -r_w2c @ center,
                "c": center,
                "keyframe_name": name,
                "frame_id": i,
            }
        )
    return infos


def extract_frame_walls(
    frame_infos: list[dict],
    dmap_dir: Path,
    rng: np.random.Generator,
    stride: int = 4,
) -> list[FrameWalls]:
    """Per-keyframe wall points + normals + init SE2 pose, in ENU meters.

    Same construction as the production stage: unproject the radial
    equirect depth map, keep pixels whose surface normal is vertical
    (|nz| < 0.3) inside the walking-height z band, and sample down to
    ``POINTS_PER_FRAME`` per frame. ``stride`` downsamples the depth
    grid before the normal computation for speed (nearest-neighbor,
    preserving the equirect 2:1 aspect the unprojection requires).
    """
    import cv2
    from ml_utils.depth_data import DepthProjection
    from ml_utils.utils import depth_map_to_world_points

    centers = np.stack(
        [np.asarray(i["c"], dtype=np.float64) for i in frame_infos]
    )
    zmin, zmax = _z_band(centers)
    frames: list[FrameWalls] = []
    for info in frame_infos:
        stem = Path(info["keyframe_name"]).stem
        dmap_path = dmap_dir / f"{stem}.npy"
        if not dmap_path.exists():
            continue
        dmap = np.load(dmap_path).astype(np.float64)
        if stride > 1:
            h2 = dmap.shape[0] // stride
            dmap = cv2.resize(
                dmap, (2 * h2, h2), interpolation=cv2.INTER_NEAREST
            )
        points = depth_map_to_world_points(
            info, dmap, DepthProjection.EQUIRECTANGULAR
        )
        center = np.asarray(info["c"], dtype=np.float64)
        normals = _frame_normals(points, center)

        flat_pts = points.reshape(-1, 3)
        flat_nrm = normals.reshape(-1, 3)
        depth = dmap.ravel()
        ok = (
            np.isfinite(depth)
            & (depth > 0)
            & np.isfinite(flat_pts).all(1)
            & (flat_pts[:, 2] > zmin)
            & (flat_pts[:, 2] < zmax)
            & (np.abs(flat_nrm[:, 2]) < WALL_NZ_MAX)
        )
        idx = np.where(ok)[0]
        if len(idx) < MIN_WALL_POINTS:
            continue
        if len(idx) > POINTS_PER_FRAME:
            idx = idx[rng.choice(len(idx), POINTS_PER_FRAME, replace=False)]

        world_xy = flat_pts[idx, :2]
        normal_xy = flat_nrm[idx, :2]
        normal_xy /= np.linalg.norm(normal_xy, axis=1, keepdims=True) + 1e-9
        forward = np.asarray(info["R"]).T @ np.array([0.0, 0.0, 1.0])
        yaw = float(np.arctan2(forward[1], forward[0]))
        frames.append(
            FrameWalls(
                frame_id=int(info["frame_id"]),
                name=stem,
                x=float(center[0]),
                y=float(center[1]),
                yaw=yaw,
                wall_body=_rotate_2d(world_xy - center[:2], -yaw),
                normal_body=_rotate_2d(normal_xy, -yaw),
            )
        )
    frames.sort(key=lambda f: f.frame_id)
    return frames


# ==========================================================================
#  SE2 solve (M2: odometry + frame priors + normal-gated SDF factors;
#  track/virtual bearing-range structure factors arrive in M3)
# ==========================================================================


def apply_se2_to_frames(
    frames: list[FrameWalls], theta: float, t: np.ndarray
) -> list[FrameWalls]:
    """Frames under the world SE2 ``p' = R(theta) p + t`` (walls are
    body-frame, so only the pose fields change)."""
    c, s = np.cos(theta), np.sin(theta)
    out = []
    for f in frames:
        out.append(
            FrameWalls(
                frame_id=f.frame_id,
                name=f.name,
                x=float(c * f.x - s * f.y + t[0]),
                y=float(s * f.x + c * f.y + t[1]),
                yaw=f.yaw + theta,
                wall_body=f.wall_body,
                normal_body=f.normal_body,
            )
        )
    return out


def coarse_prealign(
    frames: list[FrameWalls],
    sample_fn,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, dict]:
    """Global SE2 chamfer search: snap ALL wall points onto the plan at once.

    The per-frame solve cannot recover an initial error beyond the SDF
    reject gate (0.5 m) — a rotated or offset capture starts with every
    residual gated off. This coarse-to-fine grid search over (yaw about
    the wall centroid, translation) minimizes the trimmed mean
    ``min(|SDF|, PREALIGN_TRIM)`` over a subsample of all wall points,
    so out-of-sheet points saturate instead of dominating. Returns
    ``(theta, t, info)`` — the world SE2 (identity when the search
    cannot beat the initial score by ``PREALIGN_MIN_GAIN``).
    """
    pts = np.vstack(walls_world(frames, [(f.x, f.y, f.yaw) for f in frames]))
    if len(pts) > PREALIGN_MAX_POINTS:
        pts = pts[rng.choice(len(pts), PREALIGN_MAX_POINTS, replace=False)]
    centroid = pts.mean(axis=0)

    def score(theta: float, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        rel = pts - centroid
        rx = c * rel[:, 0] - s * rel[:, 1] + centroid[0]
        ry = s * rel[:, 0] + c * rel[:, 1] + centroid[1]
        east = rx[None, :] + dx[:, None]
        north = ry[None, :] + dy[:, None]
        vals = np.abs(sample_fn(east, north)[0])
        return np.minimum(vals, PREALIGN_TRIM).mean(axis=1)

    base = float(score(0.0, np.zeros(1), np.zeros(1))[0])
    best = (0.0, 0.0, 0.0, base)  # tracked on the PENALIZED score
    stages = [
        (np.radians(8.0), np.radians(1.0), PREALIGN_T_RANGE, 0.5),
        (np.radians(1.0), np.radians(0.25), 0.6, 0.1),
    ]
    for yaw_range, yaw_step, t_range, t_step in stages:
        yaw0, dx0, dy0 = best[0], best[1], best[2]
        yaws = yaw0 + np.arange(-yaw_range, yaw_range + 1e-9, yaw_step)
        offs = np.arange(-t_range, t_range + 1e-9, t_step)
        dxs = (dx0 + offs[:, None]).repeat(len(offs), 1).ravel()
        dys = np.tile(dy0 + offs, len(offs))
        for theta in yaws:
            sc = (
                score(float(theta), dxs, dys)
                + PREALIGN_PENALTY_T * np.hypot(dxs, dys)
                + PREALIGN_PENALTY_YAW * abs(float(theta))
            )
            k = int(np.argmin(sc))
            if sc[k] < best[3]:
                best = (
                    float(theta),
                    float(dxs[k]),
                    float(dys[k]),
                    float(sc[k]),
                )
    theta, dx, dy, _ = best
    sc = float(score(theta, np.array([dx]), np.array([dy]))[0])
    gain = (base - sc) / max(base, 1e-9)  # gain judged on the RAW score
    info = {
        "score_before": base,
        "score_after": sc,
        "gain": gain,
        "yaw_deg": float(np.degrees(theta)),
        "shift_m": [dx, dy],
        "applied": bool(gain >= PREALIGN_MIN_GAIN),
    }
    if not info["applied"]:
        return 0.0, np.zeros(2), info
    # rotation about the centroid + shift, expressed as p' = R p + t
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    t = centroid - rot @ centroid + np.array([dx, dy])
    return theta, t, info


def sdf_factor(key: int, walls: FrameWalls, sample_fn):
    """GTSAM CustomFactor: normal-gated, robust wall->plan distance.

    Port of the production ``_sdf_factor``: Huber weights above
    ``SDF_HUBER``, hard rejection above ``SDF_REJECT`` (unmapped
    clutter), and a normal gate dropping points whose reconstructed
    wall normal disagrees with the SDF gradient (plan-normal). The
    numeric jacobian holds the weights fixed (IRLS-style).
    """
    import gtsam

    q0, q1 = walls.wall_body[:, 0], walls.wall_body[:, 1]
    n0, n1 = walls.normal_body[:, 0], walls.normal_body[:, 1]
    k = len(q0)
    noise = gtsam.noiseModel.Isotropic.Sigma(k, SDF_SIGMA)

    def world(pose):
        c, s = np.cos(pose.theta()), np.sin(pose.theta())
        return pose.x() + c * q0 - s * q1, pose.y() + s * q0 + c * q1

    def weights(pose, residual, grad_e, grad_n) -> np.ndarray:
        a = np.abs(residual)
        w = np.ones_like(residual)
        big = a > SDF_HUBER
        w[big] = SDF_HUBER / a[big]
        w[a > SDF_REJECT] = 0.0
        c, s = np.cos(pose.theta()), np.sin(pose.theta())
        nwe = c * n0 - s * n1
        nwn = s * n0 + c * n1
        gmag = np.hypot(grad_e, grad_n) + 1e-9
        agree = (nwe * grad_e + nwn * grad_n) / gmag
        w[agree < SDF_GATE_COS] = 0.0
        return np.sqrt(w)

    def error(_this, values, jacobians=None):
        pose = values.atPose2(key)
        wx, wy = world(pose)
        residual, grad_e, grad_n = sample_fn(wx, wy)
        sw = weights(pose, residual, grad_e, grad_n)
        weighted = sw * residual
        if jacobians is not None:
            jac = np.zeros((k, 3))
            eps = 1e-4
            for dim in range(3):
                delta = np.zeros(3)
                delta[dim] = eps
                wx2, wy2 = world(pose.retract(delta))
                r2, _, _ = sample_fn(wx2, wy2)
                jac[:, dim] = (sw * r2 - weighted) / eps
            jacobians[0] = jac
        return weighted

    return gtsam.CustomFactor(noise, [key], error)


def optimize_poses(
    frames: list[FrameWalls],
    sample_fn,
    landmarks: dict[int, np.ndarray] | None = None,
    colmap_obs: list[tuple[int, int, float, float]] = (),
    virtual_obs: list[tuple[int, int, float, float]] = (),
    sdf_mask: np.ndarray | None = None,
):
    """2D BA: odometry + per-frame prior + normal-gated SDF, single pass.

    Port of the production ``optimize_poses``. ``landmarks`` +
    ``colmap_obs``/``virtual_obs`` (frame_idx, lm_key, bearing, range)
    add the structure BearingRange factors and are wired up by the M3
    structure builder; with none given, the solve is SDF + odometry +
    priors only. Returns ``(init, result)`` gtsam.Values.
    """
    import gtsam

    init = gtsam.Values()
    for i, f in enumerate(frames):
        init.insert(i, gtsam.Pose2(f.x, f.y, f.yaw))
    for lm_key, xy in (landmarks or {}).items():
        init.insert(lm_key, gtsam.Point2(float(xy[0]), float(xy[1])))

    odom = gtsam.noiseModel.Diagonal.Sigmas(np.array(ODOM_SIGMA))
    frame_prior = gtsam.noiseModel.Diagonal.Sigmas(np.array(FRAME_PRIOR_SIGMA))
    track_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([TRACK_BEARING_SIGMA, TRACK_RANGE_SIGMA])
    )
    vt_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(VT_ROBUST_K),
        gtsam.noiseModel.Diagonal.Sigmas(
            np.array([VT_BEARING_SIGMA, VT_RANGE_SIGMA])
        ),
    )

    graph = gtsam.NonlinearFactorGraph()
    for i in range(len(frames) - 1):
        xi = init.atPose2(i)
        xj = init.atPose2(i + 1)
        graph.add(gtsam.BetweenFactorPose2(i, i + 1, xi.between(xj), odom))
    for i in range(len(frames)):
        graph.add(gtsam.PriorFactorPose2(i, init.atPose2(i), frame_prior))
    for i, lm_key, bearing, rng_ in colmap_obs:
        graph.add(
            gtsam.BearingRangeFactor2D(
                i, lm_key, gtsam.Rot2(bearing), rng_, track_noise
            )
        )
    for i, lm_key, bearing, rng_ in virtual_obs:
        graph.add(
            gtsam.BearingRangeFactor2D(
                i, lm_key, gtsam.Rot2(bearing), rng_, vt_noise
            )
        )
    for i, f in enumerate(frames):
        if sdf_mask is None or sdf_mask[i]:
            graph.add(sdf_factor(i, f, sample_fn))

    params = gtsam.LevenbergMarquardtParams()
    result = gtsam.LevenbergMarquardtOptimizer(graph, init, params).optimize()
    return init, result


def values_to_poses(values, n: int) -> list[tuple[float, float, float]]:
    """gtsam.Values -> [(x, y, yaw)] for the first ``n`` Pose2 keys."""
    out = []
    for i in range(n):
        p = values.atPose2(i)
        out.append((float(p.x()), float(p.y()), float(p.theta())))
    return out


def per_frame_shifts(init, result, n_frames: int) -> np.ndarray:
    """Per-frame center shift (meters) between init and solved poses."""
    shifts = np.empty(n_frames)
    for i in range(n_frames):
        c = result.atPose2(i).compose(init.atPose2(i).inverse())
        shifts[i] = np.hypot(c.x(), c.y())
    return shifts


def correction_transform(correction) -> np.ndarray:
    """World-frame SE3 (4x4) of an SE2 correction (metric ENU frame)."""
    theta = correction.theta()
    c, s = np.cos(theta), np.sin(theta)
    g = np.eye(4)
    g[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    g[:3, 3] = [correction.x(), correction.y(), 0.0]
    return g


def write_corrected_poses(
    registered_dir: Path,
    frames: list[FrameWalls],
    init,
    result,
    out_path: Path,
) -> None:
    """poses.npz with the per-frame world SE3 correction applied.

    Mirrors the production ``write_correction``: each solved frame's
    c2w is left-multiplied by ``result_i o init_i^-1``; frames without
    wall geometry (not in the solve) keep their input pose.
    """
    poses = np.load(registered_dir / "poses.npz")
    names = [str(n) for n in poses["frame_names"]]
    row_of_name = {name: i for i, name in enumerate(names)}
    c2w = np.array(poses["c2w"], dtype=np.float64)
    for i, f in enumerate(frames):
        row = row_of_name.get(f.name)
        if row is None:
            continue
        correction = result.atPose2(i).compose(init.atPose2(i).inverse())
        c2w[row] = correction_transform(correction) @ c2w[row]
    np.savez(out_path, c2w=c2w, frame_names=np.array(names))
    logger.info("Wrote corrected poses to %s", out_path)


# ==========================================================================
#  Structure (M3: port of floorplan_tracks.build_structure, metric ENU)
# ==========================================================================


@dataclass(slots=True)
class StructureStats:
    """Summary of the structure built for a BA solve (logging/metrics)."""

    cap: int
    num_colmap_tracks: int
    num_colmap_obs: int
    num_virtual_tracks: int
    num_virtual_obs: int
    min_frame_cov: int
    starved_frames: int  # frames with no eligible SIFT track


def _bearing_range(
    xy: np.ndarray, pose: tuple[float, float, float]
) -> tuple[float, float]:
    """Bearing (rad) + range (m) from an SE2 pose to a 2D landmark."""
    x0, y0, yaw0 = pose
    ca, sa = np.cos(-yaw0), np.sin(-yaw0)
    dx, dy = xy[0] - x0, xy[1] - y0
    bx = ca * dx - sa * dy
    by = sa * dx + ca * dy
    return float(np.arctan2(by, bx)), float(np.hypot(bx, by))


def _candidates_to_observations(
    candidates: list[tuple[np.ndarray, list[int]]],
    selected: list[int],
    poses: list[tuple[float, float, float]],
    base_key: int,
) -> tuple[dict[int, np.ndarray], list[tuple[int, int, float, float]]]:
    """Selected ``(xy, frames)`` candidates -> landmarks + observations."""
    landmarks: dict[int, np.ndarray] = {}
    observations: list[tuple[int, int, float, float]] = []
    for offset, cand_idx in enumerate(selected):
        xy, frame_indices = candidates[cand_idx]
        key = base_key + offset
        landmarks[key] = xy
        for fidx in frame_indices:
            bearing, rng_ = _bearing_range(xy, poses[fidx])
            observations.append((fidx, key, bearing, rng_))
    return landmarks, observations


def _select_coverage_first(
    candidates: list[tuple[np.ndarray, list[int]]],
    order: list[int],
    coverage: np.ndarray,
    target: int,
    budget: int,
) -> list[int]:
    """Greedily pick candidates (in ``order``) raising an under-covered
    frame, updating ``coverage`` in place. Stops at ``budget`` picks."""
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= budget:
            break
        frame_indices = candidates[i][1]
        if any(coverage[f] < target for f in frame_indices):
            chosen.append(i)
            for f in frame_indices:
                coverage[f] += 1
    return chosen


def _load_depth_cache(
    frames: list[FrameWalls],
    info_by_name: dict[str, dict],
    dmap_dir: Path,
    indices: set[int],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """``(dmap, R, t, c)`` per frame-list index, loaded once."""
    cache: dict[int, tuple] = {}
    for i in indices:
        info = info_by_name[frames[i].name]
        dmap_path = dmap_dir / f"{frames[i].name}.npy"
        if not dmap_path.exists():
            continue
        cache[i] = (
            np.load(dmap_path).astype(np.float64),
            np.asarray(info["R"], dtype=np.float64),
            np.asarray(info["t"], dtype=np.float64),
            np.asarray(info["c"], dtype=np.float64),
        )
    return cache


def _anchor_points(
    dmap_a: np.ndarray,
    rot: np.ndarray,
    t: np.ndarray,
    center: np.ndarray,
    zband: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Up to ``VT_ANCHOR_PTS`` dense world points for VT reprojection,
    gated on height band and horizontal offset from the camera."""
    from ml_utils.depth_data import DepthProjection
    from ml_utils.utils import depth_map_to_world_points

    info = {"R": rot, "t": t}
    pts = depth_map_to_world_points(
        info, dmap_a, DepthProjection.EQUIRECTANGULAR
    )
    flat_pts = pts.reshape(-1, 3)
    depth = dmap_a.ravel()
    horiz = np.linalg.norm(flat_pts[:, :2] - center[:2], axis=1)
    ok = (
        np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(flat_pts).all(1)
        & (flat_pts[:, 2] > zband[0])
        & (flat_pts[:, 2] < zband[1])
        & (horiz > VT_MIN_HORIZ)
    )
    idx = np.where(ok)[0]
    if len(idx) > VT_ANCHOR_PTS:
        idx = idx[rng.choice(len(idx), VT_ANCHOR_PTS, replace=False)]
    return flat_pts[idx]


def _reprojection_matrix(
    anchor_pts: np.ndarray,
    members: list[int],
    anchor: int,
    cache: dict[int, tuple],
) -> np.ndarray:
    """Boolean ``(K, len(members))`` mask of depth-consistent reprojections."""
    from ml_utils.colmap_interface import spherical_img_from_cam

    k = len(anchor_pts)
    obs = np.zeros((k, len(members)), dtype=bool)
    for col, member in enumerate(members):
        if member == anchor:
            obs[:, col] = True
            continue
        dmap_j, rot_j, t_j, _c = cache[member]
        cam = anchor_pts @ rot_j.T + t_j
        radial = np.linalg.norm(cam, axis=1)
        h, w = dmap_j.shape
        uv = spherical_img_from_cam((w, h), cam)
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (radial > 1e-6)
        sampled = np.full(k, np.nan)
        sampled[valid] = dmap_j[v[valid], u[valid]]
        obs[:, col] = (
            valid
            & np.isfinite(sampled)
            & (sampled > 0)
            & (np.abs(radial - sampled) < VT_DEPTH_TOL)
        )
    return obs


def build_virtual_track_candidates(
    frames: list[FrameWalls],
    info_by_name: dict[str, dict],
    dmap_dir: Path,
    covisibility: dict[int, list[int]],
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, list[int]]]:
    """Dense cross-frame virtual-track candidates ``(xy, frame indices)``.

    Each anchor frame reprojects a dense sample of its depth into its
    co-visible neighbours; points seen depth-consistently by at least
    ``VT_MIN_OBS`` frames become candidates.
    """
    centers = np.stack([np.asarray(info_by_name[f.name]["c"]) for f in frames])
    zband = _z_band(centers)
    indices = set(range(len(frames)))
    cache = _load_depth_cache(frames, info_by_name, dmap_dir, indices)

    candidates: list[tuple[np.ndarray, list[int]]] = []
    for anchor in sorted(cache):
        members = [anchor] + [
            m for m in covisibility.get(anchor, []) if m in cache
        ]
        if len(members) < 2:
            continue
        dmap_a, rot, t, center = cache[anchor]
        anchor_pts = _anchor_points(dmap_a, rot, t, center, zband, rng)
        if len(anchor_pts) == 0:
            continue
        obs_matrix = _reprojection_matrix(anchor_pts, members, anchor, cache)
        rows = np.where(obs_matrix.sum(axis=1) >= VT_MIN_OBS)[0]
        for row in rows[:VT_MAX_PER_ANCHOR]:
            observing = [members[c] for c in np.where(obs_matrix[row])[0]]
            candidates.append((anchor_pts[row, :2], observing))
    return candidates


def build_structure(
    frames: list[FrameWalls],
    info_by_name: dict[str, dict],
    dmap_dir: Path,
    track_pool: list[tuple[np.ndarray, list[int]]],
    covisibility: dict[int, list[int]],
    rng: np.random.Generator,
) -> tuple[
    dict[int, np.ndarray],
    list[tuple[int, int, float, float]],
    list[tuple[int, int, float, float]],
    StructureStats,
]:
    """Coverage-first SIFT tracks + virtual-track gap fill.

    Port of the production ``build_structure``. Returns ``(landmarks,
    colmap_obs, virtual_obs, stats)``; the observation lists stay
    separate so the solver keeps clean SIFT tracks quadratic while
    robustifying the depth-derived virtual ones.
    """
    n_frames = len(frames)
    poses = [(f.x, f.y, f.yaw) for f in frames]
    cap = int(
        np.clip(TRACKS_PER_FRAME * n_frames, TRACK_CAP_MIN, TRACK_CAP_MAX)
    )

    order = sorted(range(len(track_pool)), key=lambda i: -len(track_pool[i][1]))
    coverage = np.zeros(n_frames, dtype=int)
    chosen = _select_coverage_first(
        track_pool, order, coverage, COVERAGE_TARGET, cap
    )
    chosen_set = set(chosen)
    if len(chosen) < cap:
        chosen.extend(i for i in order if i not in chosen_set)
        chosen = chosen[:cap]

    covered_by_pool: set[int] = set()
    for _xy, frame_indices in track_pool:
        covered_by_pool.update(frame_indices)
    starved = set(range(n_frames)) - covered_by_pool

    virtual_landmarks: dict[int, np.ndarray] = {}
    virtual_obs: list[tuple[int, int, float, float]] = []
    vt_coverage = np.zeros(n_frames, dtype=int)
    vt_candidates = build_virtual_track_candidates(
        frames, info_by_name, dmap_dir, covisibility, rng
    )
    if vt_candidates:
        vt_order = sorted(
            range(len(vt_candidates)), key=lambda i: -len(vt_candidates[i][1])
        )
        vt_chosen = _select_coverage_first(
            vt_candidates, vt_order, vt_coverage, COVERAGE_TARGET, cap
        )
        virtual_landmarks, virtual_obs = _candidates_to_observations(
            vt_candidates, vt_chosen, poses, base_key=n_frames + len(track_pool)
        )

    colmap_landmarks, colmap_obs = _candidates_to_observations(
        track_pool, chosen, poses, base_key=n_frames
    )
    stats = StructureStats(
        cap=cap,
        num_colmap_tracks=len(colmap_landmarks),
        num_colmap_obs=len(colmap_obs),
        num_virtual_tracks=len(virtual_landmarks),
        num_virtual_obs=len(virtual_obs),
        min_frame_cov=int((coverage + vt_coverage).min()) if n_frames else 0,
        starved_frames=len(starved),
    )
    landmarks = {**colmap_landmarks, **virtual_landmarks}
    return landmarks, colmap_obs, virtual_obs, stats


# ==========================================================================
#  COLMAP-model source (M3: track pool, covisibility, write-back)
# ==========================================================================


def _pano_stem(image_name: str) -> str:
    """``pano_camera3/keyframe_00042.jpg`` -> ``keyframe_00042``."""
    return Path(image_name.rpartition("/")[2]).stem


def load_frame_infos_from_model(model_dir: Path) -> list[dict]:
    """Rig-frame poses of the refined model as frame-info dicts.

    The rig frame IS the pano equirect frame (face cameras hang off it
    via sensor_from_rig), so ``rig_from_world`` gives the R/t that
    ``depth_map_to_world_points`` expects. Sorted by pano name; the
    pycolmap frame id rides along for the write-back.
    """
    import pycolmap

    recon = pycolmap.Reconstruction(str(model_dir))
    stem_of_frame: dict[int, str] = {}
    for image in recon.images.values():
        frame_id = image.frame_id
        if frame_id not in stem_of_frame:
            stem_of_frame[frame_id] = _pano_stem(image.name)
    infos = []
    for frame_id, stem in sorted(stem_of_frame.items(), key=lambda kv: kv[1]):
        frame = recon.frames[frame_id]
        rot = frame.rig_from_world.rotation.matrix()
        t = np.asarray(frame.rig_from_world.translation)
        infos.append(
            {
                "R": rot,
                "t": t,
                "c": -rot.T @ t,
                "keyframe_name": f"{stem}.jpg",
                "frame_id": len(infos),
                "pycolmap_frame_id": frame_id,
            }
        )
    return infos


def model_track_pool(
    model_dir: Path, frames: list[FrameWalls]
) -> list[tuple[np.ndarray, list[int]]]:
    """Triangulated tracks as ``(xy, observing frame-list indices)``."""
    import pycolmap

    recon = pycolmap.Reconstruction(str(model_dir))
    index_of_name = {f.name: i for i, f in enumerate(frames)}
    stem_of_image = {
        image.image_id: _pano_stem(image.name)
        for image in recon.images.values()
    }
    pool: list[tuple[np.ndarray, list[int]]] = []
    for point in recon.points3D.values():
        frame_indices = {
            index_of_name[stem]
            for el in point.track.elements
            if (stem := stem_of_image.get(el.image_id)) in index_of_name
        }
        if len(frame_indices) >= TRACK_MIN_OBS:
            pool.append((np.asarray(point.xyz)[:2], sorted(frame_indices)))
    return pool


def model_covisibility(
    model_dir: Path,
    frames: list[FrameWalls],
    min_points: int = COVIS_MIN_SHARED_POINTS,
) -> dict[int, list[int]]:
    """Frame-list covisibility from shared triangulated points."""
    from collections import defaultdict
    from itertools import combinations

    import pycolmap

    recon = pycolmap.Reconstruction(str(model_dir))
    index_of_name = {f.name: i for i, f in enumerate(frames)}
    stem_of_image = {
        image.image_id: _pano_stem(image.name)
        for image in recon.images.values()
    }
    shared: dict[tuple[int, int], int] = defaultdict(int)
    for point in recon.points3D.values():
        frame_indices = {
            index_of_name[stem]
            for el in point.track.elements
            if (stem := stem_of_image.get(el.image_id)) in index_of_name
        }
        for a, b in combinations(sorted(frame_indices), 2):
            shared[(a, b)] += 1
    covis: dict[int, set[int]] = {i: set() for i in range(len(frames))}
    for (a, b), count in shared.items():
        if count >= min_points:
            covis[a].add(b)
            covis[b].add(a)
    return {i: sorted(nbrs) for i, nbrs in covis.items()}


def neighbors_covisibility(
    registered_dir: Path, frames: list[FrameWalls]
) -> dict[int, list[int]]:
    """Frame-list covisibility from the registered neighbors.json."""
    payload = json.loads((registered_dir / "neighbors.json").read_text())
    neighbors = payload.get("neighbors", {})
    index_of_name = {f.name: i for i, f in enumerate(frames)}
    covis: dict[int, list[int]] = {}
    for i, f in enumerate(frames):
        entry = neighbors.get(f.name, {})
        covis[i] = sorted(
            index_of_name[n]
            for n in entry.get("neighbors", [])
            if n in index_of_name
        )
    return covis


def apply_correction_to_model(
    model_dir: Path,
    frames: list[FrameWalls],
    frame_infos: list[dict],
    init,
    result,
) -> Path:
    """Apply the SE2 correction to the model's rig frames, in place.

    Mirrors production ``ColmapSource.write_correction``: each solved
    frame's ``rig_from_world`` becomes ``old @ g^-1``; 3D points are
    left untouched (downstream consumes poses, not the sparse points).
    The untouched model is backed up beside it once.
    """
    import shutil

    import pycolmap

    backup = model_dir.parent / f"{model_dir.name}_prefloorplan"
    if not backup.exists():
        shutil.copytree(model_dir, backup)
        logger.info("Backed up %s -> %s", model_dir, backup)

    pycolmap_id_of = {
        info["frame_id"]: info["pycolmap_frame_id"] for info in frame_infos
    }
    recon = pycolmap.Reconstruction(str(model_dir))
    for i, f in enumerate(frames):
        correction = result.atPose2(i).compose(init.atPose2(i).inverse())
        g = correction_transform(correction)
        frame = recon.frames[pycolmap_id_of[f.frame_id]]
        told = np.eye(4)
        told[:3, :3] = frame.rig_from_world.rotation.matrix()
        told[:3, 3] = np.asarray(frame.rig_from_world.translation)
        tnew = told @ np.linalg.inv(g)
        frame.rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(tnew[:3, :3]), tnew[:3, 3]
        )
    recon.write(str(model_dir))
    logger.info(
        "Applied SE2 correction to %d/%d model frames in %s",
        len(frames),
        len(recon.frames),
        model_dir,
    )
    return model_dir


# ==========================================================================
#  Alignment diagnosis (M1: current poses, no solve)
# ==========================================================================


def walls_world(
    frames: list[FrameWalls], poses: list[tuple[float, float, float]]
) -> list[np.ndarray]:
    """Each frame's wall points in world xy under the given SE2 poses."""
    out = []
    for f, (x, y, yaw) in zip(frames, poses, strict=True):
        c, s = np.cos(yaw), np.sin(yaw)
        q0, q1 = f.wall_body[:, 0], f.wall_body[:, 1]
        out.append(np.column_stack([x + c * q0 - s * q1, y + s * q0 + c * q1]))
    return out


def frame_sdf_stats(
    frames: list[FrameWalls],
    poses: list[tuple[float, float, float]],
    sample_fn,
    trim: float | None = None,
) -> np.ndarray:
    """Mean |SDF| per frame (meters), optionally trimmed at ``trim``."""
    means = np.empty(len(frames))
    for i, pts in enumerate(walls_world(frames, poses)):
        vals = np.abs(sample_fn(pts[:, 0], pts[:, 1])[0])
        if trim is not None:
            vals = np.minimum(vals, trim)
        means[i] = float(np.mean(vals))
    return means


def save_debug_plots(
    sdf: _Sdf,
    frames: list[FrameWalls],
    poses: list[tuple[float, float, float]],
    per_frame: np.ndarray,
    out_dir: Path,
    poses_after: list[tuple[float, float, float]] | None = None,
) -> list[Path]:
    """Figures: walls (+solved walls) over plan edges; per-frame |SDF|.

    With ``poses_after`` the alignment figure shows before (red) and
    after (green) like the production stage, the trajectory and the
    per-frame coloring use the solved poses.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plan = sdf.plan
    edges = plan_edges(plan.gray)
    wall_pts = np.vstack(walls_world(frames, poses))
    wall_px = plan.enu_to_pixel(wall_pts[:, 0], wall_pts[:, 1])
    traj_poses = poses_after if poses_after is not None else poses
    traj = np.array([[p[0], p[1]] for p in traj_poses])
    traj_px = plan.enu_to_pixel(traj[:, 0], traj[:, 1])
    after_px = None
    if poses_after is not None:
        after_pts = np.vstack(walls_world(frames, poses_after))
        after_px = plan.enu_to_pixel(after_pts[:, 0], after_pts[:, 1])
    pad = 2.0 / plan.px
    cols = np.concatenate([wall_px[0], traj_px[0]])
    rows = np.concatenate([wall_px[1], traj_px[1]])
    xlim = (cols.min() - pad, cols.max() + pad)
    ylim = (rows.max() + pad, rows.min() - pad)

    written: list[Path] = []
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(~edges, cmap="gray", interpolation="nearest")
    label = "walls before" if after_px is not None else "depth walls"
    ax.scatter(*wall_px, s=0.3, c="red", alpha=0.35, label=label)
    if after_px is not None:
        ax.scatter(*after_px, s=0.3, c="green", alpha=0.35, label="walls after")
    ax.plot(*traj_px, color="tab:blue", linewidth=0.8, label="trajectory")
    ax.legend(loc="lower right", markerscale=20)
    ax.set_title("FloorPlanBA: depth walls vs extracted plan edges")
    ax.set_axis_off()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    path = out_dir / "report_floorplan_alignment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(~edges, cmap="gray", interpolation="nearest")
    sc = ax.scatter(*traj_px, s=14, c=per_frame, cmap="viridis")
    fig.colorbar(sc, ax=ax, fraction=0.04, label="frame mean |SDF| (m)")
    ax.set_title("FloorPlanBA: per-frame wall->plan distance")
    ax.set_axis_off()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    path = out_dir / "report_floorplan_frame_sdf.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


def run(
    workfolder: Path,
    plan_path: Path,
    anchors_path: Path,
    registered_dir: Path | None = None,
    out_dir: Path | None = None,
    solve: bool = False,
    min_frames: int = MIN_FRAMES,
    model_dir: Path | None = None,
    structure: bool = True,
    apply: bool = False,
    prealign: bool = True,
) -> dict:
    """Plan georeference + wall extraction + |SDF| report; optional solve.

    With ``prealign`` (default) the solve is preceded by a coarse
    global SE2 chamfer search (:func:`coarse_prealign`) that recovers
    initial rotations/offsets beyond the SDF reject gate, and frames
    whose walls do not correspond to drawn ink lose their SDF factor
    (out-of-sheet gate). The guard then judges the per-frame solve's
    residual movement; the write-back carries the total correction.

    Without ``solve`` this is the M1 diagnosis under the current poses.
    With ``solve`` the SE2 optimization runs and, when it passes the
    production skip-guard (|SDF| must improve, p90 frame shift under
    ``CORRECTION_P90_M``), ``corrected_poses.npz`` is written to
    ``out_dir``.

    ``model_dir`` (the refined COLMAP model, e.g. ``sfm/sparse_enu``)
    switches the pose source to the model's rig frames and enables the
    SIFT track pool + point-covisibility structure; without it, poses
    come from ``registered_dir/poses.npz`` and structure falls back to
    virtual tracks over the neighbors.json graph. With ``apply`` the
    passing correction is written back — to the model's rig frames
    (model source) and to ``registered_dir/poses.npz`` (backed up
    first) so downstream stages inherit it.
    """
    if registered_dir is None:
        for candidate in ("registered", "initial_registration"):
            if (workfolder / candidate / "poses.npz").exists():
                registered_dir = workfolder / candidate
                break
        else:
            raise FileNotFoundError(f"No poses.npz under {workfolder}")
    out_dir = out_dir or workfolder / "floorplan_ba"
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_lla = None
    ref_path = workfolder / "reference_lla.json"
    if ref_path.exists():
        reference_lla = json.loads(ref_path.read_text())
    anchors = json.loads(anchors_path.read_text())
    plan = load_plan(plan_path, anchors, reference_lla)
    sdf = build_wall_sdf(plan)
    sample_fn = sdf_sampler(sdf)

    use_model = model_dir is not None and (model_dir / "frames.bin").exists()
    if model_dir is not None and not use_model:
        logger.warning("Model %s not found; using poses.npz", model_dir)
    if use_model:
        frame_infos = load_frame_infos_from_model(model_dir)
    else:
        frame_infos = load_frame_infos(registered_dir)
    frames = extract_frame_walls(
        frame_infos, registered_dir / "dmaps", np.random.default_rng(0)
    )
    logger.info(
        "%d/%d frames with usable wall geometry (source %s)",
        len(frames),
        len(frame_infos),
        registered_dir,
    )
    if not frames:
        raise RuntimeError("No frames with wall geometry; cannot diagnose")

    poses = [(f.x, f.y, f.yaw) for f in frames]
    per_frame = frame_sdf_stats(frames, poses, sample_fn)
    report = {
        "registered_dir": str(registered_dir),
        "num_frames": len(frames),
        "num_frames_total": len(frame_infos),
        "mean_abs_sdf_m": float(per_frame.mean()),
        "median_abs_sdf_m": float(np.median(per_frame)),
        "p90_abs_sdf_m": float(np.percentile(per_frame, 90)),
        "per_frame": {
            f.name: float(v) for f, v in zip(frames, per_frame, strict=True)
        },
    }

    poses_after = None
    if solve and len(frames) < min_frames:
        logger.info(
            "Solve skipped: only %d usable frames (<%d)",
            len(frames),
            min_frames,
        )
        report["solve"] = {"skipped": f"frames < {min_frames}"}
    elif solve:
        frames_solve = frames
        if prealign:
            theta, t_pre, pre_info = coarse_prealign(
                frames, sample_fn, np.random.default_rng(2)
            )
            report["prealign"] = pre_info
            if pre_info["applied"]:
                frames_solve = apply_se2_to_frames(frames, theta, t_pre)
            logger.info(
                "pre-align: yaw %+.2f deg, shift (%+.2f, %+.2f) m, "
                "trimmed score %.3f -> %.3f (%s)",
                pre_info["yaw_deg"],
                pre_info["shift_m"][0],
                pre_info["shift_m"][1],
                pre_info["score_before"],
                pre_info["score_after"],
                "applied" if pre_info["applied"] else "not worth it",
            )

        # Out-of-sheet gate: frames whose walls do not plausibly
        # correspond to drawn ink lose their SDF factor (they keep
        # odometry, priors, and structure, so the chain stays rigid).
        # The statistic is the fraction of the frame's wall points
        # within SDF_REJECT of ink — a mean saturates too slowly
        # because panos see far and always catch some drawn walls.
        poses_pre = [(f.x, f.y, f.yaw) for f in frames_solve]
        near_frac = np.empty(len(frames_solve))
        for i, pts in enumerate(walls_world(frames_solve, poses_pre)):
            vals = np.abs(sample_fn(pts[:, 0], pts[:, 1])[0])
            near_frac[i] = float((vals <= SDF_REJECT).mean())
        med_frac = float(np.median(near_frac))
        if med_frac >= OFFSHEET_MIN_MEDIAN:
            gate = OFFSHEET_REL_FRACTION * med_frac
            sdf_mask = near_frac >= gate
        else:
            # The whole capture mismatches the plan (e.g. envelope-only
            # sheets); leave every SDF factor in and let the guard rule.
            gate = 0.0
            sdf_mask = np.ones(len(frames_solve), dtype=bool)
        report["sdf_gate"] = {
            "median_near_fraction": med_frac,
            "threshold_fraction": gate,
            "masked_frames": int((~sdf_mask).sum()),
        }
        if (~sdf_mask).any():
            logger.info(
                "out-of-sheet gate: %d/%d frames lose their SDF factor "
                "(near-ink fraction < %.2f; capture median %.2f)",
                int((~sdf_mask).sum()),
                len(frames_solve),
                gate,
                med_frac,
            )

        landmarks: dict[int, np.ndarray] = {}
        colmap_obs: list[tuple[int, int, float, float]] = []
        virtual_obs: list[tuple[int, int, float, float]] = []
        if structure:
            track_pool = (
                model_track_pool(model_dir, frames) if use_model else []
            )
            if use_model:
                covisibility = model_covisibility(model_dir, frames)
            elif (registered_dir / "neighbors.json").exists():
                covisibility = neighbors_covisibility(registered_dir, frames)
            else:
                covisibility = {i: [] for i in range(len(frames))}
            info_by_name = {
                Path(i["keyframe_name"]).stem: i for i in frame_infos
            }
            # Measurements (bearing/range) are body-frame and therefore
            # invariant to the global pre-alignment, so the structure is
            # built from the ORIGINAL geometry; only the landmark initial
            # values must follow the poses into the pre-aligned frame.
            landmarks, colmap_obs, virtual_obs, stats = build_structure(
                frames,
                info_by_name,
                registered_dir / "dmaps",
                track_pool,
                covisibility,
                np.random.default_rng(1),
            )
            if frames_solve is not frames:
                c, s = np.cos(theta), np.sin(theta)
                rot = np.array([[c, -s], [s, c]])
                landmarks = {k: rot @ xy + t_pre for k, xy in landmarks.items()}
            logger.info(
                "structure: cap %d, %d tracks (%d obs), %d virtual "
                "(%d obs), min coverage %d, %d starved frames",
                stats.cap,
                stats.num_colmap_tracks,
                stats.num_colmap_obs,
                stats.num_virtual_tracks,
                stats.num_virtual_obs,
                stats.min_frame_cov,
                stats.starved_frames,
            )
            report["structure"] = {
                "num_colmap_tracks": stats.num_colmap_tracks,
                "num_colmap_obs": stats.num_colmap_obs,
                "num_virtual_tracks": stats.num_virtual_tracks,
                "num_virtual_obs": stats.num_virtual_obs,
                "min_frame_cov": stats.min_frame_cov,
                "starved_frames": stats.starved_frames,
            }
        init_pre, result = optimize_poses(
            frames_solve,
            sample_fn,
            landmarks=landmarks,
            colmap_obs=colmap_obs,
            virtual_obs=virtual_obs,
            sdf_mask=sdf_mask,
        )
        poses_after = values_to_poses(result, len(frames_solve))
        after = frame_sdf_stats(frames_solve, poses_after, sample_fn)
        # The guard judges the SOLVE's own movement (vs the pre-aligned
        # init): the pre-alignment already proved itself on the trimmed
        # score. Total movement vs the original poses is reported too
        # and is what gets written back.
        import gtsam

        init_orig = gtsam.Values()
        for i, f in enumerate(frames):
            init_orig.insert(i, gtsam.Pose2(f.x, f.y, f.yaw))
        shifts = per_frame_shifts(init_pre, result, len(frames_solve))
        shifts_total = per_frame_shifts(init_orig, result, len(frames))
        p90_shift = float(np.percentile(shifts, 90))
        corrected = bool(
            after.mean() < per_frame.mean() and p90_shift <= CORRECTION_P90_M
        )
        report["solve"] = {
            "mean_abs_sdf_after_m": float(after.mean()),
            "median_abs_sdf_after_m": float(np.median(after)),
            "p90_abs_sdf_after_m": float(np.percentile(after, 90)),
            "median_frame_shift_m": float(np.median(shifts)),
            "p90_frame_shift_m": p90_shift,
            "max_frame_shift_m": float(shifts.max()),
            "median_total_shift_m": float(np.median(shifts_total)),
            "max_total_shift_m": float(shifts_total.max()),
            "corrected": corrected,
        }
        logger.info(
            "solve: wall->plan |dist| %.3fm -> %.3fm, residual shift "
            "median %.3fm p90 %.3fm max %.3fm, total median %.3fm (%s)",
            per_frame.mean(),
            after.mean(),
            float(np.median(shifts)),
            p90_shift,
            float(shifts.max()),
            float(np.median(shifts_total)),
            "applied" if corrected else "SKIPPED by guard",
        )
        if corrected:
            write_corrected_poses(
                registered_dir,
                frames,
                init_orig,
                result,
                out_dir / "corrected_poses.npz",
            )
            if apply and use_model:
                # The model is the pose source; correct/export propagate
                # the corrected rig frames onto dmaps and exports.
                apply_correction_to_model(
                    model_dir, frames, frame_infos, init_orig, result
                )
            elif apply:
                import shutil

                backup = registered_dir / "poses_prefloorplan.npz"
                if not backup.exists():
                    shutil.copy2(registered_dir / "poses.npz", backup)
                shutil.copy2(
                    out_dir / "corrected_poses.npz",
                    registered_dir / "poses.npz",
                )
                logger.info(
                    "Applied correction to %s (backup %s)",
                    registered_dir / "poses.npz",
                    backup.name,
                )
        per_frame = after  # color the trajectory by post-solve distances

    (out_dir / "floorplan_report.json").write_text(json.dumps(report, indent=2))
    save_debug_plots(
        sdf, frames, poses, per_frame, out_dir, poses_after=poses_after
    )
    logger.info(
        "wall->plan |dist|: mean %.3f m, median %.3f m, p90 %.3f m -> %s",
        report["mean_abs_sdf_m"],
        report["median_abs_sdf_m"],
        report["p90_abs_sdf_m"],
        out_dir,
    )
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workfolder", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan_anchors", type=Path, required=True)
    parser.add_argument(
        "--registered_dir",
        type=Path,
        default=None,
        help="poses+dmaps source (default: workfolder/registered, "
        "falling back to initial_registration)",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument(
        "--solve",
        action="store_true",
        help="run the SE2 optimization (default: diagnosis only)",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=MIN_FRAMES,
        help="minimum usable frames for the solve",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="refined COLMAP model (default: workfolder/sfm/sparse_enu "
        "when it exists); enables SIFT tracks + model write-back",
    )
    parser.add_argument(
        "--no_structure",
        action="store_true",
        help="solve with SDF + odometry + priors only (no track factors)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the passing correction back to the model / poses.npz",
    )
    parser.add_argument(
        "--no_prealign",
        action="store_true",
        help="skip the coarse global SE2 pre-alignment + out-of-sheet gate",
    )
    args = parser.parse_args()
    model_dir = args.model_dir
    if model_dir is None:
        default_model = args.workfolder / "sfm" / "sparse_enu"
        if (default_model / "frames.bin").exists():
            model_dir = default_model
    run(
        args.workfolder,
        args.plan,
        args.plan_anchors,
        registered_dir=args.registered_dir,
        out_dir=args.out_dir,
        solve=args.solve,
        min_frames=args.min_frames,
        model_dir=model_dir,
        structure=not args.no_structure,
        apply=args.apply,
        prealign=not args.no_prealign,
    )


if __name__ == "__main__":
    main()
