#!/usr/bin/env python3
"""Refine stage of the PanoVGGT equirectangular pipeline.

Takes the posed rig model + SIFT database produced by pano_sift_faces.py
and refines the calibration in place of GLUEMAP's global refinement:

  1. Triangulate SIFT matches with poses fixed (COLMAP point triangulator).
  2. Optionally merge "virtual" tracks grown from the registered PanoVGGT
     depth maps (``ml_utils.batch_registration.build_virtual_tracks``),
     but only for pano pairs whose SIFT support is weak — SIFT dominates
     where it works, depth tracks backfill weak-texture regions
     (GLUEMAP's track-selection idea, at pano-pair granularity).
  3. Rig-constrained BA rounds with reprojection filtering, rig-PnP
     re-registration of frames the filters stripped, and observation
     re-linking (reusing rig_refine.py's machinery).
  4. GPS-position-prior BA rounds. The model is already in metric ENU
     (batch_registration aligned it), so the ENU targets from the
     PanoVGGT manifest are used directly — no sim(3) refit.

Outputs (under --out_dir):
    * ``sparse_enu/``      — the final rig-constrained COLMAP model (ENU).
    * ``pano_poses.json``  — per-pano poses.
    * ``pano_centers.ply`` — trajectory for a quick visual check.

Example:
    python scripts/pano_rig_refine.py \
        --sfm_dir /data/site/sfm \
        --images_cubemap /data/site/images_cubemap \
        --registered_dir /data/site/initial_registration \
        --manifest /data/site/ml_batches/manifest.json \
        --out_dir /data/site/sfm
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pycolmap

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pano_sift_faces import FACE_PREFIX, load_registered_poses  # noqa: E402
from pano_star_solve import _project_spherical  # noqa: E402
from rig_refine import (  # noqa: E402
    frame_center,
    run_ba,
    run_gps_ba,
    write_ply,
)
from scipy.spatial.transform import Rotation  # noqa: E402

logger = logging.getLogger(__name__)


def pano_stem_of_image(name: str) -> str:
    """``pano_camera{k}/<file>`` -> pano stem."""
    _, _, filename = name.partition("/")
    return Path(filename).stem


def single_rig(recon: pycolmap.Reconstruction) -> pycolmap.Rig:
    """The reconstruction's one rig (its id is database-assigned, not 1)."""
    rigs = list(recon.rigs.values())
    if len(rigs) != 1:
        raise ValueError(f"Expected exactly 1 rig, got {len(rigs)}")
    return rigs[0]


def triangulate_sift(
    sparse_init: Path,
    database_path: Path,
    images_cubemap: Path,
    out_dir: Path,
) -> pycolmap.Reconstruction:
    """Triangulate SIFT matches into the posed rig model (poses fixed)."""
    recon = pycolmap.Reconstruction(str(sparse_init))
    options = pycolmap.IncrementalPipelineOptions()
    options.triangulation.ignore_two_view_tracks = False
    options.triangulation.min_angle = 1.0
    options.ba_global_max_refinements = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    recon = pycolmap.triangulate_points(
        recon,
        str(database_path),
        str(images_cubemap),
        str(out_dir),
        clear_points=True,
        options=options,
    )
    logger.info("Triangulated %d SIFT points", recon.num_points3D())
    return recon


def compute_pano_pair_support(
    recon: pycolmap.Reconstruction,
) -> dict[frozenset, int]:
    """SIFT track support per unordered pano pair."""
    support: dict[frozenset, int] = defaultdict(int)
    for point in recon.points3D.values():
        panos = {
            pano_stem_of_image(recon.images[el.image_id].name)
            for el in point.track.elements
        }
        for pair in combinations(sorted(panos), 2):
            support[frozenset(pair)] += 1
    return dict(support)


def build_and_select_virtual_tracks(
    registered_dir: Path,
    recon: pycolmap.Reconstruction,
    min_pair_support: int,
    image_folder: Path | None = None,
) -> list[dict]:
    """Virtual tracks from PanoVGGT depths, kept only where SIFT is weak.

    A track survives selection if at least one pano pair among its
    observations has fewer than ``min_pair_support`` SIFT tracks.
    Observations are remapped from equirect pixels onto the cube face
    that observes the track's canonical 3D position, in the face
    resolution of the reconstruction's cameras.
    """
    from ml_utils.batch_registration import (
        BatchRegistrationConf,
        build_virtual_tracks,
    )
    from ml_utils.colmap_export import _remap_tracks_to_faces
    from ml_utils.pano import get_cubemap_rotations

    conf = BatchRegistrationConf()
    tracks = build_virtual_tracks(
        registered_dir, conf, image_folder=image_folder
    )
    if not tracks:
        return []

    support = compute_pano_pair_support(recon)
    selected = []
    for track in tracks:
        panos = sorted(
            {Path(name).stem for name, _u, _v in track["observations"]}
        )
        weak = any(
            support.get(frozenset(pair), 0) < min_pair_support
            for pair in combinations(panos, 2)
        )
        if weak:
            selected.append(track)
    logger.info(
        "Virtual tracks: %d built, %d selected "
        "(pano pairs with SIFT support < %d)",
        len(tracks),
        len(selected),
        min_pair_support,
    )
    if not selected:
        return []

    names, c2w = load_registered_poses(registered_dir)
    camera = recon.cameras[min(recon.cameras)]
    face_k = np.asarray(camera.calibration_matrix())
    return _remap_tracks_to_faces(
        selected,
        names,
        c2w,
        [np.asarray(r) for r in get_cubemap_rotations()],
        face_k,
        camera.width,
        camera.height,
        min_track_length=2,
    )


def merge_virtual_tracks(
    recon: pycolmap.Reconstruction,
    remapped_tracks: list[dict],
) -> tuple[pycolmap.Reconstruction, set[int]]:
    """Rebuild the reconstruction with virtual observations and points added.

    COLMAP images accept their Point2D list only once, so a fresh
    reconstruction is built: existing images get their SIFT Point2D list
    extended by the virtual observations (model-only points — the COLMAP
    format does not require them in the database), existing 3D points are
    renumbered 1..N and virtual tracks appended as N+1..N+M. BA and the
    reprojection filters then treat SIFT and virtual points uniformly.

    Returns ``(new_reconstruction, virtual_point3D_ids)``.
    """
    image_by_face_name: dict[tuple[str, int], int] = {}
    for image in recon.images.values():
        prefix, _, filename = image.name.partition("/")
        if prefix.startswith(FACE_PREFIX):
            face_idx = int(prefix[len(FACE_PREFIX) :])
            image_by_face_name[(Path(filename).stem, face_idx)] = image.image_id

    # Stage the new Point2D entries so their indices are known up front.
    new_xy: dict[int, list[np.ndarray]] = defaultdict(list)
    staged: list[tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] = []
    for track in remapped_tracks:
        elements: list[tuple[int, int]] = []
        for name, face_idx, u, v in track["observations"]:
            image_id = image_by_face_name.get((Path(name).stem, face_idx))
            if image_id is None:
                continue
            p2d_idx = len(recon.images[image_id].points2D) + len(
                new_xy[image_id]
            )
            new_xy[image_id].append(np.array([u, v], dtype=np.float64))
            elements.append((image_id, p2d_idx))
        if len(elements) >= 2:
            staged.append(
                (
                    np.asarray(track["xyz"], dtype=np.float64),
                    np.asarray(track["color"]),
                    elements,
                )
            )

    # Renumber: SIFT points first (add_point3D assigns 1, 2, ... in call
    # order and validates that referenced 2D points already carry the id).
    sift_pids = sorted(recon.points3D)
    new_pid_of = {pid: i + 1 for i, pid in enumerate(sift_pids)}
    first_virtual = len(sift_pids) + 1
    virtual_pid_of_p2d: dict[tuple[int, int], int] = {}
    for offset, (_xyz, _color, elements) in enumerate(staged):
        for image_id, p2d_idx in elements:
            virtual_pid_of_p2d[(image_id, p2d_idx)] = first_virtual + offset

    merged = pycolmap.Reconstruction()
    for cam in recon.cameras.values():
        merged.add_camera(cam)
    merged.add_rig(single_rig(recon))
    for frame_id, frame in recon.frames.items():
        merged.add_frame(pycolmap.Frame(frame_id=frame_id, rig_id=frame.rig_id))
        for image in recon.images.values():
            if image.frame_id != frame_id:
                continue
            points2d = [
                pycolmap.Point2D(
                    p.xy,
                    point3D_id=new_pid_of.get(
                        p.point3D_id, pycolmap.INVALID_POINT3D_ID
                    ),
                )
                for p in image.points2D
            ]
            base = len(points2d)
            for k, xy in enumerate(new_xy.get(image.image_id, [])):
                points2d.append(
                    pycolmap.Point2D(
                        xy,
                        point3D_id=virtual_pid_of_p2d.get(
                            (image.image_id, base + k),
                            pycolmap.INVALID_POINT3D_ID,
                        ),
                    )
                )
            new_image = pycolmap.Image(
                image_id=image.image_id,
                camera_id=image.camera_id,
                name=image.name,
                frame_id=frame_id,
            )
            new_image.points2D = pycolmap.Point2DList(points2d)
            merged.frame(frame_id).add_data_id(new_image.data_id)
            merged.add_image(new_image)
        merged.frame(frame_id).rig_from_world = frame.rig_from_world
        merged.register_frame(frame_id)

    for pid in sift_pids:
        point = recon.points3D[pid]
        track = pycolmap.Track()
        for el in point.track.elements:
            track.add_element(el.image_id, el.point2D_idx)
        assigned = merged.add_point3D(point.xyz, track, point.color)
        assert assigned == new_pid_of[pid], f"{assigned} != {new_pid_of[pid]}"

    virtual_pids: set[int] = set()
    for offset, (xyz, color, elements) in enumerate(staged):
        track = pycolmap.Track()
        for image_id, p2d_idx in elements:
            track.add_element(image_id, p2d_idx)
        assigned = merged.add_point3D(
            xyz, track, np.asarray(color, dtype=np.uint8)
        )
        assert assigned == first_virtual + offset, (
            f"{assigned} != {first_virtual + offset}"
        )
        virtual_pids.add(assigned)

    logger.info(
        "Merged %d virtual tracks (%d observations) into the reconstruction",
        len(virtual_pids),
        sum(len(e) for _, _, e in staged),
    )
    return merged, virtual_pids


def build_obs_lookup(
    recon: pycolmap.Reconstruction,
) -> list[tuple[int, int, int]]:
    """Every linked observation as (image_id, point2D_idx, point3D_id)."""
    obs_lookup = []
    for image in recon.images.values():
        for idx, p2d in enumerate(image.points2D):
            if p2d.point3D_id != pycolmap.INVALID_POINT3D_ID:
                obs_lookup.append((image.image_id, idx, p2d.point3D_id))
    return obs_lookup


class ReprojEngine:
    """Vectorized reprojection errors over a fixed observation set.

    The per-round bookkeeping between BA solves (filter / relink /
    mean-reproj) is pure-Python-per-observation in rig_refine.py, which
    is fine at ~100k observations but becomes the single-core bottleneck
    at millions. This engine gathers the static data (observation pixel
    coordinates, per-observation image and point ids) once and evaluates
    all reprojection errors as numpy array ops; only the observations
    that actually change state go through pycolmap calls. The linked
    state is tracked internally, so the reconstruction is never
    re-scanned.
    """

    def __init__(
        self, recon: pycolmap.Reconstruction, obs_lookup: list
    ) -> None:
        obs = np.asarray(obs_lookup, dtype=np.int64)
        self.obs_iid = obs[:, 0]
        self.obs_idx = obs[:, 1]
        self.obs_pid = obs[:, 2]
        self.linked = np.ones(len(obs), dtype=bool)

        self.image_ids = np.array(sorted(recon.images.keys()))
        slot_of = {int(iid): k for k, iid in enumerate(self.image_ids)}
        self.obs_img = np.array(
            [slot_of[int(i)] for i in self.obs_iid], dtype=np.int64
        )
        self.frame_of_image = np.array(
            [recon.images[int(iid)].frame_id for iid in self.image_ids]
        )

        # Pixel coordinates never change; gather them in one pass.
        self.obs_xy = np.empty((len(obs), 2))
        by_image: dict[int, list[int]] = defaultdict(list)
        for row, iid in enumerate(self.obs_iid):
            by_image[int(iid)].append(row)
        for iid, rows in by_image.items():
            points2d = recon.images[iid].points2D
            self.obs_xy[rows] = [
                points2d[int(self.obs_idx[r])].xy for r in rows
            ]

        # SIMPLE_PINHOLE intrinsics per image (shared analytic K).
        params = np.stack(
            [
                recon.cameras[recon.images[int(iid)].camera_id].params
                for iid in self.image_ids
            ]
        )
        self.focal = params[:, 0]
        self.pp = params[:, 1:3]
        cam0 = recon.cameras[min(recon.cameras)]
        self.behind_penalty = 2.0 * float(np.hypot(cam0.width, cam0.height))

    def errors(self, recon: pycolmap.Reconstruction) -> np.ndarray:
        """Reprojection error per observation.

        Behind-camera observations get the 2x-diagonal penalty (matching
        rig_refine); observations whose point no longer exists get NaN.
        """
        mats = np.stack(
            [
                recon.images[int(iid)].cam_from_world().matrix()
                for iid in self.image_ids
            ]
        )  # (M, 3, 4)
        alive_pids = np.fromiter(recon.points3D.keys(), dtype=np.int64)
        order = np.argsort(alive_pids)
        alive_sorted = alive_pids[order]
        xyz_sorted = np.stack(
            [recon.points3D[int(p)].xyz for p in alive_sorted]
        )
        pos = np.clip(
            np.searchsorted(alive_sorted, self.obs_pid),
            0,
            len(alive_sorted) - 1,
        )
        alive = alive_sorted[pos] == self.obs_pid
        xyz = xyz_sorted[pos]

        m = mats[self.obs_img]
        x_cam = np.einsum("nij,nj->ni", m[:, :, :3], xyz) + m[:, :, 3]
        z = x_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = (
                self.focal[self.obs_img, None] * x_cam[:, :2] / z[:, None]
                + self.pp[self.obs_img]
            )
            err = np.linalg.norm(proj - self.obs_xy, axis=1)
        err = np.where(z <= 0, self.behind_penalty, err)
        err[~alive] = np.nan
        return err

    def mean_reproj_px(self, recon: pycolmap.Reconstruction) -> float:
        err = self.errors(recon)[self.linked]
        err = err[np.isfinite(err)]
        return float(err.mean()) if err.size else 0.0

    def filter_observations(
        self,
        recon: pycolmap.Reconstruction,
        max_error_px: float,
        min_track_length: int,
    ) -> tuple[int, int]:
        """Vectorized equivalent of rig_refine.filter_observations."""
        err = self.errors(recon)
        bad = self.linked & (np.isnan(err) | (err > max_error_px))
        keep = self.linked & ~bad
        # Surviving track length per point over the linked set.
        pids, counts = np.unique(self.obs_pid[keep], return_counts=True)
        short = set(pids[counts < min_track_length].tolist())
        all_pids = set(np.unique(self.obs_pid[self.linked]).tolist())
        short |= all_pids - set(pids.tolist())  # points losing every obs

        obs_removed = 0
        tracks_removed = 0
        short_arr = (
            np.fromiter(short, dtype=np.int64)
            if short
            else np.empty(0, dtype=np.int64)
        )
        kill_rows = np.nonzero(
            self.linked & (bad | np.isin(self.obs_pid, short_arr))
        )[0]
        dead_pids: set[int] = set()
        for row in kill_rows:
            pid = int(self.obs_pid[row])
            if pid in short:
                if pid not in dead_pids:
                    if pid in recon.points3D:
                        obs_removed += recon.points3D[pid].track.length()
                        recon.delete_point3D(pid)
                        tracks_removed += 1
                    dead_pids.add(pid)
                self.linked[row] = False
                continue
            recon.points3D[pid].track.delete_element(
                int(self.obs_iid[row]), int(self.obs_idx[row])
            )
            recon.images[int(self.obs_iid[row])].reset_point3D_for_point2D(
                int(self.obs_idx[row])
            )
            self.linked[row] = False
            obs_removed += 1
        return obs_removed, tracks_removed

    def relink_observations(
        self, recon: pycolmap.Reconstruction, max_error_px: float
    ) -> int:
        """Vectorized equivalent of rig_refine.relink_observations."""
        err = self.errors(recon)
        good = ~self.linked & np.isfinite(err) & (err <= max_error_px)
        relinked = 0
        for row in np.nonzero(good)[0]:
            pid = int(self.obs_pid[row])
            iid, idx = int(self.obs_iid[row]), int(self.obs_idx[row])
            image = recon.images[iid]
            if image.points2D[idx].point3D_id != pycolmap.INVALID_POINT3D_ID:
                continue
            recon.points3D[pid].track.add_element(iid, idx)
            image.set_point3D_for_point2D(idx, pid)
            self.linked[row] = True
            relinked += 1
        return relinked

    def linked_counts_by_frame(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        frames = self.frame_of_image[self.obs_img[self.linked]]
        for fid, cnt in zip(
            *np.unique(frames, return_counts=True), strict=True
        ):
            counts[int(fid)] = int(cnt)
        return counts


def reregister_frames_rig(
    recon: pycolmap.Reconstruction,
    obs_lookup: list[tuple[int, int, int]],
    max_error_px: float,
    min_frame_obs: int,
    min_inliers: int,
    linked_counts: dict[int, int] | None = None,
) -> int:
    """rig_refine.reregister_frames with the rig taken from the model itself."""
    identity = pycolmap.Rigid3d(
        rotation=np.array([0.0, 0.0, 0.0, 1.0]), translation=np.zeros(3)
    )
    rig = single_rig(recon)
    cam_ids = sorted(recon.cameras)
    cams_from_rig = []
    for cam_id in cam_ids:
        sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, cam_id)
        cams_from_rig.append(
            identity
            if rig.is_ref_sensor(sensor)
            else rig.sensor_from_rig(sensor)
        )
    cameras = [recon.cameras[cam_id] for cam_id in cam_ids]
    slot_of_camera = {cam_id: i for i, cam_id in enumerate(cam_ids)}

    linked = {fid: 0 for fid in recon.frames}
    if linked_counts is not None:
        linked.update(linked_counts)
    else:
        for image in recon.images.values():
            linked[image.frame_id] += sum(
                p.point3D_id != pycolmap.INVALID_POINT3D_ID
                for p in image.points2D
            )

    by_frame: dict[int, list[tuple[int, int, int]]] = {}
    for image_id, idx, pid in obs_lookup:
        if pid in recon.points3D:
            fid = recon.images[image_id].frame_id
            by_frame.setdefault(fid, []).append((image_id, idx, pid))

    ransac = pycolmap.RANSACOptions()
    ransac.max_error = max_error_px

    num_reregistered = 0
    for fid, n_linked in sorted(linked.items()):
        if n_linked >= min_frame_obs:
            continue
        cands = by_frame.get(fid, [])
        if len(cands) < min_inliers:
            continue
        points2d = np.array(
            [recon.images[iid].points2D[idx].xy for iid, idx, _ in cands]
        )
        points3d = np.array([recon.points3D[pid].xyz for _, _, pid in cands])
        camera_idxs = [
            slot_of_camera[recon.images[iid].camera_id] for iid, _, _ in cands
        ]
        res = pycolmap.estimate_and_refine_generalized_absolute_pose(
            points2d,
            points3d,
            camera_idxs,
            cams_from_rig,
            cameras,
            estimation_options=ransac,
        )
        if res is None or res["num_inliers"] < min_inliers:
            logger.info(
                "frame %d: re-registration failed (%s inliers of %d)",
                fid,
                "no" if res is None else res["num_inliers"],
                len(cands),
            )
            continue
        recon.frame(fid).rig_from_world = res["rig_from_world"]
        num_reregistered += 1
        logger.info(
            "frame %d: re-registered with %d/%d inliers (was %d linked obs)",
            fid,
            res["num_inliers"],
            len(cands),
            n_linked,
        )
    return num_reregistered


def fit_log_linear_correction(
    d_pred: np.ndarray,
    ratios: np.ndarray,
    rng: np.random.Generator,
    holdout_frac: float = 0.2,
    min_samples: int = 50,
    min_logd_spread: float = 0.15,
    iters: int = 3,
) -> tuple[float, float, dict] | None:
    """Robust fit of ``log(ratio) = a + b*log(d_pred)`` with held-out check.

    The multiplicative error field ``e(d) = exp(a + b*log d)`` generalizes
    the constant register-to-SfM scale (``b = 0``) to distance-dependent
    bias — the dominant depth-correlated monocular failure mode. Fitting
    is IRLS (Cauchy on the log-residual MAD); a random held-out subset
    arbitrates: the field is returned only when its held-out median
    |log residual| beats the constant model's, so a frame whose error
    really is a single scale keeps the old behavior. Returns
    ``(a, b, info)`` or ``None`` (use the constant scale).
    """
    if len(ratios) < min_samples:
        return None
    x = np.log(d_pred)
    y = np.log(ratios)
    if float(x.std()) < min_logd_spread:
        return None  # b is unconstrained on a narrow depth range
    idx = rng.permutation(len(x))
    n_hold = max(int(len(x) * holdout_frac), 10)
    hold, fit = idx[:n_hold], idx[n_hold:]
    xf, yf = x[fit], y[fit]
    w = np.ones(len(xf))
    a = b = 0.0
    for _ in range(iters):
        wsum = float(w.sum())
        xm = float((w * xf).sum()) / wsum
        ym = float((w * yf).sum()) / wsum
        var = float((w * (xf - xm) ** 2).sum())
        if var < 1e-9:
            return None
        b = float((w * (xf - xm) * (yf - ym)).sum()) / var
        a = ym - b * xm
        r = np.abs(yf - (a + b * xf))
        mad = float(np.median(r)) + 1e-9
        w = 1.0 / (1.0 + (r / (3.0 * 1.4826 * mad)) ** 2)
    const = float(np.median(yf))
    resid_field = float(np.median(np.abs(y[hold] - (a + b * x[hold]))))
    resid_const = float(np.median(np.abs(y[hold] - const)))
    if resid_field >= resid_const:
        return None
    return (
        a,
        b,
        {
            "holdout_median_field": resid_field,
            "holdout_median_const": resid_const,
            "num_fit": int(len(fit)),
            "num_holdout": int(len(hold)),
        },
    )


def rescale_dmaps_to_model(
    recon_dir: Path,
    registered_dir: Path,
    min_samples: int = 30,
    scale_range: tuple[float, float] = (0.05, 20.0),
    error_field: bool = True,
) -> dict[str, float]:
    """Per-frame depth registration against the SIFT model.

    GPS-less initializations accumulate multiplicative scale drift along
    the sequential star chain; the SIFT BA restores one globally
    consistent geometry, but the rigid per-frame pose correction cannot
    fix the depth VALUES. This applies the production mechanism
    (register-to-SfM): for every frame, compare triangulated track
    depths (SIFT and virtual alike) against the predicted depths at the
    same viewing directions. The constant per-frame scale is the median
    ratio; with ``error_field`` each frame is additionally offered the
    log-linear field ``e(d) = exp(a + b*log d)``
    (:func:`fit_log_linear_correction`) and takes it only when it wins
    on held-out samples. Frames without enough samples borrow the
    running median of the last trusted scales (densify_mapanything's
    guard against glassy/blank frames).

    Mutates ``registered_dir/dmaps`` in place and writes
    ``rescale_report.json``. Returns the per-frame (constant) scales.
    """
    report_path = registered_dir / "rescale_report.json"
    if report_path.exists():
        logger.info("dmaps already rescaled (%s exists); skipping", report_path)
        report = json.loads(report_path.read_text())
        return report.get("scales", report)

    recon = pycolmap.Reconstruction(str(recon_dir))
    camera = recon.cameras[min(recon.cameras)]
    focal = float(camera.params[0])
    cx, cy = float(camera.params[1]), float(camera.params[2])
    rig = single_rig(recon)
    identity = np.eye(3)
    rot_of_camera: dict[int, np.ndarray] = {}
    for cam_id in recon.cameras:
        sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, cam_id)
        rot_of_camera[cam_id] = (
            identity
            if rig.is_ref_sensor(sensor)
            else rig.sensor_from_rig(sensor).rotation.matrix()
        )

    scales: dict[str, float] = {}
    fields: dict[str, dict] = {}
    rng = np.random.default_rng(0)
    trusted: list[float] = []
    frames_sorted = sorted(
        recon.frames.values(),
        key=lambda f: pano_stem_of_image(
            recon.images[
                next(
                    d.id
                    for d in f.data_ids
                    if d.sensor_id.type == pycolmap.SensorType.CAMERA
                )
            ].name
        ),
    )
    for frame in frames_sorted:
        image_ids = [
            d.id
            for d in frame.data_ids
            if d.sensor_id.type == pycolmap.SensorType.CAMERA
        ]
        stem = pano_stem_of_image(recon.images[image_ids[0]].name)
        dmap_path = registered_dir / "dmaps" / f"{stem}.npy"
        if not dmap_path.exists():
            continue
        dmap = np.load(dmap_path)
        dm_h, dm_w = dmap.shape
        r_pano = frame.rig_from_world.rotation.matrix()
        center = -r_pano.T @ frame.rig_from_world.translation

        ratios: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for iid in image_ids:
            image = recon.images[iid]
            xys, pids = [], []
            for p2d in image.points2D:
                if p2d.point3D_id != pycolmap.INVALID_POINT3D_ID:
                    xys.append(p2d.xy)
                    pids.append(p2d.point3D_id)
            if not pids:
                continue
            xyz = np.stack([recon.points3D[p].xyz for p in pids])
            radial_sfm = np.linalg.norm(xyz - center, axis=1)
            xy = np.asarray(xys)
            dirs_face = np.column_stack(
                [
                    (xy[:, 0] - cx) / focal,
                    (xy[:, 1] - cy) / focal,
                    np.ones(len(xy)),
                ]
            )
            dirs_face /= np.linalg.norm(dirs_face, axis=1, keepdims=True)
            dirs_pano = dirs_face @ rot_of_camera[image.camera_id]
            u, v = _project_spherical(dirs_pano, dm_w, dm_h)
            d_pred = dmap[v, u]
            valid = np.isfinite(d_pred) & (d_pred > 0) & (radial_sfm > 0)
            if valid.any():
                ratios.append(radial_sfm[valid] / d_pred[valid])
                preds.append(d_pred[valid])
        samples = np.concatenate(ratios) if ratios else np.empty(0)
        d_samples = np.concatenate(preds) if preds else np.empty(0)
        scale = (
            float(np.median(samples)) if samples.size >= min_samples else None
        )
        own_scale = scale is not None and (
            scale_range[0] <= scale <= scale_range[1]
        )
        if own_scale:
            trusted.append(scale)
            trusted = trusted[-16:]
        elif trusted:
            scale = float(np.median(trusted))
        else:
            scale = 1.0
        scales[stem] = scale

        field = None
        if error_field and own_scale:
            field = fit_log_linear_correction(d_samples, samples, rng)
        rescaled = dmap.astype(np.float32)
        valid_px = dmap > 0
        if field is not None:
            a, b, info = field
            corr = np.ones_like(rescaled)
            corr[valid_px] = np.clip(
                np.exp(a + b * np.log(rescaled[valid_px])),
                scale_range[0],
                scale_range[1],
            )
            rescaled = rescaled * corr
            fields[stem] = {"a": a, "b": b, **info}
        else:
            rescaled = rescaled * scale
        rescaled[~valid_px] = 0.0
        np.save(dmap_path, rescaled)

    values = np.array(list(scales.values()))
    logger.info(
        "Registered %d dmaps to the SfM model: scale median %.3f "
        "(range %.3f..%.3f), %d/%d frames took the log-linear field",
        len(scales),
        float(np.median(values)),
        float(values.min()),
        float(values.max()),
        len(fields),
        len(scales),
    )
    report_path.write_text(json.dumps({"scales": scales, "fields": fields}))
    return scales


def load_gps_targets(
    manifest_path: Path,
    frame_id_by_stem: dict[str, int],
    registered_dir: Path | None = None,
) -> dict[int, np.ndarray]:
    """Per-frame ENU position targets from the PanoVGGT manifest.

    The model is metric ENU (batch_registration's ``align_output_to_gps``),
    so the manifest's ``enu_xyz`` entries are the targets directly. When the
    registered directory carries a ``reanchor.json`` (poses re-anchored to
    the production ``reference_lla.json`` origin), the same offset is added
    to the targets so priors and model share one frame.
    """
    offset = np.zeros(3)
    if registered_dir is not None:
        reanchor_path = registered_dir / "reanchor.json"
        if reanchor_path.exists():
            offset = np.asarray(
                json.loads(reanchor_path.read_text())["offset_enu"],
                dtype=np.float64,
            )
            logger.info("GPS targets re-anchored by offset %s", offset.round(3))
    manifest = json.loads(manifest_path.read_text())
    targets: dict[int, np.ndarray] = {}
    for entry in manifest.get("frames", []):
        enu = entry.get("enu_xyz")
        if enu is None or not np.all(np.isfinite(enu)):
            continue
        stem = Path(entry.get("filename", entry.get("name", ""))).stem
        fid = frame_id_by_stem.get(stem)
        if fid is not None:
            targets[fid] = np.asarray(enu, dtype=np.float64) + offset
    logger.info(
        "GPS targets: %d/%d frames carry a finite ENU fix",
        len(targets),
        len(frame_id_by_stem),
    )
    return targets


def filter_far_points(
    recon: pycolmap.Reconstruction, pad_factor: float = 3.0
) -> int:
    """Delete 3D points far outside the camera trajectory's bounding box.

    Near-parallel-ray triangulations can land points 1e6+ metres out; they
    survive the reprojection filters (their error is small) but wreck the
    model's bounding box for downstream consumers. The keep region is the
    camera-center bounding box padded by ``pad_factor`` times its diagonal.
    """
    centers = np.stack([frame_center(f) for f in recon.frames.values()])
    lo, hi = centers.min(axis=0), centers.max(axis=0)
    pad = pad_factor * max(float(np.linalg.norm(hi - lo)), 1.0)
    lo, hi = lo - pad, hi + pad
    removed = 0
    for pid in list(recon.points3D.keys()):
        xyz = recon.points3D[pid].xyz
        if np.any(xyz < lo) or np.any(xyz > hi):
            recon.delete_point3D(pid)
            removed += 1
    logger.info(
        "Filtered %d far points (outside camera bbox + %.0f m)", removed, pad
    )
    return removed


def frame_id_by_pano_stem(recon: pycolmap.Reconstruction) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for image in recon.images.values():
        mapping[pano_stem_of_image(image.name)] = image.frame_id
    return mapping


def export_pano_poses(
    recon: pycolmap.Reconstruction,
    frame_id_by_stem: dict[str, int],
    out_dir: Path,
) -> None:
    """pano_poses.json + pano_centers.ply + trajectory step stats."""
    panos = {}
    centers = []
    for stem, frame_id in sorted(frame_id_by_stem.items()):
        frame = recon.frame(frame_id)
        r_pano = (
            frame.rig_from_world.rotation.matrix()
        )  # rig frame == pano frame (face 0)
        center = frame_center(frame)
        q_xyzw = Rotation.from_matrix(r_pano).as_quat()
        panos[stem] = {
            "quat_wxyz_pano_from_world": [
                q_xyzw[3],
                q_xyzw[0],
                q_xyzw[1],
                q_xyzw[2],
            ],
            "R_pano_from_world": r_pano.tolist(),
            "center_world": center.tolist(),
            "num_faces": len(frame.data_ids),
        }
        centers.append(center)

    with open(out_dir / "pano_poses.json", "w") as f:
        json.dump(
            {
                "convention": (
                    "cam_from_world; rig-constrained BA; metric ENU frame; "
                    "pano frame: x right, y down, z forward"
                ),
                "panos": panos,
            },
            f,
            indent=2,
        )
    write_ply(out_dir / "pano_centers.ply", np.stack(centers))
    if len(centers) > 1:
        steps = np.linalg.norm(np.diff(np.stack(centers), axis=0), axis=1)
        logger.info(
            "Pano trajectory steps (m): median %.3f, p90 %.3f, max %.3f",
            *np.percentile(steps, [50, 90, 100]),
        )


def run_refine_stage(
    sfm_dir: Path,
    images_cubemap: Path,
    registered_dir: Path,
    out_dir: Path,
    manifest: Path | None = None,
    use_virtual_tracks: bool = True,
    min_pair_support: int = 300,
    fix_intrinsics: bool = True,
    max_reproj_error: float = 8.0,
    min_track_length: int = 2,
    reregister_rounds: int = 3,
    min_frame_obs: int = 300,
    min_inliers: int = 100,
    gps_prior_weight: float = 1.0,
    gps_prior_sigma_m: float = 0.25,
    image_folder: Path | None = None,
) -> pycolmap.Reconstruction:
    """Full refine stage; returns the final model (in sparse_enu/)."""
    recon = triangulate_sift(
        sfm_dir / "sparse_init",
        sfm_dir / "database.db",
        images_cubemap,
        sfm_dir / "sparse_tri",
    )

    if use_virtual_tracks:
        remapped = build_and_select_virtual_tracks(
            registered_dir, recon, min_pair_support, image_folder=image_folder
        )
        if remapped:
            recon, _virtual_pids = merge_virtual_tracks(recon, remapped)

    obs_lookup = build_obs_lookup(recon)
    engine = ReprojEngine(recon, obs_lookup)

    # GPS priors join EVERY BA round (production pose-prior-mapper style):
    # a BA without priors relaxes the model back to the visually-consistent
    # shape, re-accumulating the low-frequency drift the initialization
    # removed, and the final GPS rounds then start Huber-saturated.
    targets: dict[int, np.ndarray] = {}
    sigma_world = gps_prior_sigma_m / max(gps_prior_weight, 1e-6)
    if gps_prior_weight > 0 and manifest is not None and manifest.exists():
        targets = load_gps_targets(
            manifest, frame_id_by_pano_stem(recon), registered_dir
        )
        if len(targets) < 3:
            targets = {}
            logger.warning("Too few ENU targets; BA runs without GPS priors")

    def run_ba_round() -> None:
        if targets:
            run_gps_ba(recon, targets, sigma_world)
        else:
            run_ba(recon, fix_intrinsics)

    obs_removed, tracks_removed = engine.filter_observations(
        recon, 4 * max_reproj_error, min_track_length
    )
    logger.info(
        "Pre-BA filter: removed %d observations, %d tracks (> %.1f px at init)",
        obs_removed,
        tracks_removed,
        4 * max_reproj_error,
    )
    run_ba_round()
    logger.info(
        "After BA pass 1: mean reproj %.2f px", engine.mean_reproj_px(recon)
    )

    for round_idx in range(reregister_rounds):
        n_rereg = reregister_frames_rig(
            recon,
            obs_lookup,
            2 * max_reproj_error,
            min_frame_obs,
            min_inliers,
            linked_counts=engine.linked_counts_by_frame(),
        )
        relinked = engine.relink_observations(recon, 2 * max_reproj_error)
        logger.info(
            "Round %d: re-registered %d frames, re-linked %d observations",
            round_idx + 1,
            n_rereg,
            relinked,
        )
        if n_rereg == 0 and relinked == 0 and round_idx > 0:
            break
        obs_removed, tracks_removed = engine.filter_observations(
            recon, max_reproj_error, min_track_length
        )
        logger.info(
            "Filtered %d observations, %d tracks (> %.1f px)",
            obs_removed,
            tracks_removed,
            max_reproj_error,
        )
        run_ba_round()
        logger.info(
            "After BA round %d: mean reproj %.2f px",
            round_idx + 2,
            engine.mean_reproj_px(recon),
        )

    frame_ids = frame_id_by_pano_stem(recon)
    if targets:
        for gps_round in range(2):
            run_gps_ba(recon, targets, sigma_world)
            relinked = engine.relink_observations(recon, 2 * max_reproj_error)
            obs_removed, tracks_removed = engine.filter_observations(
                recon, max_reproj_error, min_track_length
            )
            logger.info(
                "GPS round %d: re-linked %d, filtered %d obs / %d tracks, "
                "mean reproj %.2f px",
                gps_round + 1,
                relinked,
                obs_removed,
                tracks_removed,
                engine.mean_reproj_px(recon),
            )

    filter_far_points(recon)

    out_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = out_dir / "sparse_enu"
    recon_dir.mkdir(exist_ok=True)
    recon.write(str(recon_dir))
    export_pano_poses(recon, frame_ids, out_dir)
    logger.info("Wrote %s", recon_dir)
    return recon


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sfm_dir",
        type=Path,
        required=True,
        help="dir with database.db + sparse_init/",
    )
    parser.add_argument("--images_cubemap", type=Path, required=True)
    parser.add_argument("--registered_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="ml_batches/manifest.json (GPS ENU targets)",
    )
    parser.add_argument(
        "--image_folder",
        type=Path,
        default=None,
        help="equirect images (virtual track colors)",
    )
    parser.add_argument("--no_virtual_tracks", action="store_true")
    parser.add_argument(
        "--min_pair_support",
        type=int,
        default=300,
        help="virtual tracks kept only for pano pairs with fewer "
        "SIFT tracks than this",
    )
    parser.add_argument(
        "--refine_intrinsics",
        action="store_true",
        help="refine focal (default: fixed analytic K)",
    )
    parser.add_argument("--max_reproj_error", type=float, default=8.0)
    parser.add_argument("--min_track_length", type=int, default=2)
    parser.add_argument("--reregister_rounds", type=int, default=3)
    parser.add_argument("--min_frame_obs", type=int, default=300)
    parser.add_argument("--min_inliers", type=int, default=100)
    parser.add_argument("--gps_prior_weight", type=float, default=1.0)
    parser.add_argument("--gps_prior_sigma_m", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    run_refine_stage(
        args.sfm_dir,
        args.images_cubemap,
        args.registered_dir,
        args.out_dir,
        manifest=args.manifest,
        use_virtual_tracks=not args.no_virtual_tracks,
        min_pair_support=args.min_pair_support,
        fix_intrinsics=not args.refine_intrinsics,
        max_reproj_error=args.max_reproj_error,
        min_track_length=args.min_track_length,
        reregister_rounds=args.reregister_rounds,
        min_frame_obs=args.min_frame_obs,
        min_inliers=args.min_inliers,
        gps_prior_weight=args.gps_prior_weight,
        gps_prior_sigma_m=args.gps_prior_sigma_m,
        image_folder=args.image_folder,
    )


if __name__ == "__main__":
    main()
