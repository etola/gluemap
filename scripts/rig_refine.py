#!/usr/bin/env python3
"""Rig-constrained bundle adjustment for pano_to_views reconstructions.

Rebuilds a GLUEMAP reconstruction of rendered panorama views into a
rig-aware pycolmap reconstruction — one ``Frame`` per panorama whose
faces are sensors with fixed relative rotations from ``rig.json`` —
and runs bundle adjustment over frame poses and 3D points.

The shared-optical-center and fixed-relative-rotation constraints are
enforced exactly by the parameterization (COLMAP optimizes one
``rig_from_world`` pose per frame), which eliminates the intra-pano
center scatter of the unconstrained pipeline and reduces pose unknowns
by the number of faces per pano.

Frame poses are initialized from a robust per-pano consensus of the
input face poses (largest set of faces whose implied pano rotations
mutually agree, median center), so individual misregistered faces do
not corrupt the initialization.

Outputs (under --out_dir):
    * ``rig_aba/``       — the rig-constrained COLMAP reconstruction.
    * ``pano_poses.json`` — per-pano poses (exact, from frames).
    * ``pano_centers.ply`` — fused centers for a quick visual check.

Example:
    python scripts/rig_refine.py \
        --recon_path results/run1/gluemap_aba \
        --rig /data/pano_views/rig.json \
        --out_dir results/run1/rig
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recon_path",
        type=Path,
        required=True,
        help=(
            "input reconstruction with 3D points and tracks "
            "(e.g. <write_path>/gluemap_aba)"
        ),
    )
    parser.add_argument(
        "--rig",
        type=Path,
        required=True,
        help="rig.json manifest written by pano_to_views.py",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="output folder",
    )
    parser.add_argument(
        "--inlier_rotation_deg",
        type=float,
        default=2.0,
        help=(
            "face-consensus threshold used to initialize frame poses "
            "(same semantics as compose_pano_poses.py)"
        ),
    )
    parser.add_argument(
        "--fix_intrinsics",
        action="store_true",
        help=(
            "keep focal length fixed during BA (default: refine focal, "
            "keep principal point fixed)"
        ),
    )
    parser.add_argument(
        "--max_reproj_error",
        type=float,
        default=8.0,
        help=(
            "observation reprojection-error threshold (px) used by the "
            "filter/BA rounds (4x at initialization, 2x for "
            "re-linking)"
        ),
    )
    parser.add_argument(
        "--min_track_length",
        type=int,
        default=2,
        help="tracks falling below this length after filtering are dropped",
    )
    parser.add_argument(
        "--reregister_rounds",
        type=int,
        default=3,
        help=(
            "rounds of rig-PnP re-registration for frames whose "
            "observations were stripped by the reprojection filters"
        ),
    )
    parser.add_argument(
        "--min_frame_obs",
        type=int,
        default=300,
        help=(
            "frames with fewer surviving linked observations are "
            "considered lost and re-registered"
        ),
    )
    parser.add_argument(
        "--min_inliers",
        type=int,
        default=100,
        help="minimum rig-PnP inliers to accept a re-registration",
    )
    parser.add_argument(
        "--gps_prior_weight",
        type=float,
        default=0.0,
        help=(
            "enable GPS position priors in a final BA pass with this "
            "weight (0 = off, 1 = nominal). The prior sigma is "
            "--gps_prior_sigma_m / weight, so smaller weights soften "
            "the prior. Requires 'gps' entries in the rig manifest "
            "(pano_to_views.py records them from EXIF)"
        ),
    )
    parser.add_argument(
        "--gps_prior_sigma_m",
        type=float,
        default=0.25,
        help="nominal GPS position prior standard deviation in meters",
    )
    return parser.parse_args()


def rigid(R: np.ndarray, t: np.ndarray) -> pycolmap.Rigid3d:
    """Build a pycolmap.Rigid3d from a rotation matrix and translation."""
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return pycolmap.Rigid3d(
        rotation=q_xyzw, translation=np.asarray(t, dtype=np.float64)
    )


def geodesic_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices in degrees."""
    cos = (np.trace(r_a @ r_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def consensus_pose(
    face_entries: list[dict],
    inlier_deg: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Robust pano pose from per-face implied poses.

    Args:
        face_entries: Dicts with ``R_pano_from_world`` and ``center``.
        inlier_deg: Pairwise rotation agreement threshold.

    Returns:
        ``(R_pano_from_world, center, num_inliers)``.
    """
    n = len(face_entries)
    agree = np.zeros((n, n), dtype=bool)
    for a in range(n):
        for b in range(a + 1, n):
            agree[a, b] = agree[b, a] = (
                geodesic_deg(
                    face_entries[a]["R_pano_from_world"],
                    face_entries[b]["R_pano_from_world"],
                )
                < inlier_deg
            )
    best = max(range(n), key=lambda a: agree[a].sum())
    inliers = [a for a in range(n) if a == best or agree[best, a]]

    quats = Rotation.from_matrix(
        np.stack([face_entries[a]["R_pano_from_world"] for a in inliers])
    ).as_quat()
    signs = np.sign(quats @ quats[0])
    signs[signs == 0] = 1.0
    quats = quats * signs[:, None]
    _, eigvecs = np.linalg.eigh(quats.T @ quats)
    r_fused = Rotation.from_quat(eigvecs[:, -1]).as_matrix()
    c_fused = np.median(
        np.stack([face_entries[a]["center"] for a in inliers]), axis=0
    )
    return r_fused, c_fused, len(inliers)


def parse_view_name(name: str) -> tuple[str, str] | None:
    """Split ``<stem>__<suffix>.<ext>`` into (pano_stem, view_suffix)."""
    stem = Path(name).stem
    if "__" not in stem:
        return None
    pano_stem, _, suffix = stem.rpartition("__")
    return pano_stem, suffix


def write_ply(path: Path, points: np.ndarray) -> None:
    """Write Nx3 points as a minimal ASCII PLY point cloud."""
    with open(path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def build_rig_reconstruction(
    source: pycolmap.Reconstruction,
    rig_manifest: dict,
    inlier_deg: float,
) -> tuple[pycolmap.Reconstruction, dict[str, int], list[tuple[int, int, int]]]:
    """Rebuild ``source`` with one rig frame per panorama.

    Image ids, 2D points and 3D points are carried over unchanged (3D
    point ids are renumbered by pycolmap but observations stay
    consistent); only the pose parameterization changes.

    Returns:
        ``(reconstruction, frame_id_by_stem, obs_lookup)`` where
        ``obs_lookup`` lists every original observation as
        ``(image_id, point2D_idx, point3D_id)``.
    """
    suffixes = [v["suffix"] for v in rig_manifest["views"]]
    r_view_from_pano = {
        v["suffix"]: np.array(v["R_view_from_pano"])
        for v in rig_manifest["views"]
    }
    ref_suffix = suffixes[0]
    r_ref = r_view_from_pano[ref_suffix]
    face_slot = {sfx: i for i, sfx in enumerate(suffixes)}

    # Source camera params (single shared camera expected)
    src_cam = source.cameras[next(iter(source.cameras))]

    recon = pycolmap.Reconstruction()
    for i, _sfx in enumerate(suffixes):
        recon.add_camera(
            pycolmap.Camera(
                camera_id=i + 1,
                model=src_cam.model.name,
                width=src_cam.width,
                height=src_cam.height,
                params=list(src_cam.params),
            )
        )

    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 1))
    for i, sfx in enumerate(suffixes[1:], start=2):
        # cam_face_from_rig with the rig frame anchored at the ref face
        r_face_from_ref = r_view_from_pano[sfx] @ r_ref.T
        rig.add_sensor(
            pycolmap.sensor_t(pycolmap.SensorType.CAMERA, i),
            rigid(r_face_from_ref, np.zeros(3)),
        )
    recon.add_rig(rig)

    # Group source images per pano and compute implied pano poses
    faces = defaultdict(list)
    for image in source.images.values():
        parsed = parse_view_name(image.name)
        if parsed is None or parsed[1] not in r_view_from_pano:
            continue
        stem, sfx = parsed
        pose = image.cam_from_world()
        faces[stem].append(
            {
                "suffix": sfx,
                "image": image,
                "R_pano_from_world": (
                    r_view_from_pano[sfx].T @ pose.rotation.matrix()
                ),
                "center": np.array(image.projection_center()),
            }
        )

    # Pre-assign new sequential 3D point ids (pycolmap.add_point3D
    # assigns 1, 2, 3, ... in call order and validates that the
    # referenced 2D points already carry the new id, so the ids must be
    # known before the images are built).
    kept_image_ids = {e["image"].image_id for fl in faces.values() for e in fl}
    new_pid = {}
    for pid, point in source.points3D.items():
        n_obs = sum(
            el.image_id in kept_image_ids for el in point.track.elements
        )
        if n_obs >= 2:
            new_pid[pid] = len(new_pid) + 1

    frame_id_by_stem = {}
    inlier_counts = []
    for frame_id, (stem, face_list) in enumerate(
        sorted(faces.items()), start=1
    ):
        r_pano, c_pano, n_inl = consensus_pose(face_list, inlier_deg)
        inlier_counts.append(n_inl)

        # Add the frame WITHOUT a pose first: a posed frame is
        # auto-registered by add_frame before its images exist, which
        # leaves the registered-image bookkeeping empty and produces a
        # reconstruction that serializes zero images. The pose is set
        # and the frame registered after all its images are attached.
        frame = pycolmap.Frame(frame_id=frame_id, rig_id=1)
        recon.add_frame(frame)
        frame_id_by_stem[stem] = frame_id

        for entry in face_list:
            src_im = entry["image"]
            im = pycolmap.Image(
                image_id=src_im.image_id,
                camera_id=face_slot[entry["suffix"]] + 1,
                name=src_im.name,
                frame_id=frame_id,
            )
            im.points2D = pycolmap.Point2DList(
                [
                    pycolmap.Point2D(
                        p.xy,
                        point3D_id=new_pid.get(
                            p.point3D_id, pycolmap.INVALID_POINT3D_ID
                        ),
                    )
                    for p in src_im.points2D
                ]
            )
            recon.frame(frame_id).add_data_id(im.data_id)
            recon.add_image(im)

        r_rig_from_world = r_ref @ r_pano
        recon.frame(frame_id).rig_from_world = rigid(
            r_rig_from_world, -r_rig_from_world @ c_pano
        )
        recon.register_frame(frame_id)

    ic = np.array(inlier_counts)
    logger.info(
        "Initialized %d frames; consensus inlier faces: median %d, min %d",
        len(frame_id_by_stem),
        int(np.median(ic)),
        int(ic.min()),
    )

    # Copy 3D points in the pre-assigned id order.
    for pid in sorted(new_pid, key=new_pid.get):
        point = source.points3D[pid]
        track = pycolmap.Track()
        for el in point.track.elements:
            if el.image_id in recon.images:
                track.add_element(el.image_id, el.point2D_idx)
        assigned = recon.add_point3D(point.xyz, track, point.color)
        assert assigned == new_pid[pid], (
            f"point id mismatch: {assigned} != {new_pid[pid]}"
        )

    logger.info(
        "Carried over %d/%d 3D points",
        len(new_pid),
        source.num_points3D(),
    )

    # Full observation lookup (survives later filtering) used by the
    # re-registration rounds: (image_id, point2D_idx, point3D_id).
    obs_lookup = []
    for image in recon.images.values():
        for idx, p2d in enumerate(image.points2D):
            if p2d.point3D_id != pycolmap.INVALID_POINT3D_ID:
                obs_lookup.append((image.image_id, idx, p2d.point3D_id))

    return recon, frame_id_by_stem, obs_lookup


def mean_reproj_px(recon: pycolmap.Reconstruction) -> float:
    """Mean reprojection error over all observations, in pixels.

    Computed directly (``Reconstruction.compute_mean_reprojection_error``
    averages the per-point error fields, which are unset on freshly
    constructed points). Behind-camera observations count as 2x the
    image diagonal.
    """
    total, count = 0.0, 0
    for point in recon.points3D.values():
        for el in point.track.elements:
            image = recon.images[el.image_id]
            camera = recon.cameras[image.camera_id]
            x_cam = image.cam_from_world() * point.xyz
            if x_cam[2] <= 0:
                err = 2.0 * float(np.hypot(camera.width, camera.height))
            else:
                proj = camera.img_from_cam(x_cam)
                err = float(
                    np.linalg.norm(proj - image.points2D[el.point2D_idx].xy)
                )
            total += err
            count += 1
    return total / max(count, 1)


def run_ba(recon: pycolmap.Reconstruction, fix_intrinsics: bool) -> None:
    """Run rig-constrained global bundle adjustment in place."""
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_rig_from_world = True
    opts.refine_sensor_from_rig = False
    opts.refine_focal_length = not fix_intrinsics
    opts.refine_principal_point = False
    opts.refine_extra_params = False
    opts.print_summary = True
    opts.ceres.solver_options.max_num_iterations = 100
    # Robust loss guards against residual outliers that survive the
    # reprojection-error prefilters.
    opts.ceres.loss_function_type = (
        pycolmap.LossFunctionType.SOFT_L1
        if hasattr(pycolmap, "LossFunctionType")
        else opts.ceres.loss_function_type
    )
    pycolmap.bundle_adjustment(recon, opts)


def filter_observations(
    recon: pycolmap.Reconstruction,
    max_error_px: float,
    min_track_length: int,
) -> tuple[int, int]:
    """Drop observations with reprojection error above ``max_error_px``.

    Returns ``(num_observations_removed, num_tracks_removed)``.
    """
    obs_removed = 0
    tracks_removed = 0
    for pid in list(recon.points3D.keys()):
        point = recon.points3D[pid]
        bad = []
        for el in point.track.elements:
            image = recon.images[el.image_id]
            camera = recon.cameras[image.camera_id]
            x_cam = image.cam_from_world() * point.xyz
            if x_cam[2] <= 0:
                bad.append((el.image_id, el.point2D_idx))
                continue
            proj = camera.img_from_cam(x_cam)
            err = np.linalg.norm(proj - image.points2D[el.point2D_idx].xy)
            if err > max_error_px:
                bad.append((el.image_id, el.point2D_idx))
        if point.track.length() - len(bad) < min_track_length:
            obs_removed += point.track.length()
            tracks_removed += 1
            recon.delete_point3D(pid)
        else:
            for image_id, p2d_idx in bad:
                point.track.delete_element(image_id, p2d_idx)
                recon.images[image_id].reset_point3D_for_point2D(p2d_idx)
                obs_removed += 1
    return obs_removed, tracks_removed


def reregister_frames(
    recon: pycolmap.Reconstruction,
    obs_lookup: list[tuple[int, int, int]],
    rig_manifest: dict,
    max_error_px: float,
    min_frame_obs: int,
    min_inliers: int,
) -> int:
    """Re-localize under-constrained frames via generalized (rig) PnP.

    Frames whose surviving observation count dropped below
    ``min_frame_obs`` (their initialization was wrong, so the
    reprojection filters stripped them) are re-estimated with
    LO-RANSAC from their original 2D-3D correspondences into the
    current map, using all faces of the rig jointly.

    Returns:
        Number of frames whose pose was replaced.
    """
    suffixes = [v["suffix"] for v in rig_manifest["views"]]
    r_view_from_pano = {
        v["suffix"]: np.array(v["R_view_from_pano"])
        for v in rig_manifest["views"]
    }
    r_ref = r_view_from_pano[suffixes[0]]
    cams_from_rig = [
        rigid(r_view_from_pano[sfx] @ r_ref.T, np.zeros(3)) for sfx in suffixes
    ]
    cameras = [recon.cameras[i + 1] for i in range(len(suffixes))]

    # Current linked observations per frame
    linked = {fid: 0 for fid in recon.frames}
    for image in recon.images.values():
        n = sum(
            p.point3D_id != pycolmap.INVALID_POINT3D_ID for p in image.points2D
        )
        linked[image.frame_id] += n

    # Candidate correspondences per frame from the full lookup
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
        points2D = np.array(
            [recon.images[iid].points2D[idx].xy for iid, idx, _ in cands]
        )
        points3D = np.array([recon.points3D[pid].xyz for _, _, pid in cands])
        camera_idxs = [recon.images[iid].camera_id - 1 for iid, _, _ in cands]
        res = pycolmap.estimate_and_refine_generalized_absolute_pose(
            points2D,
            points3D,
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


def relink_observations(
    recon: pycolmap.Reconstruction,
    obs_lookup: list[tuple[int, int, int]],
    max_error_px: float,
) -> int:
    """Re-attach filtered-out observations that fit the current poses.

    Returns the number of observations re-linked.
    """
    relinked = 0
    for image_id, idx, pid in obs_lookup:
        if pid not in recon.points3D:
            continue
        image = recon.images[image_id]
        p2d = image.points2D[idx]
        if p2d.point3D_id != pycolmap.INVALID_POINT3D_ID:
            continue
        camera = recon.cameras[image.camera_id]
        x_cam = image.cam_from_world() * recon.points3D[pid].xyz
        if x_cam[2] <= 0:
            continue
        err = np.linalg.norm(camera.img_from_cam(x_cam) - p2d.xy)
        if err <= max_error_px:
            recon.points3D[pid].track.add_element(image_id, idx)
            # set_point3D_for_point2D keeps the image's cached
            # num_points3D counter in sync (a bare point3D_id
            # assignment does not, which breaks Reconstruction
            # validation later).
            image.set_point3D_for_point2D(idx, pid)
            relinked += 1
    return relinked


def frame_center(frame: "pycolmap.Frame") -> np.ndarray:
    """World-space center of a frame's rig reference sensor."""
    r = frame.rig_from_world.rotation.matrix()
    return -r.T @ frame.rig_from_world.translation


def compute_gps_targets(
    recon: pycolmap.Reconstruction,
    frame_id_by_stem: dict[str, int],
    rig_manifest: dict,
    sigma_m: float,
    weight: float,
) -> tuple[dict[int, np.ndarray], float] | None:
    """GPS position targets in reconstruction-world units.

    Fits a robust sim(3) from current frame centers to the GPS track in
    a local ENU frame (reusing the georegister helpers), then maps every
    GPS fix back into the reconstruction frame.

    Returns:
        ``(targets_by_frame_id, sigma_world)`` or ``None`` when there is
        no usable GPS.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from georegister import enu_frame, geodetic_to_ecef, robust_sim3

    stems = [
        s
        for s in sorted(frame_id_by_stem)
        if "gps" in rig_manifest["panos"].get(s, {})
    ]
    if len(stems) < 3:
        logger.warning(
            "GPS priors requested but only %d panos carry GPS", len(stems)
        )
        return None

    fixes = np.array(
        [
            [
                rig_manifest["panos"][s]["gps"]["lat"],
                rig_manifest["panos"][s]["gps"]["lon"],
                rig_manifest["panos"][s]["gps"]["alt"],
            ]
            for s in stems
        ]
    )
    origin = np.median(fixes, axis=0)
    r_enu = enu_frame(origin[0], origin[1])
    ecef0 = geodetic_to_ecef(*origin)
    enu = np.stack([r_enu @ (geodetic_to_ecef(*f) - ecef0) for f in fixes])

    centers = np.stack(
        [frame_center(recon.frame(frame_id_by_stem[s])) for s in stems]
    )
    scale, r, t, inliers = robust_sim3(centers, enu, threshold=1.0)
    logger.info(
        "GPS sim(3) fit: scale %.4f m/unit, %d/%d inliers",
        scale,
        int(inliers.sum()),
        len(stems),
    )

    targets = {
        frame_id_by_stem[s]: r.T @ (enu[i] - t) / scale
        for i, s in enumerate(stems)
    }
    sigma_world = sigma_m / scale / weight
    logger.info(
        "GPS prior sigma: %.3f m -> %.4f world units (weight %.2f)",
        sigma_m,
        sigma_world,
        weight,
    )
    return targets, sigma_world


def run_gps_ba(
    recon: pycolmap.Reconstruction,
    targets: dict[int, np.ndarray],
    sigma_world: float,
    max_iterations: int = 100,
) -> None:
    """Rig BA with GPS position priors, assembled with pyceres.

    Reprojection residuals use COLMAP's ``RigReprojErrorCost`` with the
    constant ``cam_from_rig`` from the reconstruction's rig; each frame
    with a GPS fix additionally gets an ``AbsolutePosePositionPriorCost``
    with isotropic covariance ``sigma_world**2``. Intrinsics are held
    constant (they are refined by the earlier pycolmap BA rounds).
    Frame poses and 3D points are written back on completion.
    """
    import pyceres
    import pygluemap
    from pycolmap import cost_functions as cf

    prob = pyceres.Problem()
    loss = pyceres.SoftLOneLoss(1.0)

    frame_pose = {
        fid: np.array(f.rig_from_world.params)
        for fid, f in recon.frames.items()
    }
    point_params = {}
    # All face cameras share identical intrinsics by construction;
    # use a single constant block.
    shared_cam = np.array(recon.cameras[min(recon.cameras)].params)

    rigs = list(recon.rigs.values())
    assert len(rigs) == 1, f"expected exactly 1 rig, got {len(rigs)}"
    rig = rigs[0]
    identity = pycolmap.Rigid3d(
        rotation=np.array([0.0, 0.0, 0.0, 1.0]), translation=np.zeros(3)
    )
    cam_from_rig = {}
    for cam_id in recon.cameras:
        sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, cam_id)
        cam_from_rig[cam_id] = (
            identity
            if rig.is_ref_sensor(sensor)
            else rig.sensor_from_rig(sensor)
        )

    num_res = 0
    frames_in_problem = set()
    for pid, point in recon.points3D.items():
        point_params[pid] = np.array(point.xyz)
        for el in point.track.elements:
            image = recon.images[el.image_id]
            cost = cf.RigReprojErrorCost(
                recon.cameras[image.camera_id].model,
                image.points2D[el.point2D_idx].xy,
                cam_from_rig[image.camera_id],
            )
            prob.add_residual_block(
                cost,
                loss,
                [point_params[pid], frame_pose[image.frame_id], shared_cam],
            )
            frames_in_problem.add(image.frame_id)
            num_res += 1

    # Robust loss on the priors: post-processed GPS tracks can drift by
    # tens of meters in building interiors (observed: coherent ~25 m
    # offsets over whole sections of a walk). Residuals are whitened by
    # the covariance, so the Huber delta is in units of sigma - beyond
    # 3 sigma the prior's influence grows only linearly and grossly
    # wrong fixes cannot overpower the visual structure.
    prior_loss = pyceres.HuberLoss(3.0)
    cov = np.eye(3) * sigma_world**2
    num_priors = 0
    for fid, target in targets.items():
        if fid not in frame_pose:
            continue
        prob.add_residual_block(
            cf.AbsolutePosePositionPriorCost(cov, target),
            prior_loss,
            [frame_pose[fid]],
        )
        frames_in_problem.add(fid)
        num_priors += 1

    # Manifolds/constants only for blocks that entered the problem
    # (frames with neither observations nor GPS have no residuals).
    prob.set_parameter_block_constant(shared_cam)
    for fid in frames_in_problem:
        prob.set_manifold(frame_pose[fid], pygluemap.CreatePoseManifold())
    logger.info(
        "GPS BA: %d reprojection residuals, %d position priors, "
        "%d/%d frames in problem",
        num_res,
        num_priors,
        len(frames_in_problem),
        len(frame_pose),
    )

    opts = pyceres.SolverOptions()
    opts.max_num_iterations = max_iterations
    opts.num_threads = os.cpu_count() or 1
    summary = pyceres.SolverSummary()
    pyceres.solve(opts, prob, summary)
    logger.info("GPS BA: %s", summary.BriefReport())

    # Write back
    for fid, params in frame_pose.items():
        recon.frame(fid).rig_from_world = pycolmap.Rigid3d(
            rotation=params[:4], translation=params[4:]
        )
    for pid, xyz in point_params.items():
        recon.points3D[pid].xyz = xyz


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()

    with open(args.rig) as f:
        rig_manifest = json.load(f)

    source = pycolmap.Reconstruction(str(args.recon_path))
    logger.info(
        "Source: %d images, %d points3D, mean reproj %.2f px",
        source.num_images(),
        source.num_points3D(),
        source.compute_mean_reprojection_error(),
    )

    recon, frame_id_by_stem, obs_lookup = build_rig_reconstruction(
        source, rig_manifest, args.inlier_rotation_deg
    )
    logger.info(
        "Rig reconstruction: %d frames, mean reproj before BA %.2f px",
        recon.num_frames(),
        mean_reproj_px(recon),
    )

    # Observations from outlier faces / broken trajectory sections are
    # grossly wrong under the consensus initialization; drop them
    # before the first BA so they cannot derail it. They are candidates
    # for re-linking after re-registration.
    obs_removed, tracks_removed = filter_observations(
        recon, 4 * args.max_reproj_error, args.min_track_length
    )
    logger.info(
        "Pre-BA filter: removed %d observations, %d tracks (> %.1f px at init)",
        obs_removed,
        tracks_removed,
        4 * args.max_reproj_error,
    )

    run_ba(recon, args.fix_intrinsics)
    logger.info("After BA pass 1: mean reproj %.2f px", mean_reproj_px(recon))

    # Re-registration rounds: rig-PnP frames that lost their
    # observations to the filters, re-link now-consistent observations,
    # filter, and re-run BA.
    for round_idx in range(args.reregister_rounds):
        n_rereg = reregister_frames(
            recon,
            obs_lookup,
            rig_manifest,
            2 * args.max_reproj_error,
            args.min_frame_obs,
            args.min_inliers,
        )
        relinked = relink_observations(
            recon, obs_lookup, 2 * args.max_reproj_error
        )
        logger.info(
            "Round %d: re-registered %d frames, re-linked %d observations",
            round_idx + 1,
            n_rereg,
            relinked,
        )
        if n_rereg == 0 and relinked == 0 and round_idx > 0:
            break
        obs_removed, tracks_removed = filter_observations(
            recon, args.max_reproj_error, args.min_track_length
        )
        logger.info(
            "Filtered %d observations, %d tracks (> %.1f px)",
            obs_removed,
            tracks_removed,
            args.max_reproj_error,
        )
        run_ba(recon, args.fix_intrinsics)
        logger.info(
            "After BA round %d: mean reproj %.2f px",
            round_idx + 2,
            mean_reproj_px(recon),
        )

    # --- GPS-prior BA ----------------------------------------------------
    if args.gps_prior_weight > 0:
        for gps_round in range(2):
            result = compute_gps_targets(
                recon,
                frame_id_by_stem,
                rig_manifest,
                args.gps_prior_sigma_m,
                args.gps_prior_weight,
            )
            if result is None:
                break
            targets, sigma_world = result
            run_gps_ba(recon, targets, sigma_world)
            # Frames moved onto their GPS anchors may fit previously
            # filtered observations again; relink and clean up.
            relinked = relink_observations(
                recon, obs_lookup, 2 * args.max_reproj_error
            )
            obs_removed, tracks_removed = filter_observations(
                recon, args.max_reproj_error, args.min_track_length
            )
            logger.info(
                "GPS round %d: re-linked %d, filtered %d obs / %d "
                "tracks, mean reproj %.2f px",
                gps_round + 1,
                relinked,
                obs_removed,
                tracks_removed,
                mean_reproj_px(recon),
            )

    # --- Outputs -------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = args.out_dir / "rig_aba"
    recon_dir.mkdir(exist_ok=True)
    recon.write(str(recon_dir))

    suffixes = [v["suffix"] for v in rig_manifest["views"]]
    r_ref = np.array(rig_manifest["views"][0]["R_view_from_pano"])
    panos = {}
    centers = []
    for stem, frame_id in sorted(frame_id_by_stem.items()):
        frame = recon.frame(frame_id)
        r_rig_w = frame.rig_from_world.rotation.matrix()
        r_pano = r_ref.T @ r_rig_w
        c = -r_rig_w.T @ frame.rig_from_world.translation
        q_xyzw = Rotation.from_matrix(r_pano).as_quat()
        panos[stem] = {
            "quat_wxyz_pano_from_world": [
                q_xyzw[3],
                q_xyzw[0],
                q_xyzw[1],
                q_xyzw[2],
            ],
            "R_pano_from_world": r_pano.tolist(),
            "center_world": c.tolist(),
            "num_faces": len(frame.data_ids),
        }
        centers.append(c)

    poses_path = args.out_dir / "pano_poses.json"
    with open(poses_path, "w") as f:
        json.dump(
            {
                "convention": (
                    "cam_from_world; rig-constrained BA; pano frame: "
                    "x right, y down, z forward"
                ),
                "recon_path": str(args.recon_path),
                "rig": str(args.rig),
                "num_view_slots": len(suffixes),
                "panos": panos,
            },
            f,
            indent=2,
        )
    write_ply(args.out_dir / "pano_centers.ply", np.stack(centers))

    steps = np.linalg.norm(np.diff(np.stack(centers), axis=0), axis=1)
    logger.info(
        "Pano trajectory steps: median %.4f, p90 %.4f, max %.4f",
        *np.percentile(steps, [50, 90, 100]),
    )
    logger.info("Wrote %s and %s", recon_dir, poses_path)


if __name__ == "__main__":
    main()
