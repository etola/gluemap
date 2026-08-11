#!/usr/bin/env python3
"""PanoVGGT-driven SfM pipeline for equirectangular captures.

End-to-end driver: PanoVGGT depth/pose inference on overlapping batches
of equirect images, GPS-anchored global batch registration, cubemap
extraction (RGB + masks + perspective z-depth), masked SIFT on the
faces, rig-constrained augmented bundle adjustment with GPS priors, and
finally depth-map correction + exports for fusion.

Workfolder layout (inputs marked *):

    workfolder/
      images/*                  equirectangular panoramas (W = 2H); GPS from
                                exif_overrides.json sidecar (next to the
                                images or the dataset dir) or EXIF
      masks/*                   optional binary masks (255 = valid, 0 = invalid)
      segmentations/*           optional class-id label maps
                                (alternative to masks/)
      ml_batches/               [infer]    PanoVGGT batches + manifest.json
      initial_registration/     [register] poses.npz, dmaps/,
                                neighbors.json (metric ENU)
      images_cubemap/           [cubemap]  pano_camera{0..5}/ face RGB
      masks_cubemap/            [cubemap]  per-face masks (255 = valid)
      dmaps_cubemap/            [cubemap]  per-face perspective z-depth .npy
      sfm/                      [sift+refine] database.db, pairs.txt,
                                sparse_init/,
                                sparse_tri/, sparse_enu/, pano_poses.json
      registered/               [correct]  dmaps consistent with sparse_enu
      exports/                  [export]   camera_poses.json,
                                scene.mvsi (+ .ply)

Stages are idempotent: each is skipped when its primary output exists
(rerun with --force to redo). Select a range with --from_stage /
--until_stage.

Example:
    python scripts/pano_panovggt_pipeline.py --workfolder /data/site \
        --batch_size 10 --overlap 0.375 --batch_strategy gps
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

logger = logging.getLogger(__name__)

STAGES = [
    "infer",
    "register",
    "cubemap",
    "sift",
    "refine",
    "floorplan",
    "correct",
    "export",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _first_image_size(images_dir: Path) -> tuple[int, int]:
    import PIL.Image

    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTS:
            with PIL.Image.open(path) as img:
                return img.size  # (W, H)
    raise FileNotFoundError(f"No images found in {images_dir}")


def _find_manifest(workfolder: Path) -> Path:
    """The manifest holding GPS/ENU metadata for whichever init mode ran."""
    for candidate in (
        workfolder / "ml_batches" / "manifest.json",
        workfolder / "star_init" / "manifest.json",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No manifest under {workfolder}")


def _panovggt_python(args: argparse.Namespace) -> str:
    """Python interpreter of the PanoVGGT environment (star inference)."""
    if args.panovggt_python is not None:
        return str(args.panovggt_python)
    infer_cmd = Path(args.infer_cmd)
    if infer_cmd.is_absolute() and (infer_cmd.parent / "python").exists():
        return str(infer_cmd.parent / "python")
    return sys.executable


def stage_infer(args: argparse.Namespace) -> None:
    """S1: PanoVGGT inference — batches (default) or per-pano stars."""
    if args.init_mode == "star":
        star_dir = args.workfolder / "star_init"
        if (star_dir / "stars.json").exists() and not args.force:
            logger.info("[infer] %s exists; skipping", star_dir / "stars.json")
            return
        loop_pairs = star_dir / "loop_pairs.json"
        if not args.no_loop_closure and not loop_pairs.exists():
            # Appearance-based revisit candidates (SALAD, gluemap env):
            # the loop mini-stars pin the scale chain that a
            # sequential-only star graph leaves free to drift.
            star_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "pano_loop_pairs.py"),
                    "--images",
                    str(args.workfolder / "images"),
                    "--out",
                    str(loop_pairs),
                ],
                check=True,
            )
        cmd = [
            _panovggt_python(args),
            str(SCRIPTS_DIR / "pano_star_infer.py"),
            "--images",
            str(args.workfolder / "images"),
            "--out_dir",
            str(star_dir),
            "--star_size",
            str(args.star_size),
            "--seq_window",
            str(args.seq_window),
        ]
        if loop_pairs.exists():
            cmd += ["--extra_pairs", str(loop_pairs)]
        masks_dir = args.workfolder / "masks"
        if masks_dir.is_dir():
            cmd += ["--mask_dir", str(masks_dir)]
        if args.checkpoint is not None:
            cmd += ["--checkpoint", str(args.checkpoint)]
        logger.info("[infer] %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return
    manifest = args.workfolder / "ml_batches" / "manifest.json"
    if manifest.exists() and not args.force:
        logger.info("[infer] %s exists; skipping", manifest)
        return
    cmd = [
        args.infer_cmd,
        "--workfolder",
        str(args.workfolder),
        "--batch-size",
        str(args.batch_size),
        "--batch-overlap",
        str(args.overlap),
        "--batch-strategy",
        args.batch_strategy,
    ]
    masks_dir = args.workfolder / "masks"
    if masks_dir.is_dir():
        cmd += ["--mask-dir", str(masks_dir)]
    if args.checkpoint is not None:
        cmd += ["--checkpoint", str(args.checkpoint)]
    if args.infer_args:
        cmd += args.infer_args.split()
    logger.info("[infer] %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _resolve_reference_lla(workfolder: Path) -> dict:
    """The scene origin, replicating the production referential.

    The photogrammetry service anchors the reconstruction's ENU frame at
    the origin recorded in ``reference_lla.json`` (written by the
    walkthrough pre-SfM stage from the plan; schema
    ``{"latitude", "longitude", "altitude"}``). If the workfolder carries
    that file (e.g. copied from a production run of the same capture) it
    is used verbatim, so the two reconstructions share one referential.
    Otherwise a fallback origin is computed from the mean of the frames'
    GPS fixes (circular mean for longitude, altitude 0.0 — matching the
    service's ``GPSTrajectory3D.center`` convention) and written out.
    """
    ref_path = workfolder / "reference_lla.json"
    if ref_path.exists():
        reference = json.loads(ref_path.read_text())
        logger.info("[register] using existing %s", ref_path)
        return reference

    manifest = json.loads(_find_manifest(workfolder).read_text())
    lats, lons = [], []
    for frame in manifest.get("frames", []):
        gps = frame.get("gps")
        if gps and gps.get("lat_deg") is not None:
            lats.append(float(gps["lat_deg"]))
            lons.append(float(gps["lon_deg"]))
    if not lats:
        raise ValueError(
            "No reference_lla.json and no GPS in the manifest; cannot "
            "establish the scene origin"
        )
    lon_rad = np.radians(lons)
    reference = {
        "latitude": float(np.mean(lats)),
        "longitude": float(
            np.degrees(
                np.arctan2(np.mean(np.sin(lon_rad)), np.mean(np.cos(lon_rad)))
            )
        ),
        "altitude": 0.0,
    }
    ref_path.write_text(json.dumps(reference))
    logger.info("[register] wrote fallback %s (mean GPS origin)", ref_path)
    return reference


def _reanchor_to_reference(workfolder: Path, registered_dir: Path) -> None:
    """Translate the registered poses into the reference_lla ENU frame.

    ``infer_batched`` anchors its ENU frame at the centroid of the GPS
    fixes; the production reconstruction is anchored at
    ``reference_lla.json``. Both are local ENU tangent frames metres
    apart, so the mapping is a pure translation (the tangent-plane
    rotation between origins this close is microradians). The offset is
    recorded in ``reanchor.json`` so the refine stage shifts its GPS
    targets identically.
    """
    import pymap3d

    manifest = json.loads(_find_manifest(workfolder).read_text())
    anchor = manifest["gps"]["enu_anchor"]
    if anchor is None:
        # GPS-less capture: the solve was aligned to the plan frame
        # (start/end anchors) — nothing to translate. reference_lla is
        # recorded only when the plan provides one.
        ref_path = workfolder / "reference_lla.json"
        reference = (
            json.loads(ref_path.read_text()) if ref_path.exists() else None
        )
        (registered_dir / "reanchor.json").write_text(
            json.dumps(
                {"offset_enu": [0.0, 0.0, 0.0], "reference_lla": reference}
            )
        )
        logger.info("[register] no GPS anchor; plan-frame poses kept as-is")
        return
    reference = _resolve_reference_lla(workfolder)
    # The anchor's EXIF altitude is deliberately replaced by the reference
    # altitude: the production referential carries no altitude priors (its
    # vertical datum is the reference origin, with the walk plane near
    # z = 0), so injecting the EXIF altitude here would offset the whole
    # model vertically against the production reconstruction.
    offset = np.array(
        pymap3d.geodetic2enu(
            anchor["lat_deg"],
            anchor["lon_deg"],
            reference["altitude"],
            reference["latitude"],
            reference["longitude"],
            reference["altitude"],
        )
    )
    poses_path = registered_dir / "poses.npz"
    poses = dict(np.load(poses_path))
    poses["c2w"][:, :3, 3] += offset
    np.savez(poses_path, **poses)
    (registered_dir / "reanchor.json").write_text(
        json.dumps({"offset_enu": offset.tolist(), "reference_lla": reference})
    )
    logger.info(
        "[register] re-anchored poses to reference_lla (offset ENU %s m)",
        np.round(offset, 3).tolist(),
    )


def stage_register(args: argparse.Namespace) -> None:
    """S2: GPS-anchored global Sim(3) registration of the batches."""
    out_dir = args.workfolder / "initial_registration"
    if (out_dir / "poses.npz").exists() and not args.force:
        logger.info("[register] %s exists; skipping", out_dir / "poses.npz")
        return
    if args.init_mode == "star":
        from pano_star_solve import solve as star_solve

        start_end = args.workfolder / "start_end_points.json"
        star_solve(
            args.workfolder / "star_init",
            out_dir,
            min_edge_score=args.min_edge_score,
            start_end_json=start_end if start_end.exists() else None,
        )
        _reanchor_to_reference(args.workfolder, out_dir)
        return
    from ml_utils.batch_registration import (
        BatchRegistrationConf,
        run_from_workfolder,
    )

    conf = BatchRegistrationConf(
        gps_mode=args.gps_mode,
        sigma_gps_z=args.sigma_gps_z,
        per_frame_refine=not args.no_per_frame_refine,
        refine_z=not args.no_refine_z,
        # PanoVGGT can tilt individual batches (a monotone z-ramp inside a
        # batch is absorbed by the batch Sim(3) rotation, so neither the
        # batch solve nor the warp splitter sees it). The per-frame refine
        # is the mechanism that can bend it back onto GPS, but at the
        # library default (k=3) its IRLS treats a 1m+ arch as bad GPS and
        # refuses to chase; this pipeline pre-gates GPS quality, so trust
        # GPS harder here.
        refine_robust_k=args.refine_robust_k,
        # Split internally-warped batches (rigid per-batch Sim(3) cannot
        # fix a bend across a batch).
        batch_split_enabled=not args.no_batch_split,
    )
    run_from_workfolder(args.workfolder, conf, out_dir=out_dir)
    _reanchor_to_reference(args.workfolder, out_dir)


def stage_cubemap(args: argparse.Namespace) -> None:
    """S3: cubemap faces (RGB), per-face masks, and per-face z-depth maps."""
    images_dir = args.workfolder / "images"
    images_cubemap = args.workfolder / "images_cubemap"
    masks_cubemap = args.workfolder / "masks_cubemap"
    dmaps_cubemap = args.workfolder / "dmaps_cubemap"
    registered = args.workfolder / "initial_registration"

    pano_w, pano_h = _first_image_size(images_dir)
    face_w = pano_w // 4

    names = sorted(
        p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    face0 = images_cubemap / "pano_camera0"
    # Completeness check, not existence: an interrupted render (e.g. OOM)
    # must not be mistaken for a finished stage.
    missing = (
        names
        if args.force or not face0.is_dir()
        else [n for n in names if not (face0 / n).exists()]
    )
    if not missing:
        logger.info("[cubemap] %s complete; skipping RGB faces", images_cubemap)
    else:
        from ml_utils.pano import render_perspective_images

        logger.info(
            "[cubemap] rendering %d/%d panos to faces", len(missing), len(names)
        )
        seg_dir = args.workfolder / "segmentations"
        render_perspective_images(
            missing,
            images_dir,
            images_cubemap,
            args.workfolder / "segmentations_cubemap",
            seg_dir if seg_dir.is_dir() else args.workfolder / "segmentations",
            pano_w,
            pano_h,
        )

    masks_dir = args.workfolder / "masks"
    if masks_dir.is_dir():
        import cv2
        from ml_utils.pano import render_cubemap_mask

        for face_idx in range(6):
            (masks_cubemap / f"pano_camera{face_idx}").mkdir(
                parents=True, exist_ok=True
            )
        rendered = 0
        for mask_path in sorted(masks_dir.glob("*.png")):
            out0 = masks_cubemap / "pano_camera0" / mask_path.name
            if out0.exists() and not args.force:
                continue
            mask_eq = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_eq is None:
                logger.warning("[cubemap] unreadable mask %s", mask_path)
                continue
            faces = render_cubemap_mask(mask_eq, face_width=face_w)
            for face_idx, face in enumerate(faces):
                cv2.imwrite(
                    str(
                        masks_cubemap
                        / f"pano_camera{face_idx}"
                        / mask_path.name
                    ),
                    face,
                )
            rendered += 1
        logger.info("[cubemap] rendered face masks for %d panos", rendered)

    from ml_utils.pano import render_cubemap_depth_maps

    converted = render_cubemap_depth_maps(
        registered / "dmaps",
        dmaps_cubemap,
        face_width=args.depth_face_width,
        mask_dir=masks_dir if masks_dir.is_dir() else None,
    )
    logger.info(
        "[cubemap] converted %d equirect depth maps to face z-depths", converted
    )


def stage_sift(args: argparse.Namespace) -> None:
    """S4+S5: pairs from registered geometry, masked SIFT, posed rig model."""
    sfm_dir = args.workfolder / "sfm"
    if (sfm_dir / "sparse_init" / "frames.bin").exists() and not args.force:
        logger.info("[sift] %s exists; skipping", sfm_dir / "sparse_init")
        return
    from pano_sift_faces import run_sift_stage

    masks_cubemap = args.workfolder / "masks_cubemap"
    run_sift_stage(
        args.workfolder / "initial_registration",
        args.workfolder / "images_cubemap",
        sfm_dir,
        masks_cubemap=masks_cubemap if masks_cubemap.is_dir() else None,
        samples_per_frame=args.pair_samples,
        min_covis=args.pair_min_covis,
        device=args.device,
    )


def stage_refine(args: argparse.Namespace) -> None:
    """S6: triangulation + virtual tracks + rig-constrained BA + GPS priors."""
    sfm_dir = args.workfolder / "sfm"
    if (sfm_dir / "sparse_enu" / "frames.bin").exists() and not args.force:
        logger.info("[refine] %s exists; skipping", sfm_dir / "sparse_enu")
        return
    from pano_rig_refine import run_refine_stage

    run_refine_stage(
        sfm_dir,
        args.workfolder / "images_cubemap",
        args.workfolder / "initial_registration",
        sfm_dir,
        manifest=_find_manifest(args.workfolder),
        use_virtual_tracks=not args.no_virtual_tracks,
        min_pair_support=args.min_pair_support,
        fix_intrinsics=not args.refine_intrinsics,
        gps_prior_weight=args.gps_prior_weight,
        gps_prior_sigma_m=args.gps_prior_sigma_m,
        image_folder=args.workfolder / "images",
    )


def stage_floorplan(args: argparse.Namespace) -> None:
    """S6b (optional): snap the refined model onto the customer floor plan.

    Runs only when --floorplan_plan / --floorplan_anchors are given.
    The SE2 floor-plan BA corrects the sparse_enu rig frames in place
    (guarded: only applied when wall->plan distance improves), so the
    correct/export stages inherit the correction unchanged.
    """
    if args.floorplan_plan is None:
        logger.info("[floorplan] no --floorplan_plan; skipping")
        return
    if args.floorplan_anchors is None:
        raise SystemExit("--floorplan_plan requires --floorplan_anchors")
    out_dir = args.workfolder / "floorplan_ba"
    if (out_dir / "floorplan_report.json").exists() and not args.force:
        logger.info(
            "[floorplan] %s exists; skipping",
            out_dir / "floorplan_report.json",
        )
        return
    from pano_floorplan_ba import run as run_floorplan

    run_floorplan(
        args.workfolder,
        args.floorplan_plan,
        args.floorplan_anchors,
        registered_dir=args.workfolder / "initial_registration",
        out_dir=out_dir,
        solve=True,
        model_dir=args.workfolder / "sfm" / "sparse_enu",
        apply=True,
    )


def stage_correct(args: argparse.Namespace) -> None:
    """S7: carry the BA pose correction back onto the registered depth maps."""
    registered = args.workfolder / "registered"
    if (registered / "poses.npz").exists() and not args.force:
        logger.info("[correct] %s exists; skipping", registered / "poses.npz")
        return
    from ml_utils.batch_registration import (
        apply_external_correction_from_workfolder,
    )

    apply_external_correction_from_workfolder(
        args.workfolder,
        initial_registration_dir=args.workfolder / "initial_registration",
        corrected_colmap_dir=args.workfolder / "sfm" / "sparse_enu",
        out_dir=registered,
    )

    # GPS-less initializations carry per-frame scale drift that the rigid
    # pose correction cannot fix; register each depth map's scale to the
    # SIFT model (production's register-to-SfM mechanism).
    manifest = json.loads(_find_manifest(args.workfolder).read_text())
    if manifest["gps"]["enu_anchor"] is None or args.rescale_dmaps:
        from pano_rig_refine import rescale_dmaps_to_model

        rescale_dmaps_to_model(
            args.workfolder / "sfm" / "sparse_enu",
            registered,
            error_field=not args.no_error_field,
        )

    # Re-render the face z-depths from the corrected dmaps so the MVS inputs
    # match the final calibration.
    from ml_utils.pano import render_cubemap_depth_maps

    dmaps_cubemap = args.workfolder / "dmaps_cubemap"
    if dmaps_cubemap.is_dir():
        shutil.rmtree(dmaps_cubemap)
    masks_dir = args.workfolder / "masks"
    converted = render_cubemap_depth_maps(
        registered / "dmaps",
        dmaps_cubemap,
        face_width=args.depth_face_width,
        mask_dir=masks_dir if masks_dir.is_dir() else None,
    )
    logger.info(
        "[correct] re-rendered %d corrected face z-depth maps", converted
    )


def stage_export(args: argparse.Namespace) -> None:
    """S8: cameras.xml-compatible poses + depth-map fusion."""
    registered = args.workfolder / "registered"
    exports = args.workfolder / "exports"
    exports.mkdir(exist_ok=True)

    from ml_utils.panovggt_scene import get_cubemap_camera_poses

    poses = get_cubemap_camera_poses(registered)
    poses_path = exports / "camera_poses.json"
    with open(poses_path, "w") as f:
        json.dump(
            {
                "convention": "cam_to_world 4x4 per cubemap face image",
                "poses": {
                    name: np.asarray(mat).tolist() for name, mat in poses
                },
            },
            f,
        )
    logger.info("[export] wrote %d face poses to %s", len(poses), poses_path)

    if args.skip_fuse:
        return
    if args.fuse_via_colmap:
        # "COLMAP input" fusion path: ColmapInterface over the calibrated
        # sparse_enu model (enables SfM-dependent refinement stages).
        from ml_utils.fuse_depthmaps import FusionConfig, fuse_depthmaps

        conf = FusionConfig(
            dataset=args.workfolder,
            colmap_folder=Path("sfm") / "sparse_enu",
            image_folder=args.workfolder / "images",
            out_folder=exports,
            out_file=exports / "scene.mvsi",
            dmap_folder=registered / "dmaps",
            out_pts=exports / "scene.ply",
        )
        fuse_depthmaps(conf)
    else:
        from ml_utils.fuse_depthmaps import fuse_from_registered

        seg_dir = args.workfolder / "segmentations"
        masks_dir = args.workfolder / "masks"
        fuse_from_registered(
            registered,
            args.workfolder / "images",
            exports / "scene.mvsi",
            out_folder=exports,
            segmentation_folder=seg_dir if seg_dir.is_dir() else None,
            mask_folder=masks_dir if masks_dir.is_dir() else None,
        )
    logger.info("[export] fusion complete: %s", exports / "scene.mvsi")


STAGE_FUNCS = {
    "infer": stage_infer,
    "register": stage_register,
    "cubemap": stage_cubemap,
    "sift": stage_sift,
    "refine": stage_refine,
    "floorplan": stage_floorplan,
    "correct": stage_correct,
    "export": stage_export,
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workfolder", type=Path, required=True)
    parser.add_argument("--from_stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--until_stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument(
        "--force", action="store_true", help="redo stages even if outputs exist"
    )
    parser.add_argument("--device", default="cuda")
    # infer (S1)
    parser.add_argument(
        "--init_mode",
        choices=["batch", "star"],
        default="batch",
        help="'batch': overlapping-batch inference + batch_registration "
        "Sim(3) solve. 'star': one star per pano, fused by GLUEMAP's "
        "rotation/similarity averaging (redundant, no rigid-chain "
        "failure modes; ~star_size/overlap x the GPU cost)",
    )
    parser.add_argument(
        "--panovggt_python",
        type=Path,
        default=None,
        help="python of the PanoVGGT env for star inference (default: "
        "derived from --infer_cmd's directory)",
    )
    parser.add_argument("--star_size", type=int, default=8)
    parser.add_argument("--seq_window", type=int, default=3)
    parser.add_argument(
        "--no_loop_closure",
        action="store_true",
        help="skip SALAD loop-closure mini-stars (star mode)",
    )
    parser.add_argument(
        "--no_error_field",
        action="store_true",
        help="constant per-frame dmap rescale only (no log-linear "
        "error field) in the correct stage",
    )
    parser.add_argument(
        "--floorplan_plan",
        type=Path,
        default=None,
        help="floor-plan image; enables the floorplan stage (SE2 "
        "plan-snap BA on the refined model)",
    )
    parser.add_argument(
        "--floorplan_anchors",
        type=Path,
        default=None,
        help="JSON with the plan's corner coordinates (corners_lla or "
        "corners_enu, UL/UR/LR/LL)",
    )
    parser.add_argument(
        "--min_edge_score",
        type=float,
        default=0.15,
        help="star edges below this covisibility score are dropped",
    )
    parser.add_argument("--infer_cmd", default="ddpy-panovggt-infer-batched")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.375,
        help="overlapping batches are required: shared frames are the only "
        "inter-batch scale/orientation constraint",
    )
    parser.add_argument(
        "--batch_strategy", choices=["sequential", "gps"], default="gps"
    )
    # register (S2)
    parser.add_argument(
        "--gps_mode",
        choices=["horizontal", "anisotropic", "isotropic"],
        default="anisotropic",
        help="GPS prior mode for batch registration. 'anisotropic' with a "
        "finite --sigma_gps_z suppresses the vertical drift of chained "
        "batches on single-level walks; use 'horizontal' when the GPS "
        "altitude is truly unusable (multi-level capture with constant "
        "EXIF altitude)",
    )
    parser.add_argument("--sigma_gps_z", type=float, default=1.0)
    parser.add_argument(
        "--no_per_frame_refine",
        action="store_true",
        help="disable the smooth per-frame GPS delta refinement (it bends "
        "the registered trajectory onto the GPS track; rigid batch Sim(3)s "
        "alone cannot remove low-frequency drift)",
    )
    parser.add_argument(
        "--no_refine_z",
        action="store_true",
        help="keep reconstruction Z (use when GPS altitude is unusable, "
        "e.g. multi-level captures with constant EXIF altitude)",
    )
    parser.add_argument(
        "--no_batch_split",
        action="store_true",
        help="disable splitting of internally-warped batches",
    )
    parser.add_argument(
        "--refine_robust_k",
        type=float,
        default=10.0,
        help="IRLS outlier threshold of the per-frame GPS refine, in MAD "
        "units (library default 3.0 ignores reconstruction arches beyond "
        "~3 MAD as if they were bad GPS; lower this for noisy/raw GPS)",
    )
    parser.add_argument(
        "--infer_args",
        default="",
        help="extra args passed to the infer command",
    )
    # cubemap (S3)
    parser.add_argument(
        "--depth_face_width",
        type=int,
        default=None,
        help="face z-depth resolution (default: native dmap width / 4)",
    )
    # sift (S5)
    parser.add_argument("--pair_samples", type=int, default=400)
    parser.add_argument("--pair_min_covis", type=int, default=8)
    # refine (S6)
    parser.add_argument("--no_virtual_tracks", action="store_true")
    parser.add_argument("--min_pair_support", type=int, default=300)
    parser.add_argument("--refine_intrinsics", action="store_true")
    parser.add_argument("--gps_prior_weight", type=float, default=1.0)
    parser.add_argument("--gps_prior_sigma_m", type=float, default=0.25)
    # correct (S7)
    parser.add_argument(
        "--rescale_dmaps",
        action="store_true",
        help="force per-frame depth-scale registration to the SIFT model "
        "(automatic for GPS-less runs)",
    )
    # export (S8)
    parser.add_argument("--skip_fuse", action="store_true")
    parser.add_argument("--fuse_via_colmap", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    start = STAGES.index(args.from_stage)
    end = STAGES.index(args.until_stage)
    if start > end:
        raise SystemExit(
            f"--from_stage {args.from_stage} is after "
            f"--until_stage {args.until_stage}"
        )
    for stage in STAGES[start : end + 1]:
        logger.info("=== stage: %s ===", stage)
        STAGE_FUNCS[stage](args)


if __name__ == "__main__":
    main()
