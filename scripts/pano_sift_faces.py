#!/usr/bin/env python3
"""SIFT-on-cubemap-faces stage for the PanoVGGT equirectangular pipeline.

Consumes a ``batch_registration`` output directory (``poses.npz`` with
world-frame ``c2w`` + ``frame_names``, ``dmaps/<name>.npy`` registered
radial depth, ``neighbors.json``) plus the cubemap faces rendered from
the equirectangular images, and produces:

    * ``pairs.txt``      — face-level match pairs derived from the
      registered geometry (pano neighbor graph + per-face covisibility
      of unprojected depth samples). Same-pano face pairs are excluded
      (zero baseline: no scale information, mirror-track hazard).
    * ``database.db``    — COLMAP database with masked SIFT features on
      the faces (fixed analytic intrinsics) and matches over pairs.txt.
    * ``sparse_init/``   — a rig-aware pycolmap reconstruction: one
      ``Frame`` per panorama posed from the registered PanoVGGT poses,
      6 face sensors with the exactly-known cubemap rotations, no 3D
      points (the refine stage triangulates SIFT into it).

The face layout and rotations follow ``ml_utils.pano`` conventions
(``pano_camera{0..5}/`` prefixes, ``get_cubemap_rotations()`` order,
face 0 = pano frame). Faces are 90x90 degrees, so the analytic
intrinsics are exact by construction: ``f = W / 2``, principal point at
the image center (COLMAP half-pixel convention).

Example:
    python scripts/pano_sift_faces.py \
        --registered_dir /data/site/initial_registration \
        --images_cubemap /data/site/images_cubemap \
        --masks_cubemap /data/site/masks_cubemap \
        --out_dir /data/site/sfm
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

NUM_FACES = 6
FACE_PREFIX = "pano_camera"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def get_cubemap_rotations() -> list[np.ndarray]:
    """The 6 cam_from_pano rotations (= ml_utils.pano.get_cubemap_rotations).

    Face order: yaw 0/90/180/270 (faces 0-3), then pitch -90 and +90
    (faces 4-5). Face 0 is the pano frame itself (identity). Imported
    from ml_utils when available so the convention has a single source
    of truth; the fallback keeps this script runnable standalone.
    """
    try:
        from ml_utils.pano import get_cubemap_rotations as ml_rotations

        return [np.asarray(r) for r in ml_rotations()]
    except ImportError:
        pass
    rotations = []
    for yaw_deg in np.linspace(0, 360, 4, endpoint=False):
        rotations.append(
            Rotation.from_euler("XY", [0.0, -yaw_deg], degrees=True).as_matrix()
        )
    rotations.append(
        Rotation.from_euler("XY", [-90.0, 0.0], degrees=True).as_matrix()
    )
    rotations.append(
        Rotation.from_euler("XY", [90.0, 0.0], degrees=True).as_matrix()
    )
    return rotations


def rigid(R: np.ndarray, t: np.ndarray) -> pycolmap.Rigid3d:
    """Build a pycolmap.Rigid3d from a rotation matrix and translation."""
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return pycolmap.Rigid3d(
        rotation=q_xyzw, translation=np.asarray(t, dtype=np.float64)
    )


def load_registered_poses(registered_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load frame names and world-frame c2w poses from poses.npz."""
    poses = np.load(registered_dir / "poses.npz")
    names = [str(n) for n in poses["frame_names"]]
    c2w = np.asarray(poses["c2w"], dtype=np.float64)
    if len(names) != c2w.shape[0]:
        raise ValueError(
            f"poses.npz mismatch: {len(names)} names vs {c2w.shape[0]} poses"
        )
    return names, c2w


def load_neighbors(registered_dir: Path) -> dict[str, list[str]]:
    """Per-frame ranked neighbor names from neighbors.json."""
    payload = json.loads((registered_dir / "neighbors.json").read_text())
    neighbors = payload.get("neighbors", payload)
    return {
        name: entry.get("neighbors", []) for name, entry in neighbors.items()
    }


def build_face_filename_index(images_cubemap: Path) -> dict[str, str]:
    """Map registered frame name -> face image filename (extension included).

    ``poses.npz`` frame names match the ``dmaps/<name>.npy`` keys, which may
    or may not carry the image extension; the rendered faces keep the source
    panorama's filename. Resolved by scanning the face-0 folder.
    """
    index: dict[str, str] = {}
    face0 = images_cubemap / f"{FACE_PREFIX}0"
    for path in sorted(face0.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        index[path.name] = path.name
        index.setdefault(path.stem, path.name)
    return index


def _unproject_depth_samples(
    dmap: np.ndarray,
    c2w: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """World points from valid pixels of an equirect radial depth map."""
    height, width = dmap.shape
    valid_v, valid_u = np.nonzero(np.isfinite(dmap) & (dmap > 0))
    if valid_v.size == 0:
        return np.zeros((0, 3))
    if valid_v.size > num_samples:
        sel = rng.choice(valid_v.size, size=num_samples, replace=False)
        valid_v, valid_u = valid_v[sel], valid_u[sel]
    depth = dmap[valid_v, valid_u]
    u = (valid_u + 0.5) / width
    v = (valid_v + 0.5) / height
    yaw = (u * 2 - 1) * np.pi
    pitch = (1 - 2 * v) * np.pi / 2
    pts_cam = np.stack(
        [
            depth * np.sin(yaw) * np.cos(pitch),
            -depth * np.sin(pitch),
            depth * np.cos(yaw) * np.cos(pitch),
        ],
        axis=-1,
    )
    return pts_cam @ c2w[:3, :3].T + c2w[:3, 3]


def _assign_faces(
    world_pts: np.ndarray,
    c2w: np.ndarray,
    face_rotations: list[np.ndarray],
    margin: float = 1.02,
) -> np.ndarray:
    """(N, 6) bool: which faces of the pano at ``c2w`` see each world point.

    A point can fall on up to three faces near corners; the small
    ``margin`` on the |x/z|,|y/z| <= 1 test keeps seam overlaps matched.
    """
    dirs = (world_pts - c2w[:3, 3]) @ c2w[:3, :3]  # rows @ R_c2w = R_w2c @ cols
    hits = np.zeros((world_pts.shape[0], NUM_FACES), dtype=bool)
    for k, r_face in enumerate(face_rotations):
        v = dirs @ r_face.T
        with np.errstate(divide="ignore", invalid="ignore"):
            hits[:, k] = (
                (v[:, 2] > 0)
                & (np.abs(v[:, 0]) <= margin * v[:, 2])
                & (np.abs(v[:, 1]) <= margin * v[:, 2])
            )
    return hits


def generate_face_pairs(
    registered_dir: Path,
    images_cubemap: Path,
    pairs_path: Path,
    samples_per_frame: int = 400,
    min_covis: int = 8,
    seed: int = 0,
) -> int:
    """Write face-level match pairs derived from the registered geometry.

    For each pano pair in the neighbor graph, depth samples of both panos
    are assigned to the faces of both, and every face pair sharing at
    least ``min_covis`` co-observed samples is emitted. Same-pano pairs
    are never emitted.

    Returns the number of face pairs written.
    """
    names, c2w = load_registered_poses(registered_dir)
    neighbors = load_neighbors(registered_dir)
    name_to_idx = {name: i for i, name in enumerate(names)}
    face_files = build_face_filename_index(images_cubemap)
    face_rotations = get_cubemap_rotations()
    rng = np.random.default_rng(seed)
    dmaps_dir = registered_dir / "dmaps"

    world_samples: list[np.ndarray] = []
    for i, name in enumerate(names):
        dmap_path = dmaps_dir / f"{name}.npy"
        if not dmap_path.exists():
            world_samples.append(np.zeros((0, 3)))
            continue
        dmap = np.asarray(np.load(dmap_path))
        world_samples.append(
            _unproject_depth_samples(dmap, c2w[i], samples_per_frame, rng)
        )

    pano_pairs = set()
    for name, nbrs in neighbors.items():
        i = name_to_idx.get(name)
        if i is None:
            continue
        for nbr in nbrs:
            j = name_to_idx.get(nbr)
            if j is None or j == i:
                continue
            pano_pairs.add((min(i, j), max(i, j)))

    face_pair_lines = []
    skipped_missing = 0
    for i, j in sorted(pano_pairs):
        if names[i] not in face_files or names[j] not in face_files:
            skipped_missing += 1
            continue
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for pts in (world_samples[i], world_samples[j]):
            if pts.shape[0] == 0:
                continue
            hits_i = _assign_faces(pts, c2w[i], face_rotations)
            hits_j = _assign_faces(pts, c2w[j], face_rotations)
            # Co-observation counts per face pair, vectorized over samples.
            pair_counts = hits_i.astype(np.int64).T @ hits_j.astype(np.int64)
            for fi in range(NUM_FACES):
                for fj in range(NUM_FACES):
                    if pair_counts[fi, fj] > 0:
                        counts[(fi, fj)] += int(pair_counts[fi, fj])
        for (fi, fj), count in sorted(counts.items()):
            if count >= min_covis:
                face_pair_lines.append(
                    f"{FACE_PREFIX}{fi}/{face_files[names[i]]} "
                    f"{FACE_PREFIX}{fj}/{face_files[names[j]]}"
                )

    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(
        "\n".join(face_pair_lines) + "\n" if face_pair_lines else ""
    )
    logger.info(
        "Wrote %d face pairs from %d pano pairs to %s (%d pano pairs skipped: "
        "no rendered faces)",
        len(face_pair_lines),
        len(pano_pairs),
        pairs_path,
        skipped_missing,
    )
    return len(face_pair_lines)


def _prepare_colmap_masks(
    masks_cubemap: Path, images_cubemap: Path, out_dir: Path
) -> Path:
    """Expose ``masks_cubemap`` in COLMAP mask convention via symlinks.

    COLMAP expects ``<mask_path>/<image_subpath>.png`` — the full image
    filename with ``.png`` appended — whereas the rendered masks are named
    ``<stem>.png``. Masks are binary with 255 = valid, 0 = invalid, which
    matches COLMAP's non-zero-is-valid convention directly.
    """
    face_files = build_face_filename_index(images_cubemap)
    linked = 0
    for face_idx in range(NUM_FACES):
        src_dir = masks_cubemap / f"{FACE_PREFIX}{face_idx}"
        dst_dir = out_dir / f"{FACE_PREFIX}{face_idx}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.is_dir():
            continue
        for mask in src_dir.glob("*.png"):
            image_name = face_files.get(mask.stem) or face_files.get(mask.name)
            if image_name is None:
                continue
            link = dst_dir / f"{image_name}.png"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(mask.resolve())
            linked += 1
    logger.info("Linked %d COLMAP-convention masks under %s", linked, out_dir)
    return out_dir


def extract_and_match(
    database_path: Path,
    images_cubemap: Path,
    pairs_path: Path,
    masks_cubemap: Path | None = None,
    device: str = "cuda",
    remove_existing: bool = True,
) -> list[str]:
    """Masked SIFT extraction (fixed analytic K) + matching over pairs.txt.

    Returns the list of image names (relative paths) in the database.
    """
    import PIL.Image

    if device == "cpu":
        gpu_index, use_gpu = "-1", False
    elif device == "cuda":
        gpu_index, use_gpu = "0", True
    elif device.startswith("cuda:"):
        gpu_index, use_gpu = device.split(":", 1)[1], True
    else:
        raise ValueError(f"Unsupported device: {device!r}")

    images_list = sorted(
        str(p.relative_to(images_cubemap))
        for p in images_cubemap.glob(f"{FACE_PREFIX}*/*")
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not images_list:
        raise FileNotFoundError(f"No face images under {images_cubemap}")

    with PIL.Image.open(images_cubemap / images_list[0]) as sample:
        face_w, face_h = sample.size
    if face_w != face_h:
        raise ValueError(f"Cubemap faces must be square, got {face_w}x{face_h}")

    if database_path.exists() and remove_existing:
        database_path.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "SIMPLE_PINHOLE"
    # Analytic K of a 90x90 degree face: f = W/2, principal point at the
    # image center (exact by construction of the cubemap rendering).
    reader_opts.camera_params = f"{face_w / 2.0},{face_w / 2.0},{face_h / 2.0}"
    if masks_cubemap is not None and masks_cubemap.is_dir():
        colmap_mask_dir = database_path.parent / "sift_masks"
        reader_opts.mask_path = str(
            _prepare_colmap_masks(
                masks_cubemap, images_cubemap, colmap_mask_dir
            )
        )

    logger.info(
        "Extracting SIFT on %d faces (%dpx, f=%.1f) into %s",
        len(images_list),
        face_w,
        face_w / 2.0,
        database_path,
    )
    pycolmap.extract_features(
        str(database_path),
        str(images_cubemap),
        images_list,
        "PER_FOLDER",
        reader_opts,
        extraction_options=pycolmap.FeatureExtractionOptions(
            num_threads=16, gpu_index=gpu_index, use_gpu=use_gpu
        ),
    )
    apply_pano_rig_config(database_path)

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.gpu_index = gpu_index
    matching_options.use_gpu = use_gpu
    pairing_options = pycolmap.ImportedPairingOptions()
    pairing_options.match_list_path = str(pairs_path)
    pycolmap.match_image_pairs(
        database_path=str(database_path),
        matching_options=matching_options,
        pairing_options=pairing_options,
    )
    return images_list


def apply_pano_rig_config(database_path: Path) -> None:
    """Rewrite the database rigs/frames as one 6-face rig per panorama.

    ``extract_features`` creates a trivial single-camera rig per camera;
    ``pycolmap.triangulate_points`` cross-checks database rigs against the
    model's, so the database must carry the real cubemap rig. Uses the
    ``pano_camera{k}/`` image prefixes and the exactly-known face rotations
    (zero translation, face 0 = reference sensor).
    """
    face_rotations = get_cubemap_rotations()
    rig_cameras = []
    for face_idx in range(NUM_FACES):
        if face_idx == 0:
            cam_from_rig = None
        else:
            cam_from_rig = rigid(
                face_rotations[face_idx] @ face_rotations[0].T, np.zeros(3)
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=face_idx == 0,
                image_prefix=f"{FACE_PREFIX}{face_idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    database = pycolmap.Database.open(str(database_path))
    pycolmap.apply_rig_config(
        [pycolmap.RigConfig(cameras=rig_cameras)], database
    )


def build_posed_rig_reconstruction(
    database_path: Path,
    registered_dir: Path,
    out_dir: Path,
) -> pycolmap.Reconstruction:
    """Rig reconstruction posed from the registered PanoVGGT c2w poses.

    Mirrors the database's rig and frames exactly (ids included), so
    ``pycolmap.triangulate_points`` can consume the model directly: the
    rig is anchored at face 0 (identity rotation w.r.t. the pano frame),
    hence ``rig_from_world`` is simply the pano's world-to-camera pose.
    No 3D points are created here.
    """
    names, c2w = load_registered_poses(registered_dir)

    database = pycolmap.Database.open(str(database_path))
    db_images = {image.image_id: image for image in database.read_all_images()}
    db_cameras = {cam.camera_id: cam for cam in database.read_all_cameras()}
    db_rigs = database.read_all_rigs()
    db_frames = database.read_all_frames()
    if len(db_rigs) != 1:
        raise ValueError(
            f"Expected 1 rig in the database "
            f"(run apply_pano_rig_config first), "
            f"got {len(db_rigs)}"
        )

    recon = pycolmap.Reconstruction()
    for cam in db_cameras.values():
        recon.add_camera(
            pycolmap.Camera(
                camera_id=cam.camera_id,
                model=cam.model.name,
                width=cam.width,
                height=cam.height,
                params=list(cam.params),
            )
        )
    recon.add_rig(db_rigs[0])

    # World poses by pano stem; poses.npz names may be stems while the
    # rendered faces keep the extension.
    pose_by_name: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        pose_by_name[name] = c2w[i]
        pose_by_name.setdefault(Path(name).stem, c2w[i])

    num_missing = 0
    for db_frame in db_frames:
        image_ids = [
            data_id.id
            for data_id in db_frame.data_ids
            if data_id.sensor_id.type == pycolmap.SensorType.CAMERA
        ]
        if not image_ids:
            continue
        filename = db_images[image_ids[0]].name.partition("/")[2]
        pose = pose_by_name.get(filename, pose_by_name.get(Path(filename).stem))
        if pose is None:
            num_missing += 1
            continue
        frame = pycolmap.Frame(
            frame_id=db_frame.frame_id, rig_id=db_frame.rig_id
        )
        recon.add_frame(frame)
        for image_id in image_ids:
            db_image = db_images[image_id]
            image = pycolmap.Image(
                image_id=image_id,
                camera_id=db_image.camera_id,
                name=db_image.name,
                frame_id=db_frame.frame_id,
            )
            recon.frame(db_frame.frame_id).add_data_id(image.data_id)
            recon.add_image(image)
        r_pano_from_world = pose[:3, :3].T
        center = pose[:3, 3]
        recon.frame(db_frame.frame_id).rig_from_world = rigid(
            r_pano_from_world, -r_pano_from_world @ center
        )
        recon.register_frame(db_frame.frame_id)

    if num_missing:
        logger.warning(
            "%d database frames have no registered pose and were dropped",
            num_missing,
        )
    logger.info(
        "Posed rig reconstruction: %d frames, %d images, %d cameras",
        recon.num_frames(),
        recon.num_images(),
        len(recon.cameras),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    recon.write(str(out_dir))
    return recon


def run_sift_stage(
    registered_dir: Path,
    images_cubemap: Path,
    out_dir: Path,
    masks_cubemap: Path | None = None,
    samples_per_frame: int = 400,
    min_covis: int = 8,
    device: str = "cuda",
) -> pycolmap.Reconstruction:
    """Full stage: pairs -> masked SIFT + matching -> posed rig model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.txt"
    database_path = out_dir / "database.db"
    generate_face_pairs(
        registered_dir,
        images_cubemap,
        pairs_path,
        samples_per_frame=samples_per_frame,
        min_covis=min_covis,
    )
    extract_and_match(
        database_path,
        images_cubemap,
        pairs_path,
        masks_cubemap=masks_cubemap,
        device=device,
    )
    return build_posed_rig_reconstruction(
        database_path, registered_dir, out_dir / "sparse_init"
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--registered_dir", type=Path, required=True)
    parser.add_argument("--images_cubemap", type=Path, required=True)
    parser.add_argument("--masks_cubemap", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--samples_per_frame", type=int, default=400)
    parser.add_argument(
        "--min_covis",
        type=int,
        default=8,
        help="minimum co-observed depth samples for a face pair to be matched",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    run_sift_stage(
        args.registered_dir,
        args.images_cubemap,
        args.out_dir,
        masks_cubemap=args.masks_cubemap,
        samples_per_frame=args.samples_per_frame,
        min_covis=args.min_covis,
        device=args.device,
    )


if __name__ == "__main__":
    main()
