#!/usr/bin/env python3
"""Generate binary validity masks from segmentation label images.

Reads class-id segmentation images from an input folder and writes one
binary mask per image to the output folder: pixels whose label is in
the given list become 0 (invalid), everything else 255 (valid) — the
mask convention used across the pano pipeline and COLMAP
(non-zero = valid).

A label file (default: ``mask_label.json`` inside the input folder)
maps label ids to class names. It is used to report which classes are
being masked and lets ``-l`` accept class names as well as ids.
Accepted JSON shapes: ``{"7": "sky", ...}``, ``{"sky": 7, ...}``, or
``{"labels": [{"id": 7, "name": "sky"}, ...]}``.

Example:
    python scripts/make_binary_masks.py \
        -i /data/site/segmentations -o /data/site/masks -l 7,8,9
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def load_label_names(label_file: Path) -> dict[int, str]:
    """Label id -> class name from mask_label.json (tolerant to shape)."""
    if not label_file.exists():
        logger.warning(
            "Label file %s not found; ids will be unnamed", label_file
        )
        return {}
    data = json.loads(label_file.read_text())
    if isinstance(data, dict) and isinstance(data.get("labels"), list):
        data = {entry["id"]: entry["name"] for entry in data["labels"]}
    names: dict[int, str] = {}
    for key, value in data.items():
        if isinstance(value, (int, float)) or str(value).isdigit():
            names[int(value)] = str(key)  # {"sky": 7}
        else:
            names[int(key)] = str(value)  # {"7": "sky"}
    return names


def parse_labels(tokens: str, names: dict[int, str]) -> list[int]:
    """Parse ``-l`` tokens: numeric ids, or class names from the label file."""
    name_to_id = {name: label for label, name in names.items()}
    labels = []
    for token in tokens.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lstrip("-").isdigit():
            labels.append(int(token))
        elif token in name_to_id:
            labels.append(name_to_id[token])
        else:
            raise SystemExit(
                f"Unknown label '{token}' (not an id, not a class name "
                f"in the label file; known: {sorted(name_to_id)})"
            )
    if not labels:
        raise SystemExit("No labels given")
    return sorted(set(labels))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True, help="segmentation folder"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="output mask folder"
    )
    parser.add_argument(
        "-l",
        "--labels",
        required=True,
        help="comma-separated label ids (or class names) to mask out, "
        "e.g. 7,8,9",
    )
    parser.add_argument(
        "-j",
        "--label_file",
        type=Path,
        default=None,
        help="label json (default: <input>/mask_label.json)",
    )
    args = parser.parse_args()

    label_file = (
        args.label_file
        if args.label_file is not None
        else args.input / "mask_label.json"
    )
    names = load_label_names(label_file)
    labels = parse_labels(args.labels, names)
    logger.info(
        "Masking out labels: %s",
        ", ".join(f"{label} ({names.get(label, '?')})" for label in labels),
    )

    seg_paths = sorted(
        p
        for p in args.input.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and p.name != label_file.name
    )
    if not seg_paths:
        raise SystemExit(f"No segmentation images in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    label_arr = np.array(labels)
    seen_labels: set[int] = set()
    for idx, path in enumerate(seg_paths):
        seg = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if seg is None:
            logger.warning("Skipping unreadable %s", path.name)
            continue
        seen_labels.update(np.unique(seg).tolist())
        mask = np.where(np.isin(seg, label_arr), 0, 255).astype(np.uint8)
        cv2.imwrite(str(args.output / f"{path.stem}.png"), mask)
        if (idx + 1) % 50 == 0 or idx + 1 == len(seg_paths):
            logger.info("%d/%d masks written", idx + 1, len(seg_paths))

    unused = [label for label in labels if label not in seen_labels]
    if unused:
        logger.warning(
            "Labels never seen in any segmentation: %s (present labels: %s)",
            unused,
            sorted(seen_labels),
        )
    logger.info("Done: %d masks in %s", len(seg_paths), args.output)


if __name__ == "__main__":
    main()
