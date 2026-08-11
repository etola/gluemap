#!/usr/bin/env python3
"""Fetch a session's floor-plan overlay for the floorplan BA stage.

Mirrors the production lookup chain (photogrammetry
``threedn_service/utils/web/overlay.py`` / ``plan.py``) with plain
requests, so it runs outside the service environment:

  1. GET  /api/v1/plan/{session_id}                 -> plan id
  2. GET  /api/v1/overlays/base_floor_plan/{plan}   -> overlay
     (fallback: GET /api/v1/overlays?plan_id=...,
      latest entry that carries a signed_url)
  3. GET  /api/v1/levels/get_level_scale_factor/{level}  (warn only:
     a scale factor != 1 means the overlay georeference was drawn on
     an unscaled map -- the one error the SE2 plan BA cannot absorb)
  4. download overlay.signed_url                    -> plan.png

Auth is the pipeline user's basic auth (``DroneDeploy:$DD_PIPELINE_KEY``),
identical to ``drone_common.dd_api_requests``.

Outputs (to --out_dir):
  plan.png       the overlay image
  anchors.json   {"corners_lla": [[lon, lat] x4]} in UL, UR, LR, LL
                 order (the overlay's polygon ring order, matching
                 production georeference_image) + provenance fields

Usage:
  DD_PIPELINE_KEY=... pano_fetch_floorplan.py \\
      --session a946b2a676_GROUNDAPP --out_dir plan/
"""

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PREFIX_OF_ENV = {
    "prod": "https://www.dronedeploy.com",
    "test": "https://test.dronedeploy.com",
    "stage": "https://stage.dronedeploy.com",
}


def _get(session, prefix: str, endpoint: str, **kwargs):
    """GET an api/v1 endpoint; returns parsed JSON or None on any error."""
    url = f"{prefix}/api/v1/{endpoint}"
    try:
        response = session.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning("GET %s failed: %s", url, exc)
        return None


def fetch(
    session_id: str,
    out_dir: Path,
    env: str = "prod",
    key: str | None = None,
) -> dict:
    import requests

    key = key or os.environ.get("DD_PIPELINE_KEY")
    if not key:
        raise SystemExit(
            "DD_PIPELINE_KEY is not set (the pipeline user's access key, "
            "the same credential drone_common.dd_api_requests uses)"
        )
    prefix = PREFIX_OF_ENV[env]
    http = requests.Session()
    http.auth = ("DroneDeploy", key)

    plan = _get(http, prefix, f"plan/{session_id}")
    if plan is None or "id" not in plan:
        raise SystemExit(f"Could not resolve plan for session {session_id}")
    plan_id = plan["id"]
    logger.info("session %s -> plan %s", session_id, plan_id)

    overlay = _get(http, prefix, f"overlays/base_floor_plan/{plan_id}")
    if not overlay or not overlay.get("signed_url"):
        items = _get(http, prefix, "overlays", params={"plan_id": plan_id})
        overlay = next(
            (o for o in reversed(items or []) if o.get("signed_url")), None
        )
    if not overlay:
        raise SystemExit(f"No floor-plan overlay found for plan {plan_id}")

    ring = overlay["geometry"]["coordinates"][0]
    if len(ring) != 5 or ring[0] != ring[-1]:
        raise SystemExit(f"Overlay polygon is not a closed quad: {ring}")
    corners = [[float(lon), float(lat)] for lon, lat in ring[:4]]

    scale_factor = None
    group_ids = overlay.get("group_ids") or []
    if group_ids:
        data = _get(
            http, prefix, f"levels/get_level_scale_factor/{group_ids[0]}"
        )
        scale_factor = (data or {}).get("scale_factor")
        if scale_factor not in (None, 1.0):
            logger.warning(
                "level scale factor is %s (!= 1): the overlay was "
                "georeferenced on an unscaled map; the plan corners must "
                "be scaled the way production LevelStage does before the "
                "SE2 BA can use them",
                scale_factor,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_png = out_dir / "plan.png"
    with http.get(overlay["signed_url"], stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(plan_png, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logger.info("downloaded plan image -> %s", plan_png)

    anchors = {
        "corners_lla": corners,
        "session_id": session_id,
        "plan_id": plan_id,
        "overlay_id": overlay.get("id"),
        "level_id": group_ids[0] if group_ids else None,
        "level_scale_factor": scale_factor,
    }
    anchors_path = out_dir / "anchors.json"
    anchors_path.write_text(json.dumps(anchors, indent=2))
    logger.info("wrote %s", anchors_path)
    return anchors


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session",
        required=True,
        help="session id (the debug-prefix component, "
        "e.g. a946b2a676_GROUNDAPP)",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--env", choices=sorted(PREFIX_OF_ENV), default="prod")
    parser.add_argument(
        "--key",
        default=None,
        help="pipeline access key (default: $DD_PIPELINE_KEY)",
    )
    args = parser.parse_args()
    fetch(args.session, args.out_dir, env=args.env, key=args.key)


if __name__ == "__main__":
    main()
