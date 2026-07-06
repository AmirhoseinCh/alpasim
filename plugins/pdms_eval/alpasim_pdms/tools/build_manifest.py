# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""``alpasim-pdms-manifest`` — build a scene_id -> NAVSIM metric-cache manifest.

For the navtest path this is a thin convenience over the by-token cache layout:
it scans ``<metric_cache_dir>/<token>/metric_cache.pkl`` and writes a JSON
manifest mapping each token (== Alpasim ``scene_id``) to its cache file.  The
scorer can also consume the by-token directory directly (``pdms.metric_cache_dir``);
this tool exists to (a) materialise an explicit, reviewable mapping, (b) restrict
/ rename scenes, and (c) optionally emit a frame-alignment debug summary.

Usage::

    alpasim-pdms-manifest \
        --metric-cache-dir /data/navtest/metric_cache \
        --output /data/navtest/pdms_manifest.json

Registered under both the ``alpasim.tools`` entry-point group (as
``pdms-manifest``) and as the ``alpasim-pdms-manifest`` console script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from alpasim_pdms.manifest import SceneManifest

logger = logging.getLogger("alpasim_pdms.build_manifest")


def build_manifest(
    metric_cache_dir: str | None,
    manifest_path: str | None,
    scene_ids: list[str] | None = None,
) -> dict[str, str]:
    """Return a ``{scene_id: cache_path}`` mapping (paths as absolute strings)."""
    manifest = SceneManifest.resolve(manifest_path, metric_cache_dir)
    entries = {sid: str(path) for sid, path in manifest.entries.items()}
    if scene_ids is not None:
        wanted = set(scene_ids)
        missing = sorted(wanted - set(entries))
        if missing:
            logger.warning(
                "No metric cache found for %d scenes: %s", len(missing), missing
            )
        entries = {sid: p for sid, p in entries.items() if sid in wanted}
    return dict(sorted(entries.items()))


def _write_manifest(entries: dict[str, str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %d entries to %s", len(entries), output)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric-cache-dir",
        type=str,
        default=None,
        help="Directory laid out as <dir>/<token>/metric_cache.pkl.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Existing scene_id -> cache JSON to normalise/filter (optional).",
    )
    parser.add_argument(
        "--scene-ids",
        type=str,
        default=None,
        help="Optional path to a newline- or comma-separated list of scene ids "
        "to restrict the manifest to.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON manifest path.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def _load_scene_ids(path_str: str | None) -> list[str] | None:
    if not path_str:
        return None
    text = Path(path_str).read_text(encoding="utf-8")
    tokens = [t.strip() for line in text.splitlines() for t in line.split(",")]
    return [t for t in tokens if t]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.metric_cache_dir and not args.manifest_path:
        logger.error("Provide --metric-cache-dir and/or --manifest-path.")
        return 2
    scene_ids = _load_scene_ids(args.scene_ids)
    entries = build_manifest(args.metric_cache_dir, args.manifest_path, scene_ids)
    if not entries:
        logger.error("No manifest entries produced; nothing written.")
        return 1
    _write_manifest(entries, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
