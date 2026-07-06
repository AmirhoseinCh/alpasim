# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Resolve Alpasim scene ids to NAVSIM metric caches (navtest path).

For the ``navtest`` split, NAVSIM metric caches can be precomputed offline
(``run_metric_caching.py``).  Each cache is pinned to one token and ships the
PDM-closed trajectory, route centerline, ``route_lane_ids`` and a serialized
drivable-area map — everything the scorer needs *except* the policy's actual
states — so no live nuPlan map API or map-frame transform derivation is required
at scoring time.

This module only handles *resolution* (scene_id -> cache path) and lazy loading;
the nuPlan/NAVSIM deserialization types are imported lazily so importing the
plugin never requires the nuplan stack.

Resolution order:

1. An explicit ``manifest_path`` JSON: ``{scene_id: cache_file_or_dir}`` (or a
   list of ``{"scene_id": ..., "metric_cache": ...}`` records).
2. A ``metric_cache_dir`` laid out by token, i.e.
   ``<metric_cache_dir>/<token>/metric_cache.pkl`` (navsim's default), with the
   Alpasim ``scene_id`` used directly as the token.
"""

from __future__ import annotations

import dataclasses
import functools
import gzip
import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger("alpasim_pdms.manifest")

#: Filenames navsim uses for a per-token metric cache.
_CACHE_BASENAMES = ("metric_cache.pkl", "metric_cache.gz", "metric_cache")


class ManifestError(RuntimeError):
    """Raised when a scene id cannot be resolved to a metric cache."""


@dataclasses.dataclass
class SceneManifest:
    """A ``scene_id -> metric-cache path`` mapping for navtest scoring."""

    entries: dict[str, Path]

    def __contains__(self, scene_id: str) -> bool:
        return scene_id in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, scene_id: str) -> Path | None:
        return self.entries.get(scene_id)

    @classmethod
    def from_json(cls, manifest_path: str | Path) -> "SceneManifest":
        """Load a manifest from a JSON file (mapping or list-of-records)."""
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise ManifestError(f"Manifest file not found: {manifest_path}")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries: dict[str, Path] = {}
        base = manifest_path.parent
        if isinstance(raw, dict):
            items = raw.items()
        elif isinstance(raw, list):
            items = [
                (rec["scene_id"], rec.get("metric_cache") or rec.get("path"))
                for rec in raw
            ]
        else:  # pragma: no cover - defensive
            raise ManifestError(f"Unsupported manifest structure in {manifest_path}")
        for scene_id, path_str in items:
            if not path_str:
                continue
            path = Path(path_str)
            if not path.is_absolute():
                path = (base / path).resolve()
            entries[str(scene_id)] = path
        logger.info("Loaded %d manifest entries from %s", len(entries), manifest_path)
        return cls(entries=entries)

    @classmethod
    def from_cache_dir(cls, metric_cache_dir: str | Path) -> "SceneManifest":
        """Discover ``<dir>/<token>/metric_cache.pkl`` caches under a directory."""
        metric_cache_dir = Path(metric_cache_dir)
        if not metric_cache_dir.is_dir():
            raise ManifestError(f"metric_cache_dir does not exist: {metric_cache_dir}")
        entries: dict[str, Path] = {}
        for token_dir in sorted(p for p in metric_cache_dir.iterdir() if p.is_dir()):
            cache_file = _find_cache_file(token_dir)
            if cache_file is not None:
                entries[token_dir.name] = cache_file
        logger.info(
            "Discovered %d token caches under %s", len(entries), metric_cache_dir
        )
        return cls(entries=entries)

    @classmethod
    def resolve(
        cls,
        manifest_path: str | Path | None,
        metric_cache_dir: str | Path | None,
    ) -> "SceneManifest":
        """Build a manifest from an explicit file and/or a by-token directory.

        When both are given, ``manifest_path`` entries take precedence and the
        directory fills in any gaps.
        """
        merged: dict[str, Path] = {}
        if metric_cache_dir is not None:
            merged.update(cls.from_cache_dir(metric_cache_dir).entries)
        if manifest_path is not None:
            merged.update(cls.from_json(manifest_path).entries)
        if not merged:
            raise ManifestError(
                "No metric caches resolved: set pdms.manifest_path and/or "
                "pdms.metric_cache_dir."
            )
        return cls(entries=merged)


def _find_cache_file(token_dir: Path) -> Path | None:
    for name in _CACHE_BASENAMES:
        candidate = token_dir / name
        if candidate.is_file():
            return candidate
    # Fall back to any single pickle in the directory.
    pickles = sorted(token_dir.glob("*.pkl")) + sorted(token_dir.glob("*.gz"))
    return pickles[0] if pickles else None


def _load_pickle_maybe_gzip(path: Path) -> Any:
    """Load a pickle that may be raw or gzip / xz(lzma) / zstd compressed.

    nuPlan-devkit ``run_metric_caching`` writes the per-token ``metric_cache.pkl``
    as an **LZMA/xz**-compressed pickle, so gzip-only sniffing is insufficient.
    We detect the container by magic bytes and decompress accordingly.
    """
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":  # gzip
        data = gzip.decompress(raw)
    elif raw[:6] == b"\xfd7zXZ\x00" or raw[:1] == b"\xfd":  # xz / lzma
        import lzma

        data = lzma.decompress(raw)
    elif raw[:4] == b"\x28\xb5\x2f\xfd":  # zstd
        import zstandard  # type: ignore[import-untyped]

        data = zstandard.ZstdDecompressor().decompress(raw)
    else:  # assume raw pickle
        data = raw
    return pickle.loads(data)


@functools.lru_cache(maxsize=256)
def load_metric_cache(cache_path_str: str) -> Any:
    """Load and cache a NAVSIM ``MetricCache`` from disk.

    Tries NAVSIM's own loader first (handles the current on-disk format and
    class layout), then falls back to a plain pickle/gzip load.  Requires the
    nuplan/navsim stack because the pickled objects reference their classes.

    Cached by path so repeated rollouts of the same scene deserialize once.
    """
    cache_path = Path(cache_path_str)
    if not cache_path.exists():
        raise ManifestError(f"Metric cache not found: {cache_path}")
    try:
        from navsim.planning.metric_caching.metric_cache import (  # type: ignore
            MetricCache,
        )

        if hasattr(MetricCache, "load"):
            return MetricCache.load(cache_path)
    except ImportError:
        logger.debug("navsim MetricCache not importable; using raw pickle load.")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("MetricCache.load failed (%s); using raw pickle load.", e)
    return _load_pickle_maybe_gzip(cache_path)
