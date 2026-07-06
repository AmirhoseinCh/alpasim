# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for scene_id -> metric-cache manifest resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from alpasim_pdms.manifest import ManifestError, SceneManifest


def _make_cache_dir(root: Path, tokens: list[str]) -> Path:
    cache_dir = root / "metric_cache"
    for token in tokens:
        token_dir = cache_dir / token
        token_dir.mkdir(parents=True)
        (token_dir / "metric_cache.pkl").write_bytes(b"not-a-real-pickle")
    return cache_dir


def test_from_cache_dir_discovers_tokens(tmp_path):
    cache_dir = _make_cache_dir(tmp_path, ["tok_a", "tok_b"])
    manifest = SceneManifest.from_cache_dir(cache_dir)
    assert len(manifest) == 2
    assert "tok_a" in manifest
    assert manifest.get("tok_b").name == "metric_cache.pkl"


def test_from_cache_dir_missing_raises(tmp_path):
    with pytest.raises(ManifestError):
        SceneManifest.from_cache_dir(tmp_path / "nope")


def test_from_json_mapping(tmp_path):
    cache_dir = _make_cache_dir(tmp_path, ["tok_a"])
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps({"scene_1": str(cache_dir / "tok_a" / "metric_cache.pkl")})
    )
    manifest = SceneManifest.from_json(manifest_file)
    assert "scene_1" in manifest
    assert manifest.get("scene_1").is_file()


def test_from_json_list_of_records(tmp_path):
    cache_dir = _make_cache_dir(tmp_path, ["tok_a"])
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps(
            [
                {
                    "scene_id": "scene_1",
                    "metric_cache": "metric_cache/tok_a/metric_cache.pkl",
                }
            ]
        )
    )
    manifest = SceneManifest.from_json(manifest_file)
    # Relative paths resolve against the manifest's directory.
    assert manifest.get("scene_1") == (cache_dir / "tok_a" / "metric_cache.pkl")


def test_resolve_prefers_manifest_over_dir(tmp_path):
    cache_dir = _make_cache_dir(tmp_path, ["scene_1"])
    override = tmp_path / "override" / "custom.pkl"
    override.parent.mkdir()
    override.write_bytes(b"x")
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"scene_1": str(override)}))

    manifest = SceneManifest.resolve(manifest_file, cache_dir)
    assert manifest.get("scene_1") == override


def test_resolve_requires_some_source():
    with pytest.raises(ManifestError):
        SceneManifest.resolve(None, None)


def test_missing_scene_returns_none(tmp_path):
    cache_dir = _make_cache_dir(tmp_path, ["tok_a"])
    manifest = SceneManifest.from_cache_dir(cache_dir)
    assert manifest.get("not_there") is None
