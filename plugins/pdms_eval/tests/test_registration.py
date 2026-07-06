# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for plugin registration, cheap import, and graceful degradation."""

from __future__ import annotations

import dataclasses
import importlib.metadata

import pytest


def test_package_imports_without_nuplan():
    """Importing the package must not require nuplan/navsim or eval."""
    import alpasim_pdms

    assert alpasim_pdms.__version__


def test_lazy_attribute_access():
    import alpasim_pdms

    # Accessing PDMSScorer/PDMSConfig triggers lazy import without nuplan.
    assert alpasim_pdms.PDMSConfig is not None
    assert alpasim_pdms.PDMSScorer is not None

    with pytest.raises(AttributeError):
        _ = alpasim_pdms.DoesNotExist


def test_scorer_entry_point_registered():
    """The ``pdms`` scorer entry point must resolve to ``PDMSScorer``."""
    try:
        eps = importlib.metadata.entry_points(group="alpasim.scorers")
    except TypeError:  # pragma: no cover - very old importlib API
        eps = importlib.metadata.entry_points().get("alpasim.scorers", [])
    names = {ep.name for ep in eps}
    if "pdms" not in names:
        pytest.skip("alpasim-pdms not installed into the environment")
    ep = next(ep for ep in eps if ep.name == "pdms")
    loaded = ep.load()
    from alpasim_pdms.scorer import PDMSScorer

    assert loaded is PDMSScorer


def test_tools_entry_point_registered():
    try:
        eps = importlib.metadata.entry_points(group="alpasim.tools")
    except TypeError:  # pragma: no cover
        eps = importlib.metadata.entry_points().get("alpasim.tools", [])
    names = {ep.name for ep in eps}
    if "pdms-manifest" not in names:
        pytest.skip("alpasim-pdms not installed into the environment")


@dataclasses.dataclass
class _FakeSessionMetadata:
    scene_id: str = "scene_x"
    session_uuid: str = "uuid"
    start_timestamp_us: int = 0
    control_timestep_us: int = 500_000


@dataclasses.dataclass
class _FakeSimResult:
    session_metadata: _FakeSessionMetadata

    @property
    def first_driven_timestamp_us(self):
        return None

    @property
    def timestamps_us(self):
        import numpy as np

        return np.array([0, 500_000, 1_000_000], dtype=np.uint64)


@dataclasses.dataclass
class _FakeEvalConfig:
    plugin_scorer_configs: dict
    enabled_plugin_scorers: list | None = None


def test_scorer_disabled_without_manifest_returns_empty():
    """With no manifest configured, calculate() degrades to an empty list."""
    from alpasim_pdms.scorer import PDMSScorer

    cfg = _FakeEvalConfig(plugin_scorer_configs={"pdms": {"enabled": True}})
    scorer = PDMSScorer(cfg)
    result = scorer.calculate(_FakeSimResult(_FakeSessionMetadata()))
    assert result == []


def test_scorer_disabled_flag_returns_empty():
    from alpasim_pdms.scorer import PDMSScorer

    cfg = _FakeEvalConfig(plugin_scorer_configs={"pdms": {"enabled": False}})
    scorer = PDMSScorer(cfg)
    assert scorer.calculate(_FakeSimResult(_FakeSessionMetadata())) == []


def test_scorer_missing_scene_returns_empty(tmp_path):
    """A configured manifest that lacks the scene degrades gracefully."""
    from alpasim_pdms.scorer import PDMSScorer

    # Build a by-token cache dir with a different token than the scene.
    (tmp_path / "metric_cache" / "other_token").mkdir(parents=True)
    (tmp_path / "metric_cache" / "other_token" / "metric_cache.pkl").write_bytes(b"x")

    cfg = _FakeEvalConfig(
        plugin_scorer_configs={
            "pdms": {
                "enabled": True,
                "metric_cache_dir": str(tmp_path / "metric_cache"),
            }
        }
    )
    scorer = PDMSScorer(cfg)
    result = scorer.calculate(_FakeSimResult(_FakeSessionMetadata(scene_id="scene_x")))
    assert result == []
