# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for consumption of ``alpasim.scorers`` plugin scorers by eval.

These tests cover the Phase 2 upstream patch that makes
``create_scorer_group`` append plugin-registered scorers to the built-in
scorers, gated by ``EvalConfig.enabled_plugin_scorers``.
"""

from __future__ import annotations

import pytest
from conftest import create_test_eval_config

from eval.data import AggregationType, MetricReturn, SimulationResult
from eval.scorers import SCORERS, _create_plugin_scorers, create_scorer_group
from eval.scorers.base import Scorer


class _DummyPluginScorer(Scorer):
    """Minimal plugin-style scorer used only for tests."""

    name = "dummy"

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        return [
            MetricReturn(
                name="dummy/score",
                values=[1.0],
                valid=[True],
                timestamps_us=[0],
                time_aggregation=AggregationType.LAST,
            )
        ]


class _AnotherPluginScorer(_DummyPluginScorer):
    name = "another"


class _FakeRegistry:
    """Stand-in for ``alpasim_plugins.scorers`` (a ``PluginRegistry``)."""

    def __init__(self, mapping: dict[str, type[Scorer]]):
        self._mapping = mapping

    def get_names(self) -> list[str]:
        return sorted(self._mapping)

    def get(self, name: str) -> type[Scorer]:
        return self._mapping[name]


@pytest.fixture
def patch_registry(monkeypatch):
    """Patch the plugin registry with a controllable fake."""

    def _apply(mapping: dict[str, type[Scorer]]):
        monkeypatch.setattr("alpasim_plugins.scorers", _FakeRegistry(mapping))

    return _apply


def test_builtins_present_without_plugins(patch_registry):
    """With no plugin scorers registered, only built-ins are created."""
    patch_registry({})
    cfg = create_test_eval_config()
    group = create_scorer_group(cfg)
    assert len(group.scorers) == len(SCORERS)


def test_plugin_scorer_appended_after_builtins(patch_registry):
    """A registered plugin scorer is appended after the built-in scorers."""
    patch_registry({"dummy": _DummyPluginScorer})
    cfg = create_test_eval_config()
    group = create_scorer_group(cfg)
    assert len(group.scorers) == len(SCORERS) + 1
    assert isinstance(group.scorers[-1], _DummyPluginScorer)


def test_enabled_plugin_scorers_none_runs_all(patch_registry):
    """``None`` (default) enables every registered plugin scorer."""
    patch_registry({"dummy": _DummyPluginScorer, "another": _AnotherPluginScorer})
    cfg = create_test_eval_config()
    cfg.enabled_plugin_scorers = None
    plugins = _create_plugin_scorers(cfg)
    assert {type(s) for s in plugins} == {_DummyPluginScorer, _AnotherPluginScorer}


def test_enabled_plugin_scorers_empty_disables_all(patch_registry):
    """An empty list disables all plugin scorers."""
    patch_registry({"dummy": _DummyPluginScorer})
    cfg = create_test_eval_config()
    cfg.enabled_plugin_scorers = []
    assert _create_plugin_scorers(cfg) == []


def test_enabled_plugin_scorers_selection(patch_registry):
    """Only the named plugin scorers are instantiated."""
    patch_registry({"dummy": _DummyPluginScorer, "another": _AnotherPluginScorer})
    cfg = create_test_eval_config()
    cfg.enabled_plugin_scorers = ["dummy"]
    plugins = _create_plugin_scorers(cfg)
    assert [type(s) for s in plugins] == [_DummyPluginScorer]


def test_unknown_plugin_scorer_is_skipped(patch_registry):
    """Requesting an unregistered scorer is skipped, not fatal."""
    patch_registry({"dummy": _DummyPluginScorer})
    cfg = create_test_eval_config()
    cfg.enabled_plugin_scorers = ["does_not_exist"]
    assert _create_plugin_scorers(cfg) == []


def test_registry_failure_degrades_gracefully(monkeypatch):
    """If plugin discovery raises, eval still returns the built-in scorers."""

    class _BrokenRegistry:
        def get_names(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("alpasim_plugins.scorers", _BrokenRegistry())
    cfg = create_test_eval_config()
    group = create_scorer_group(cfg)
    assert len(group.scorers) == len(SCORERS)


def test_construction_error_is_isolated(patch_registry):
    """A plugin scorer that fails to construct does not break the group."""

    class _ExplodingScorer(Scorer):
        def __init__(self, cfg):
            raise ValueError("cannot build")

        def calculate(self, simulation_result):
            return []

    patch_registry({"boom": _ExplodingScorer, "dummy": _DummyPluginScorer})
    cfg = create_test_eval_config()
    plugins = _create_plugin_scorers(cfg)
    assert [type(s) for s in plugins] == [_DummyPluginScorer]
