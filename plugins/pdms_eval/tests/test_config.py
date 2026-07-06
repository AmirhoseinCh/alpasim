# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for PDMS config resolution from an ``EvalConfig``-like object."""

from __future__ import annotations

import dataclasses

import pytest
from alpasim_pdms.config import (
    EgoTrajectorySource,
    PDMSConfig,
    ProgressReference,
    ScoringMode,
)


@dataclasses.dataclass
class _FakeEvalConfig:
    """Duck-types the two ``EvalConfig`` fields PDMSConfig reads."""

    plugin_scorer_configs: dict
    enabled_plugin_scorers: list | None = None


def test_defaults_when_absent():
    cfg = PDMSConfig.from_eval_config(_FakeEvalConfig(plugin_scorer_configs={}))
    assert cfg.enabled is True
    assert cfg.progress_reference == ProgressReference.METRIC_CACHE
    assert cfg.ego_trajectory_source == EgoTrajectorySource.DRIVEN
    assert cfg.mode == ScoringMode.FULL_ROLLOUT
    assert cfg.proposal_sampling_num_poses == 40


def test_reads_pdms_sub_mapping():
    raw = {
        "pdms": {
            "metric_cache_dir": "/data/cache",
            "progress_reference": "gt",
            "human_penalty_filter": False,
            "anchor_tolerance_m": 2.5,
        }
    }
    cfg = PDMSConfig.from_eval_config(_FakeEvalConfig(plugin_scorer_configs=raw))
    assert cfg.metric_cache_dir == "/data/cache"
    assert cfg.progress_reference == ProgressReference.GT
    assert cfg.human_penalty_filter is False
    assert cfg.anchor_tolerance_m == 2.5


def test_unknown_key_rejected():
    raw = {"pdms": {"not_a_real_key": 1}}
    with pytest.raises(Exception):
        PDMSConfig.from_eval_config(_FakeEvalConfig(plugin_scorer_configs=raw))


def test_missing_container_is_ok():
    class _Empty:
        pass

    cfg = PDMSConfig.from_eval_config(_Empty())
    assert isinstance(cfg, PDMSConfig)
    assert cfg.enabled is True


def test_resolved_progress_reference_falls_back_to_gt():
    cfg = PDMSConfig(progress_reference=ProgressReference.METRIC_CACHE)
    assert cfg.resolved_progress_reference(has_cache=False) == ProgressReference.GT
    assert (
        cfg.resolved_progress_reference(has_cache=True)
        == ProgressReference.METRIC_CACHE
    )

    gt_cfg = PDMSConfig(progress_reference=ProgressReference.GT)
    assert gt_cfg.resolved_progress_reference(has_cache=True) == ProgressReference.GT
