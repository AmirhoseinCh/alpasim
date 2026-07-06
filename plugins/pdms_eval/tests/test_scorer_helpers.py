# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the scorer's pure aggregate / filter math.

Importing ``alpasim_pdms.scorer`` does not require the nuplan stack (all
nuplan/eval imports there are lazy), and the vendored ``pdm_enums`` module only
depends on the stdlib ``enum`` — so these helpers are unit-testable directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from alpasim_pdms.scorer import (
    _extract_raw_metrics,
    _normalized_progress,
    _weighted_term,
)
from alpasim_pdms.vendored.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)

# WorldEngine weights: progress=5, ttc=5, comfort=2, lane_keeping=0, dir=0.
WEIGHTS = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
WEIGHTS[WeightedMetricIndex.PROGRESS] = 5.0
WEIGHTS[WeightedMetricIndex.TTC] = 5.0
WEIGHTS[WeightedMetricIndex.COMFORTABLE] = 2.0


class _FakeScorer:
    """Holds ``_multi_metrics`` / ``_weighted_metrics`` like a scored PDMScorer."""

    def __init__(self, multi: np.ndarray, weighted: np.ndarray) -> None:
        self._multi_metrics = multi
        self._weighted_metrics = weighted


def _make_scorer(nc, dac, progress, ttc, comfort, n_proposals=2, idx=1):
    multi = np.ones((len(MultiMetricIndex), n_proposals))
    weighted = np.zeros((len(WeightedMetricIndex), n_proposals))
    multi[MultiMetricIndex.NO_COLLISION, idx] = nc
    multi[MultiMetricIndex.DRIVABLE_AREA, idx] = dac
    weighted[WeightedMetricIndex.PROGRESS, idx] = progress
    weighted[WeightedMetricIndex.TTC, idx] = ttc
    weighted[WeightedMetricIndex.COMFORTABLE, idx] = comfort
    return _FakeScorer(multi, weighted)


class TestNormalizedProgress:
    def test_policy_matches_reference(self):
        progress_raw = np.array([20.0, 20.0])  # ref, policy
        assert _normalized_progress(progress_raw, idx=1) == pytest.approx(1.0)

    def test_policy_half_of_reference(self):
        progress_raw = np.array([20.0, 10.0])
        assert _normalized_progress(progress_raw, idx=1) == pytest.approx(0.5)

    def test_policy_exceeds_reference_clipped(self):
        progress_raw = np.array([10.0, 25.0])
        # denom = max(10, 25) = 25 -> 25/25 = 1.0
        assert _normalized_progress(progress_raw, idx=1) == pytest.approx(1.0)

    def test_no_meaningful_progress_returns_one(self):
        progress_raw = np.array([0.0, 0.0])
        assert _normalized_progress(progress_raw, idx=1) == pytest.approx(1.0)


class TestWeightedTerm:
    def test_all_ones_is_one(self):
        term = _weighted_term(
            progress_norm=1.0,
            ttc=1.0,
            comfort=1.0,
            lane_keeping=0.0,
            driving_direction=0.0,
            weights=WEIGHTS,
            index_cls=WeightedMetricIndex,
        )
        assert term == pytest.approx(1.0)

    def test_weighted_average(self):
        # progress=1, ttc=0, comfort=1 -> (5*1 + 5*0 + 2*1)/12 = 7/12
        term = _weighted_term(
            progress_norm=1.0,
            ttc=0.0,
            comfort=1.0,
            lane_keeping=0.0,
            driving_direction=0.0,
            weights=WEIGHTS,
            index_cls=WeightedMetricIndex,
        )
        assert term == pytest.approx(7.0 / 12.0)

    def test_zero_weights_returns_zero(self):
        term = _weighted_term(
            progress_norm=1.0,
            ttc=1.0,
            comfort=1.0,
            lane_keeping=1.0,
            driving_direction=1.0,
            weights=np.zeros(len(WeightedMetricIndex)),
            index_cls=WeightedMetricIndex,
        )
        assert term == 0.0


class TestExtractRawMetrics:
    def test_reads_expected_cells(self):
        scorer = _make_scorer(nc=0.0, dac=1.0, progress=0.9, ttc=1.0, comfort=0.5)
        raw = _extract_raw_metrics(scorer, 1, MultiMetricIndex, WeightedMetricIndex)
        assert raw["nc"] == 0.0
        assert raw["dac"] == 1.0
        assert raw["ttc"] == 1.0
        assert raw["comfort"] == 0.5


class TestHumanPenaltyFilterArithmetic:
    """The gate un-penalization + aggregate recomputation done in ``_score``."""

    def _aggregate(self, nc, dac, progress, ttc, comfort):
        term = _weighted_term(
            progress_norm=progress,
            ttc=ttc,
            comfort=comfort,
            lane_keeping=0.0,
            driving_direction=0.0,
            weights=WEIGHTS,
            index_cls=WeightedMetricIndex,
        )
        return nc * dac * term

    def test_policy_zeroed_by_dac_without_filter(self):
        score = self._aggregate(nc=1.0, dac=0.0, progress=1.0, ttc=1.0, comfort=1.0)
        assert score == 0.0

    def test_filter_restores_when_human_also_fails(self):
        # Human fails DAC too -> un-penalize policy DAC to 1.0.
        gt_dac = 0.0
        policy_dac = 0.0
        restored_dac = 1.0 if gt_dac == 0.0 else policy_dac
        score = self._aggregate(
            nc=1.0, dac=restored_dac, progress=1.0, ttc=1.0, comfort=1.0
        )
        assert score == pytest.approx(1.0)

    def test_filter_does_not_restore_when_human_passes(self):
        gt_dac = 1.0
        policy_dac = 0.0
        restored_dac = 1.0 if gt_dac == 0.0 else policy_dac
        score = self._aggregate(
            nc=1.0, dac=restored_dac, progress=1.0, ttc=1.0, comfort=1.0
        )
        assert score == 0.0
