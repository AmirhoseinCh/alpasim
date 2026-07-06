# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""``PDMSScorer`` — the ``alpasim.scorers`` entry point for PDM scoring.

The scorer computes NAVSIM-style PDM sub-scores and the aggregate PDMS for an
Alpasim rollout by driving the vendored WorldEngine ``PDMScorer`` over the
achieved (already-simulated) ego states, using a NAVSIM metric cache resolved by
scene id for the map-derived references (navtest path).

Design constraints honoured here:

* **Never breaks eval.**  ``ScorerGroup`` already catches per-scorer exceptions,
  but this scorer additionally degrades to an empty result (with a logged
  warning) whenever the nuplan/navsim stack, the metric cache or the map data
  are unavailable — so PDMS is purely additive to the built-in metrics.
* **Extends, never overrides.**  Sub-scores are emitted under a ``pdms/*``
  namespace alongside the built-in scene score; nothing built-in is replaced.
* **Reports provenance.**  The progress reference actually used (``metric_cache``
  vs ``gt``) is recorded so numbers are never silently mixed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from alpasim_pdms.adapter.pdm_inputs import (
    SCORE,
    SUBSCORE_COMFORT,
    SUBSCORE_DRIVABLE_AREA,
    SUBSCORE_DRIVING_DIRECTION,
    SUBSCORE_LANE_KEEPING,
    SUBSCORE_NO_COLLISION,
    SUBSCORE_PROGRESS,
    SUBSCORE_TTC,
    PDMDependencyError,
    PDMScoreInputs,
    build_inputs_from_cache,
)
from alpasim_pdms.config import PDMSConfig, ProgressReference
from alpasim_pdms.manifest import ManifestError, SceneManifest, load_metric_cache

if TYPE_CHECKING:  # pragma: no cover - typing only
    from eval.data import MetricReturn, SimulationResult
    from eval.schema import EvalConfig

logger = logging.getLogger("alpasim_pdms.scorer")

# Progress values below this (m) are treated as "no meaningful progress"
# (matches the vendored scorer's PROGRESS_DISTANCE_THRESHOLD).
_PROGRESS_DISTANCE_THRESHOLD = 0.1


class PDMSScorer:
    """Alpasim ``Scorer`` computing NAVSIM-style PDM scores for a rollout.

    This class intentionally does not import :class:`eval.scorers.base.Scorer`
    at module import time (to keep the plugin importable without ``alpasim_eval``
    for pure-adapter use); it is duck-typed to the ``Scorer`` interface
    (``__init__(cfg)`` + ``calculate(simulation_result) -> list[MetricReturn]``)
    and registered via the ``alpasim.scorers`` entry point.
    """

    def __init__(self, cfg: "EvalConfig") -> None:
        self.cfg = cfg
        self.config = PDMSConfig.from_eval_config(cfg)
        self._manifest: SceneManifest | None = None
        self._manifest_error: str | None = None
        if self.config.enabled:
            self._resolve_manifest()

    # -- setup ---------------------------------------------------------------
    def _resolve_manifest(self) -> None:
        try:
            self._manifest = SceneManifest.resolve(
                self.config.manifest_path, self.config.metric_cache_dir
            )
        except ManifestError as e:
            self._manifest_error = str(e)
            logger.warning("PDMS scorer disabled: %s", e)

    # -- Scorer interface ----------------------------------------------------
    def calculate(self, simulation_result: "SimulationResult") -> list["MetricReturn"]:
        """Compute PDM metric series for the rollout (empty on any soft failure)."""
        if not self.config.enabled:
            return []
        if self._manifest is None:
            logger.warning(
                "PDMS scorer has no manifest (%s); skipping scene.",
                self._manifest_error or "not configured",
            )
            return []

        scene_id = simulation_result.session_metadata.scene_id
        cache_path = self._manifest.get(scene_id)
        if cache_path is None:
            logger.warning("No metric cache for scene '%s'; skipping PDMS.", scene_id)
            return []

        try:
            metric_cache = load_metric_cache(str(cache_path))
            inputs = build_inputs_from_cache(
                simulation_result, metric_cache, self.config
            )
            subscores, reference = self._score(inputs, metric_cache)
        except PDMDependencyError as e:
            logger.warning("PDMS unavailable for scene '%s': %s", scene_id, e)
            return []
        except Exception as e:  # noqa: BLE001 - never break the eval run
            logger.exception("PDMS failed for scene '%s': %s", scene_id, e)
            return []

        return self._emit_metrics(simulation_result, subscores, reference, inputs)

    # -- scoring -------------------------------------------------------------
    def _score(
        self, inputs: PDMScoreInputs, metric_cache: Any
    ) -> tuple[dict[str, float], ProgressReference]:
        """Run the vendored scorer and read out (optionally filtered) sub-scores."""
        from alpasim_pdms.vendored.pdm_planner.scoring.pdm_scorer import (
            WEIGHTED_METRICS_WEIGHTS,
            PDMScorer,
        )
        from alpasim_pdms.vendored.pdm_planner.utils.pdm_enums import (
            MultiMetricIndex,
            WeightedMetricIndex,
        )

        scorer = PDMScorer(inputs.proposal_sampling)
        scorer.score_proposals(
            inputs.states,
            inputs.initial_ego_state,
            inputs.observation,
            inputs.centerline,
            inputs.route_lane_dict,
            inputs.drivable_area_map,
            inputs.map_api,
            batch=True,
        )

        idx = inputs.policy_index
        raw = _extract_raw_metrics(scorer, idx, MultiMetricIndex, WeightedMetricIndex)
        gates = {"nc": raw["nc"], "dac": raw["dac"]}

        if self.config.human_penalty_filter and inputs.gt_index is not None:
            gt_raw = _extract_raw_metrics(
                scorer, inputs.gt_index, MultiMetricIndex, WeightedMetricIndex
            )
            # Un-penalize the policy for binary gates the human also fails.
            if gt_raw["nc"] == 0.0:
                gates["nc"] = 1.0
            if gt_raw["dac"] == 0.0:
                gates["dac"] = 1.0

        # Normalized (ungated) ego progress relative to the reference (proposal 0).
        progress_norm = _normalized_progress(scorer._progress_raw, idx)

        weighted_term = _weighted_term(
            progress_norm=progress_norm,
            ttc=raw["ttc"],
            comfort=raw["comfort"],
            lane_keeping=raw["lane_keeping"],
            driving_direction=raw["driving_direction"],
            weights=WEIGHTED_METRICS_WEIGHTS,
            index_cls=WeightedMetricIndex,
        )
        aggregate = gates["nc"] * gates["dac"] * weighted_term

        subscores = {
            SUBSCORE_NO_COLLISION: gates["nc"],
            SUBSCORE_DRIVABLE_AREA: gates["dac"],
            SUBSCORE_PROGRESS: progress_norm,
            SUBSCORE_TTC: raw["ttc"],
            SUBSCORE_COMFORT: raw["comfort"],
            SUBSCORE_DRIVING_DIRECTION: raw["driving_direction"],
            SUBSCORE_LANE_KEEPING: raw["lane_keeping"],
            SCORE: aggregate,
        }

        reference = self.config.resolved_progress_reference(
            _has_reference(metric_cache)
        )
        first_violation = _first_violation_us(scorer, idx)
        if first_violation is not None:
            subscores["first_violation_timestamp"] = float(first_violation)
        return subscores, reference

    # -- emit ----------------------------------------------------------------
    def _emit_metrics(
        self,
        simulation_result: "SimulationResult",
        subscores: dict[str, float],
        reference: ProgressReference,
        inputs: PDMScoreInputs,
    ) -> list["MetricReturn"]:
        """Emit constant per-timestep series over the driven timestamps."""
        from eval.data import MetricReturn

        prefix = self.config.metric_prefix
        timestamps = _driven_timestamps(simulation_result)
        if len(timestamps) == 0:
            timestamps = [int(t) for t in simulation_result.timestamps_us]

        info = f"progress_reference={reference};anchor_gap_m={inputs.anchor_gap_m:.3f}"
        metrics: list[MetricReturn] = []
        for name, value in subscores.items():
            metric_name = f"{prefix}/{name}"
            # Gates/score aggregate as MIN over time (a violation anywhere sticks);
            # progress uses LAST; comfort/ttc use MIN. All series are constant in
            # full_rollout mode, so the choice only matters for windowed mode.
            agg = _aggregation_for(name)
            metrics.append(
                MetricReturn(
                    name=metric_name,
                    timestamps_us=list(timestamps),
                    values=[float(value)] * len(timestamps),
                    valid=[True] * len(timestamps),
                    time_aggregation=agg,
                    info=info,
                )
            )
        return metrics


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without nuplan).
# ---------------------------------------------------------------------------
def _extract_raw_metrics(
    scorer: Any, idx: int, multi_index_cls: Any, weighted_index_cls: Any
) -> dict[str, float]:
    """Read the raw per-proposal metric values out of a scored ``PDMScorer``."""
    multi = scorer._multi_metrics
    weighted = scorer._weighted_metrics
    return {
        "nc": float(multi[multi_index_cls.NO_COLLISION, idx]),
        "dac": float(multi[multi_index_cls.DRIVABLE_AREA, idx]),
        "ttc": float(weighted[weighted_index_cls.TTC, idx]),
        "comfort": float(weighted[weighted_index_cls.COMFORTABLE, idx]),
        "lane_keeping": float(weighted[weighted_index_cls.LANE_KEEPING, idx]),
        "driving_direction": float(weighted[weighted_index_cls.DRIVING_DIRECTION, idx]),
    }


def _normalized_progress(progress_raw: np.ndarray, idx: int, ref_idx: int = 0) -> float:
    """Ego progress normalized against the reference proposal (batch semantics).

    Reproduces the ungated branch of the vendored scorer's batch progress
    normalization: ``progress[idx] / max(progress[ref], progress[idx])``,
    clipped to ``[0, 1]``, and 1.0 when neither makes meaningful progress.
    """
    progress_raw = np.asarray(progress_raw, dtype=np.float64)
    ref = progress_raw[ref_idx]
    own = progress_raw[idx]
    denom = max(ref, own)
    if denom <= _PROGRESS_DISTANCE_THRESHOLD:
        return 1.0
    return float(np.clip(own / denom, 0.0, 1.0))


def _weighted_term(
    progress_norm: float,
    ttc: float,
    comfort: float,
    lane_keeping: float,
    driving_direction: float,
    weights: np.ndarray,
    index_cls: Any,
) -> float:
    """Weighted-average term of the PDMS aggregate, using the vendored weights."""
    contributions = {
        index_cls.PROGRESS: progress_norm,
        index_cls.TTC: ttc,
        index_cls.COMFORTABLE: comfort,
        index_cls.LANE_KEEPING: lane_keeping,
        index_cls.DRIVING_DIRECTION: driving_direction,
    }
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return 0.0
    numerator = sum(float(weights[i]) * v for i, v in contributions.items())
    return numerator / total_weight


def _first_violation_us(scorer: Any, idx: int) -> int | None:
    """Timestamp offset (µs from anchor) of the first at-fault collision, if any."""
    try:
        collision_time = scorer.time_to_at_fault_collision(idx)
    except Exception:  # noqa: BLE001 - optional diagnostic
        collision_time = float("inf")
    if not np.isfinite(collision_time):
        return None
    # inputs' grid started at the anchor; the returned time is an offset (s)
    # from the anchor, so express the first violation as a µs offset.
    return int(round(collision_time * 1_000_000))


def _has_reference(metric_cache: Any) -> bool:
    return any(
        hasattr(metric_cache, n) for n in ("trajectory", "pdm_closed_trajectory")
    )


def _aggregation_for(name: str) -> Any:
    """Pick a sensible time-aggregation for each sub-score series."""
    from eval.data import AggregationType

    if name in (SUBSCORE_PROGRESS,):
        return AggregationType.LAST
    if name == "first_violation_timestamp":
        return AggregationType.MIN
    # Gates, TTC, comfort, and the aggregate: a violation anywhere should stick.
    return AggregationType.MIN


def _driven_timestamps(simulation_result: "SimulationResult") -> list[int]:
    """Rollout timestamps at/after policy engagement (warmup masked out)."""
    first = simulation_result.first_driven_timestamp_us
    all_ts = [int(t) for t in simulation_result.timestamps_us]
    if first is None:
        return all_ts
    return [t for t in all_ts if t >= first]
