# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Configuration for the PDMS eval plugin.

The scorer only receives the core :class:`eval.schema.EvalConfig`.  Following
the plugin guideline that plugin scorers must not add fields to the core
``ScorersConfig``, PDMS reads its own parameters from the generic
``EvalConfig.plugin_scorer_configs["pdms"]`` sub-mapping (populated by the
``eval/pdms.yaml`` Hydra overlay shipped with this plugin).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from omegaconf import OmegaConf

#: The key under ``EvalConfig.plugin_scorer_configs`` that holds this config,
#: and the plugin's registered ``alpasim.scorers`` entry-point name.
PLUGIN_NAME = "pdms"


class ProgressReference(StrEnum):
    """Which trajectory defines the ego-progress (EP) denominator.

    * ``metric_cache``: the NAVSIM PDM-closed planner trajectory shipped in the
      metric cache (canonical NAVSIM EP semantics). Falls back to ``gt`` when
      no token-aligned cache is available.
    * ``gt``: the recorded ground-truth (expert) trajectory, matching
      WorldEngine's expert-referenced progress.
    """

    METRIC_CACHE = "metric_cache"
    GT = "gt"


class EgoTrajectorySource(StrEnum):
    """Which ego trajectory is scored."""

    #: The achieved (true ``local``-frame) rollout trajectory.
    DRIVEN = "driven"
    #: The driver's (noised) estimated trajectory.
    DRIVER_ESTIMATED = "driver_estimated"


class ScoringMode(StrEnum):
    """How the driven segment is turned into PDM evaluation window(s)."""

    #: A single PDM evaluation over the whole driven segment.
    FULL_ROLLOUT = "full_rollout"
    #: Sliding fixed-horizon windows, sub-scores emitted per window.
    WINDOWED = "windowed"


class MetricSet(StrEnum):
    """Which aggregate metric(s) to emit."""

    #: NAVSIM v2 extended PDMS (adds TLC gate, lane keeping, two-frame comfort).
    EPDMS = "epdms"
    #: Classic 5-term PDMS.
    PDMS = "pdms"
    #: Emit both aggregates from the same sub-scores.
    BOTH = "both"


@dataclass
class PDMSConfig:
    """Parameters for :class:`alpasim_pdms.scorer.PDMSScorer`."""

    # Master switch (independent of ``EvalConfig.enabled_plugin_scorers``, which
    # controls *whether the scorer is constructed at all*).  Handy for keeping
    # the scorer registered while disabling its computation from a config.
    enabled: bool = True

    # -- Scene -> NAVSIM metric-cache resolution (navtest path) ---------------
    # Directory containing per-token NAVSIM metric caches
    # (``<token>/metric_cache.pkl``), as produced by navsim
    # ``run_metric_caching.py``.
    metric_cache_dir: str | None = None
    # Optional explicit ``scene_id -> cache file`` manifest (JSON), overriding
    # the by-token lookup in ``metric_cache_dir``.
    manifest_path: str | None = None

    # -- Scoring references / sources ----------------------------------------
    # These are stored as plain strings (validated against the corresponding
    # StrEnum in ``__post_init__``) so that lowercase YAML values like
    # ``progress_reference: gt`` work: OmegaConf validates enum fields by member
    # *name* (``GT``), not value (``gt``), so a str field + value-coercion keeps
    # the config ergonomic while still rejecting invalid values.
    progress_reference: str = ProgressReference.METRIC_CACHE.value
    ego_trajectory_source: str = EgoTrajectorySource.DRIVEN.value
    mode: str = ScoringMode.FULL_ROLLOUT.value
    metric: str = MetricSet.EPDMS.value
    # Score the GT rollout as a side proposal and un-penalize the policy for any
    # binary metric the human also fails (NAVSIM ``human_penalty_filter``).
    # Robust against MTGS/navtest map annotation noise; default on.
    human_penalty_filter: bool = True

    # -- Sampling / windowing -------------------------------------------------
    # PDM proposal sampling (NAVSIM default: 4 s horizon @ 10 Hz).
    proposal_sampling_num_poses: int = 40
    proposal_sampling_interval_s: float = 0.1
    # Grid the rollout is resampled onto before scoring.
    resample_hz: float = 10.0
    # Windowed mode: horizon of each window in seconds.
    window_s: float = 4.0

    # -- Anchor / map ---------------------------------------------------------
    # Max distance between the rollout ego at engagement and the cache's anchor
    # ego state; a larger gap indicates a manifest/frame-transform bug.
    anchor_tolerance_m: float = 1.0
    # nuPlan maps root, used only for the non-cache fallback path.
    nuplan_map_root: str | None = None
    map_radius_m: float = 100.0
    # Directory of nuplan-track per-scene configs (``<scene_id>.yaml`` carrying a
    # ``city`` field).  Used to resolve the nuPlan map location (city) for the
    # TTC test when the metric cache does not carry one (navtest path).
    nuplan_configs_root: str | None = None
    # Explicit nuPlan map location (city) override; takes precedence over the
    # per-scene config lookup.  navtest spans four cities, so per-scene lookup
    # is preferred; this is mainly for single-city runs / debugging.
    map_location: str | None = None

    # -- Output ---------------------------------------------------------------
    # Prefix for emitted metric names, e.g. ``pdms/score``.
    metric_prefix: str = "pdms"

    def __post_init__(self) -> None:
        # Validate + normalize the string-typed enum fields by *value*.
        self.progress_reference = ProgressReference(self.progress_reference).value
        self.ego_trajectory_source = EgoTrajectorySource(
            self.ego_trajectory_source
        ).value
        self.mode = ScoringMode(self.mode).value
        self.metric = MetricSet(self.metric).value

    def resolved_progress_reference(self, has_cache: bool) -> ProgressReference:
        """``metric_cache`` degrades to ``gt`` when no cache is available."""
        reference = ProgressReference(self.progress_reference)
        if reference == ProgressReference.METRIC_CACHE and not has_cache:
            return ProgressReference.GT
        return reference

    @classmethod
    def from_eval_config(cls, eval_cfg: Any) -> "PDMSConfig":
        """Build a validated :class:`PDMSConfig` from an ``EvalConfig``.

        Reads ``eval_cfg.plugin_scorer_configs['pdms']`` (an OmegaConf mapping
        or plain dict) and merges it onto the structured defaults, which both
        coerces types (including the string enums) and rejects unknown keys.
        """
        raw = _get_plugin_sub_config(eval_cfg, PLUGIN_NAME)
        merged = OmegaConf.merge(OmegaConf.structured(cls), raw)
        return OmegaConf.to_object(merged)  # type: ignore[return-value]


def _get_plugin_sub_config(eval_cfg: Any, name: str) -> dict[str, Any]:
    """Return ``eval_cfg.plugin_scorer_configs[name]`` as a plain dict."""
    container = getattr(eval_cfg, "plugin_scorer_configs", None)
    if container is None:
        return {}
    # Support both attribute-style (OmegaConf/dataclass) and mapping access.
    if OmegaConf.is_config(container):
        container = OmegaConf.to_container(container, resolve=True)
    if isinstance(container, dict):
        sub = container.get(name, {})
    else:  # pragma: no cover - defensive
        sub = getattr(container, name, {})
    return dict(sub) if sub else {}
