# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging

from eval.schema import EvalConfig
from eval.scorers.base import Scorer, ScorerGroup
from eval.scorers.collision import CollisionScorer
from eval.scorers.ground_truth import GroundTruthScorer
from eval.scorers.image import ImageScorer
from eval.scorers.min_distance_to_obstacle import MinDistanceToObstacleScorer
from eval.scorers.minADE import MinADEScorer
from eval.scorers.offroad import OffRoadScorer
from eval.scorers.plan_deviation import PlanDeviationScorer
from eval.scorers.safety import SafetyScorer

logger = logging.getLogger("alpasim_eval")

SCORERS = [
    CollisionScorer,
    OffRoadScorer,
    MinDistanceToObstacleScorer,
    GroundTruthScorer,
    MinADEScorer,
    PlanDeviationScorer,
    ImageScorer,
    SafetyScorer,
]


def _create_plugin_scorers(cfg: EvalConfig) -> list[Scorer]:
    """Instantiate scorers registered by plugins under ``alpasim.scorers``.

    Plugin scorers extend (never replace) the built-in scorers.  Which plugin
    scorers run is controlled by ``cfg.enabled_plugin_scorers``:

    * ``None`` (default) runs every registered plugin scorer.
    * A list restricts execution to the named scorers (unknown names are
      logged and skipped).
    * An empty list disables plugin scorers entirely.

    Discovery failures (e.g. ``alpasim_plugins`` not installed) and per-scorer
    construction errors are logged and skipped so that a misbehaving plugin can
    never prevent the built-in metrics from running.
    """
    try:
        from alpasim_plugins import scorers as scorer_registry

        available = scorer_registry.get_names()
    except Exception as e:  # pragma: no cover - defensive, plugins is core
        logger.warning("Plugin scorer discovery unavailable: %s", e)
        return []

    if cfg.enabled_plugin_scorers is None:
        selected = available
    else:
        selected = list(cfg.enabled_plugin_scorers)
        for name in selected:
            if name not in available:
                logger.warning(
                    "Requested plugin scorer '%s' is not registered. Available: %s",
                    name,
                    available,
                )

    plugin_scorers: list[Scorer] = []
    for name in selected:
        if name not in available:
            continue
        try:
            plugin_scorers.append(scorer_registry.get(name)(cfg))
            logger.info("Enabled plugin scorer '%s'.", name)
        except Exception as e:
            logger.error("Failed to instantiate plugin scorer '%s': %s", name, e)
    return plugin_scorers


def create_scorer_group(cfg: EvalConfig) -> ScorerGroup:
    """Initialize all scorers (built-in first, then plugin scorers)."""
    scorers: list[Scorer] = []
    for scorer in SCORERS:
        scorers.append(scorer(cfg))
    scorers.extend(_create_plugin_scorers(cfg))
    return ScorerGroup(scorers)
