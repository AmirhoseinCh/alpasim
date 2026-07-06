# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Assemble PDM scorer inputs and read out sub-scores (nuplan-dependent).

This is the orchestration layer that mirrors WorldEngine's
``metric_manager.compute_pdm_scores()`` recipe, adapted to score *already
simulated* Alpasim rollout states directly (no re-simulation; see README
"Deviations from NAVSIM").  Everything here imports ``nuplan``/``navsim`` and the
vendored scorer lazily, so importing the module is cheap and the rest of the
plugin works without the nuplan stack.

Inputs are built for the **navtest metric-cache path**:

* anchor ego state, PDM-closed reference trajectory, route centerline,
  ``route_lane_ids`` and drivable-area map come from the NAVSIM metric cache;
* the policy (and GT) ego states and the background actor boxes come from the
  Alpasim rollout, mapped into the nuPlan map frame via an anchor-derived
  :class:`~alpasim_pdms.adapter.frame.Rigid2D`;
* a nuPlan ``map_api`` is loaded once per map (maps-only download) for the
  scorer's intersection test in the TTC metric.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from alpasim_pdms.adapter.ego import build_state_array, stack_proposals
from alpasim_pdms.adapter.frame import Rigid2D, anchor_gap_m
from alpasim_pdms.adapter.resample import TrajectorySample, make_time_grid
from alpasim_pdms.adapter.tracked_objects import (
    boxes_to_detections_tracks,
    extract_actor_boxes,
)
from alpasim_pdms.config import PDMSConfig, ProgressReference

if TYPE_CHECKING:  # pragma: no cover - typing only
    from eval.data import RenderableTrajectory, SimulationResult

logger = logging.getLogger("alpasim_pdms.pdm_inputs")

# Sub-score names (without the configurable prefix).
SUBSCORE_NO_COLLISION = "no_at_fault_collision"
SUBSCORE_DRIVABLE_AREA = "drivable_area_compliance"
SUBSCORE_PROGRESS = "ego_progress"
SUBSCORE_TTC = "ttc_within_bound"
SUBSCORE_COMFORT = "comfort"
SUBSCORE_DRIVING_DIRECTION = "driving_direction_compliance"
SUBSCORE_LANE_KEEPING = "lane_keeping"
SCORE = "score"


class PDMDependencyError(RuntimeError):
    """Raised when the nuplan/navsim stack or map data is unavailable."""


@dataclasses.dataclass
class PDMSubscores:
    """Sub-scores and aggregate for one scored proposal."""

    no_at_fault_collision: float
    drivable_area_compliance: float
    ego_progress: float
    ttc_within_bound: float
    comfort: float
    driving_direction_compliance: float
    lane_keeping: float
    score: float
    first_violation_timestamp_us: int | None = None

    def as_named(self, prefix: str) -> dict[str, float]:
        """Return a ``{f"{prefix}/{name}": value}`` mapping of the sub-scores."""
        return {
            f"{prefix}/{SUBSCORE_NO_COLLISION}": self.no_at_fault_collision,
            f"{prefix}/{SUBSCORE_DRIVABLE_AREA}": self.drivable_area_compliance,
            f"{prefix}/{SUBSCORE_PROGRESS}": self.ego_progress,
            f"{prefix}/{SUBSCORE_TTC}": self.ttc_within_bound,
            f"{prefix}/{SUBSCORE_COMFORT}": self.comfort,
            f"{prefix}/{SUBSCORE_DRIVING_DIRECTION}": self.driving_direction_compliance,
            f"{prefix}/{SUBSCORE_LANE_KEEPING}": self.lane_keeping,
            f"{prefix}/{SCORE}": self.score,
        }


def build_proposal_sampling(config: PDMSConfig) -> Any:
    """Build the NAVSIM ``TrajectorySampling`` for proposals (lazy import)."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    return TrajectorySampling(
        num_poses=config.proposal_sampling_num_poses,
        interval_length=config.proposal_sampling_interval_s,
    )


@functools.lru_cache(maxsize=8)
def load_map_api(nuplan_map_root: str, map_location: str) -> Any:
    """Load (and cache) a nuPlan ``map_api`` for a map location (lazy import)."""
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    return get_maps_api(nuplan_map_root, "nuplan-maps-v1.0", map_location)


def _cache_attr(metric_cache: Any, *names: str) -> Any:
    """Return the first present attribute among ``names`` on the metric cache."""
    for name in names:
        if hasattr(metric_cache, name):
            return getattr(metric_cache, name)
    raise PDMDependencyError(
        f"Metric cache is missing all of {names!r}; incompatible navsim version?"
    )


def _ego_state_pose_xytheta(ego_state: Any) -> tuple[float, float, float]:
    """Extract ``(x, y, heading)`` of a nuPlan ``EgoState`` rear axle."""
    rear = ego_state.rear_axle
    return float(rear.x), float(rear.y), float(rear.heading)


def _rollout_anchor_pose_local(
    ego: "RenderableTrajectory", anchor_us: int
) -> tuple[float, float, float]:
    """Ego rear-axle-ish planar pose in the local frame at the anchor time."""
    interp = ego.interpolate(np.asarray([anchor_us], dtype=np.uint64))
    x, y = float(interp.positions[0, 0]), float(interp.positions[0, 1])
    heading = float(interp.yaws[0])
    return x, y, heading


def _select_ego(
    sim_result: "SimulationResult", config: PDMSConfig
) -> "RenderableTrajectory":
    from alpasim_pdms.config import EgoTrajectorySource

    if config.ego_trajectory_source == EgoTrajectorySource.DRIVER_ESTIMATED:
        return sim_result.driver_estimated_trajectory
    return sim_result.actor_trajectories["EGO"]


def _resample_in_nuplan_frame(
    renderable: "RenderableTrajectory", grid_us: np.ndarray, transform: Rigid2D
) -> TrajectorySample:
    """Resample a trajectory and map it into the nuPlan frame."""
    interp = renderable.interpolate(np.asarray(grid_us, dtype=np.uint64))
    positions_xy = np.asarray(interp.positions, dtype=np.float64)[:, :2]
    headings = np.asarray(interp.yaws, dtype=np.float64)
    positions_xy, headings = transform.apply_poses(positions_xy, headings)
    return TrajectorySample.from_arrays(grid_us, positions_xy, headings)


@dataclasses.dataclass
class PDMScoreInputs:
    """Fully-assembled inputs to the vendored ``PDMScorer.score_proposals``."""

    states: np.ndarray  # (P, T, 11); proposal 0 = reference, 1 = policy, [2 = GT]
    initial_ego_state: Any
    observation: Any
    centerline: Any
    route_lane_dict: dict
    drivable_area_map: Any
    map_api: Any
    proposal_sampling: Any
    policy_index: int
    gt_index: int | None
    anchor_gap_m: float


def build_inputs_from_cache(
    sim_result: "SimulationResult",
    metric_cache: Any,
    config: PDMSConfig,
) -> PDMScoreInputs:
    """Assemble PDM scorer inputs from a rollout + NAVSIM metric cache.

    Proposal layout follows WorldEngine's expert-referenced batch mode: index 0
    is the reference (PDM-closed or GT, per ``config.progress_reference``), index
    1 is the policy, and — when ``config.human_penalty_filter`` is on — index 2
    is the GT rollout used to un-penalize shared infractions.
    """
    proposal_sampling = build_proposal_sampling(config)
    n_frames = config.proposal_sampling_num_poses + 1

    # --- anchor + frame transform ------------------------------------------
    anchor_ego_state = _cache_attr(metric_cache, "ego_state", "initial_ego_state")
    cache_anchor = _ego_state_pose_xytheta(anchor_ego_state)

    anchor_us = sim_result.first_driven_timestamp_us
    if anchor_us is None:
        anchor_us = int(sim_result.timestamps_us[0])
    ego = _select_ego(sim_result, config)
    rollout_anchor = _rollout_anchor_pose_local(ego, anchor_us)
    transform = Rigid2D.from_anchor(rollout_anchor, cache_anchor)
    gap = anchor_gap_m(np.asarray(rollout_anchor), np.asarray(cache_anchor), transform)
    if gap > config.anchor_tolerance_m:
        logger.warning(
            "Anchor gap %.3f m exceeds tolerance %.3f m for scene %s; frame "
            "alignment may be wrong.",
            gap,
            config.anchor_tolerance_m,
            sim_result.session_metadata.scene_id,
        )

    # --- time grid (anchor -> anchor + horizon) ----------------------------
    horizon_us = int(
        round(
            config.proposal_sampling_num_poses
            * config.proposal_sampling_interval_s
            * 1_000_000
        )
    )
    last_us = int(sim_result.timestamps_us[-1])
    end_us = min(anchor_us + horizon_us, last_us)
    grid_us = make_time_grid(anchor_us, end_us, config.resample_hz)
    grid_us = _pad_or_trim_grid(grid_us, n_frames, config.resample_hz)

    # --- policy + GT state arrays (nuPlan frame) ---------------------------
    policy_sample = _resample_in_nuplan_frame(ego, grid_us, transform)
    policy_states = build_state_array(policy_sample)

    gt_ego = sim_result.ego_recorded_ground_truth_trajectory
    gt_sample = _resample_in_nuplan_frame(gt_ego, grid_us, transform)
    gt_states = build_state_array(gt_sample)

    # --- reference (proposal 0) --------------------------------------------
    has_cache_reference = _has_pdm_closed_reference(metric_cache)
    progress_reference = config.resolved_progress_reference(has_cache_reference)
    if progress_reference == ProgressReference.METRIC_CACHE and has_cache_reference:
        reference_states = _pdm_closed_states_from_cache(
            metric_cache, anchor_ego_state, proposal_sampling, n_frames
        )
    else:
        reference_states = gt_states

    proposals = [reference_states, policy_states]
    gt_index: int | None = None
    if config.human_penalty_filter:
        proposals.append(gt_states)
        gt_index = 2
    states = stack_proposals(*proposals)

    # --- observation from rollout actor boxes ------------------------------
    observation = _build_observation(
        sim_result, grid_us, transform, config, proposal_sampling
    )

    # --- map-derived inputs from the cache ---------------------------------
    centerline = _cache_attr(metric_cache, "centerline")
    drivable_area_map = _cache_attr(metric_cache, "drivable_area_map")
    route_lane_dict = _route_lane_dict_from_cache(metric_cache)

    map_api = _resolve_map_api(metric_cache, config)

    return PDMScoreInputs(
        states=states,
        initial_ego_state=anchor_ego_state,
        observation=observation,
        centerline=centerline,
        route_lane_dict=route_lane_dict,
        drivable_area_map=drivable_area_map,
        map_api=map_api,
        proposal_sampling=proposal_sampling,
        policy_index=1,
        gt_index=gt_index,
        anchor_gap_m=gap,
    )


def _pad_or_trim_grid(grid_us: np.ndarray, n_frames: int, hz: float) -> np.ndarray:
    """Force the grid to exactly ``n_frames`` samples (scorer requires T fixed).

    Short rollouts are padded by holding the last timestamp (the resampled pose
    will hold the final ego pose, i.e. a stopped ego), long grids are trimmed.
    """
    grid_us = np.asarray(grid_us, dtype=np.uint64)
    if len(grid_us) == n_frames:
        return grid_us
    if len(grid_us) > n_frames:
        return grid_us[:n_frames]
    if len(grid_us) == 0:
        raise PDMDependencyError("Empty resampling grid; rollout has no timestamps.")
    pad = np.full(n_frames - len(grid_us), grid_us[-1], dtype=np.uint64)
    return np.concatenate([grid_us, pad])


def _build_observation(
    sim_result: "SimulationResult",
    grid_us: np.ndarray,
    transform: Rigid2D,
    config: PDMSConfig,
    proposal_sampling: Any,
) -> Any:
    """Build a ``PDMObservation`` from the rollout's actual actor tracks."""
    from alpasim_pdms.vendored.pdm_planner.observation.pdm_observation import (
        PDMObservation,
    )

    frames = extract_actor_boxes(
        sim_result.actor_trajectories, grid_us, transform=transform
    )
    detections = boxes_to_detections_tracks(frames, [int(t) for t in grid_us])
    # No traffic-light channel in Alpasim: empty per-frame lists.
    traffic_lights: list[list] = [[] for _ in grid_us]

    observation = PDMObservation(
        proposal_sampling,
        proposal_sampling,
        config.map_radius_m,
        observation_sample_res=1,
    )
    observation.update_detections_tracks(
        detections,
        traffic_lights,
        {},  # route_lane_dict only used for traffic-light lane filtering
        compute_traffic_light_data=False,
    )
    return observation


def _has_pdm_closed_reference(metric_cache: Any) -> bool:
    return any(
        hasattr(metric_cache, n) for n in ("trajectory", "pdm_closed_trajectory")
    )


def _pdm_closed_states_from_cache(
    metric_cache: Any,
    anchor_ego_state: Any,
    proposal_sampling: Any,
    n_frames: int,
) -> np.ndarray:
    """Sample the cached PDM-closed trajectory into a scorer state array."""
    from alpasim_pdms.vendored.pdm_planner.utils.pdm_array_representation import (
        ego_states_to_state_array,
    )

    trajectory = _cache_attr(metric_cache, "trajectory", "pdm_closed_trajectory")
    interval = proposal_sampling.interval_length
    time_points = [
        anchor_ego_state.time_point + _time_point(int(round(i * interval * 1e6)))
        for i in range(n_frames)
    ]
    ego_states = [anchor_ego_state] + trajectory.get_state_at_times(time_points[1:])
    state_array = ego_states_to_state_array(ego_states)
    return np.asarray(state_array, dtype=np.float64)[:n_frames]


def _time_point(delta_us: int) -> Any:
    from nuplan.common.actor_state.state_representation import TimePoint

    return TimePoint(delta_us)


def _route_lane_dict_from_cache(metric_cache: Any) -> dict:
    """Return a ``{lane_id: obj}`` dict; the scorer only uses the keys."""
    if hasattr(metric_cache, "route_lane_dict") and metric_cache.route_lane_dict:
        return dict(metric_cache.route_lane_dict)
    for name in ("route_lane_ids", "route_ids"):
        ids = getattr(metric_cache, name, None)
        if ids:
            return {str(lane_id): None for lane_id in ids}
    logger.warning("Metric cache has no route lane ids; DAC on-route mask empty.")
    return {}


def _resolve_map_api(metric_cache: Any, config: PDMSConfig) -> Any:
    """Return a nuPlan ``map_api`` from the cache or ``nuplan_map_root``."""
    for name in ("map_api", "_map_api"):
        api = getattr(metric_cache, name, None)
        if api is not None:
            return api
    map_location = None
    for name in ("map_name", "map_location", "map"):
        map_location = getattr(metric_cache, name, None)
        if map_location:
            break
    if config.nuplan_map_root and map_location:
        return load_map_api(config.nuplan_map_root, str(map_location))
    raise PDMDependencyError(
        "No map_api available: the metric cache carries no map and "
        "pdms.nuplan_map_root / cache map_name are unset. The TTC metric "
        "requires a nuPlan map."
    )
