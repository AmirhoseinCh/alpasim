# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Convert Alpasim actor trajectories into per-frame tracked objects.

``PDMObservation`` builds its own occupancy maps from nuPlan ``DetectionsTracks``
(boxes), not from Alpasim's ``actor_polygons`` STRtrees, so we feed it boxes.
This module first extracts frame-aligned :class:`ActorBox` records (pure numpy,
unit-tested) and then, only when scoring, converts them into nuPlan
``DetectionsTracks`` via lazily imported nuplan types.

Traffic-light handling: Alpasim has no traffic-light state channel, so callers
pass empty per-frame traffic-light lists.  Under NAVSIM v2 this trivially sets
the traffic-light-compliance gate to 1 (see README "Limitations").
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
from alpasim_pdms.adapter.frame import Rigid2D

if TYPE_CHECKING:  # pragma: no cover - typing only
    from eval.data import RenderableTrajectory

EGO_ID = "EGO"

#: Alpasim/generic category -> nuPlan ``TrackedObjectType`` name.  Alpasim's
#: ``SimulationResult.actor_trajectories`` does not currently carry per-actor
#: class labels, so callers default to ``vehicle`` unless a ``label_by_id`` map
#: is supplied (e.g. derived from ``RolloutMetadata.ActorDefinitions``).
CATEGORY_TO_NUPLAN_TYPE = {
    "vehicle": "VEHICLE",
    "pedestrian": "PEDESTRIAN",
    "bicycle": "BICYCLE",
    "generic": "GENERIC_OBJECT",
    "traffic_cone": "TRAFFIC_CONE",
    "barrier": "BARRIER",
    "czone_sign": "CZONE_SIGN",
}
DEFAULT_CATEGORY = "vehicle"


@dataclasses.dataclass
class ActorBox:
    """A single actor's oriented bounding box at one timestep (nuPlan frame)."""

    token: str
    timestamp_us: int
    x: float
    y: float
    heading: float
    length: float
    width: float
    category: str = DEFAULT_CATEGORY

    @property
    def nuplan_type_name(self) -> str:
        return CATEGORY_TO_NUPLAN_TYPE.get(self.category, "VEHICLE")


def _actor_time_bounds(renderable: "RenderableTrajectory") -> tuple[int, int]:
    ts = np.asarray(renderable.timestamps_us, dtype=np.uint64)
    return int(ts.min()), int(ts.max())


def extract_actor_boxes(
    actor_trajectories: Mapping[str, "RenderableTrajectory"],
    grid_us: np.ndarray,
    transform: Rigid2D | None = None,
    label_by_id: Mapping[str, str] | None = None,
    ego_id: str = EGO_ID,
) -> list[list[ActorBox]]:
    """Extract per-frame actor boxes aligned to ``grid_us`` (nuPlan frame).

    For each non-ego actor, the trajectory is interpolated onto the grid and, for
    grid timestamps within the actor's own time range, an :class:`ActorBox` is
    produced.  Positions/headings are mapped through ``transform`` (identity if
    ``None``); box extents come from the actor's ``RAABB`` (length ``size_x``,
    width ``size_y``).

    Returns a list of length ``len(grid_us)``; entry ``i`` is the list of boxes
    present at ``grid_us[i]``.
    """
    grid_us = np.asarray(grid_us, dtype=np.uint64)
    transform = transform or Rigid2D.identity()
    label_by_id = label_by_id or {}
    frames: list[list[ActorBox]] = [[] for _ in range(len(grid_us))]

    for actor_id, renderable in actor_trajectories.items():
        if actor_id == ego_id:
            continue
        if len(renderable) == 0:
            continue
        t_min, t_max = _actor_time_bounds(renderable)
        in_range = (grid_us >= t_min) & (grid_us <= t_max)
        if not in_range.any():
            continue

        interp = renderable.interpolate(grid_us)
        positions_xy = np.asarray(interp.positions, dtype=np.float64)[:, :2]
        headings = np.asarray(interp.yaws, dtype=np.float64)
        positions_xy, headings = transform.apply_poses(positions_xy, headings)

        raabb = renderable.raabb
        length = float(raabb.size_x) if raabb is not None else 4.5
        width = float(raabb.size_y) if raabb is not None else 2.0
        category = label_by_id.get(actor_id, DEFAULT_CATEGORY)

        for i, present in enumerate(in_range):
            if not present:
                continue
            frames[i].append(
                ActorBox(
                    token=actor_id,
                    timestamp_us=int(grid_us[i]),
                    x=float(positions_xy[i, 0]),
                    y=float(positions_xy[i, 1]),
                    heading=float(headings[i]),
                    length=length,
                    width=width,
                    category=category,
                )
            )
    return frames


# ---------------------------------------------------------------------------
# nuPlan conversion (lazy imports; requires the nuplan stack).
# ---------------------------------------------------------------------------
def boxes_to_detections_tracks(
    frames: Iterable[list[ActorBox]],
    timestamps_us: Iterable[int],
) -> list[Any]:
    """Convert per-frame :class:`ActorBox` lists to nuPlan ``DetectionsTracks``.

    Imports ``nuplan`` lazily so the rest of the adapter stays import-clean when
    the nuplan stack is absent.  Each box becomes an ``Agent`` with a static
    (zero-velocity) ``StateVector2D``; the PDM scorer derives interactions from
    geometry, so per-agent velocity is not required for the sub-scores used here.
    """
    from nuplan.common.actor_state.agent import Agent
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.scene_object import SceneObjectMetadata
    from nuplan.common.actor_state.state_representation import (
        StateSE2,
        StateVector2D,
        TimePoint,
    )
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

    detections: list[Any] = []
    for frame_boxes, ts_us in zip(frames, timestamps_us):
        tracked = []
        for box in frame_boxes:
            obj_type = TrackedObjectType[box.nuplan_type_name]
            oriented_box = OrientedBox(
                StateSE2(box.x, box.y, box.heading),
                length=box.length,
                width=box.width,
                height=1.8,
            )
            metadata = SceneObjectMetadata(
                timestamp_us=int(ts_us),
                token=box.token,
                track_id=None,
                track_token=box.token,
            )
            tracked.append(
                Agent(
                    tracked_object_type=obj_type,
                    oriented_box=oriented_box,
                    velocity=StateVector2D(0.0, 0.0),
                    metadata=metadata,
                )
            )
        detections.append(
            DetectionsTracks(
                tracked_objects=TrackedObjects(tracked),
                _time_point=TimePoint(int(ts_us)),
            )
        )
    return detections
