# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Resampling and finite-difference kinematics for PDM state arrays.

PDM scoring assumes fixed-interval states (NAVSIM: 10 Hz).  Alpasim rollouts
run at ``session_metadata.control_timestep_us``, so ego and actor trajectories
are resampled onto a uniform grid before scoring.  Velocities/accelerations are
derived by finite differences and expressed in the ego body frame (x forward,
y left), matching nuPlan's ``EgoState`` conventions and WorldEngine's
``compute_pdm_scores`` (which likewise injects externally computed body-frame
velocities into the scorer state array).

Everything here is pure numpy and unit-tested without the nuplan stack; the only
function that touches Alpasim types is :func:`sample_from_renderable`, a thin
wrapper over ``RenderableTrajectory.interpolate``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from eval.data import RenderableTrajectory

US_PER_S = 1_000_000.0


def make_time_grid(start_us: int, end_us: int, hz: float) -> np.ndarray:
    """Return a uniform ``[start_us, end_us]`` timestamp grid at ``hz`` (uint64).

    ``end_us`` is included when it lands on the grid (within rounding).  The grid
    always contains at least ``start_us``.
    """
    if hz <= 0:
        raise ValueError(f"hz must be positive, got {hz}")
    if end_us < start_us:
        raise ValueError(f"end_us ({end_us}) must be >= start_us ({start_us})")
    step_us = US_PER_S / hz
    n_steps = int(np.floor((end_us - start_us) / step_us + 1e-6))
    idx = np.arange(n_steps + 1, dtype=np.float64)
    grid = np.round(start_us + idx * step_us).astype(np.uint64)
    return grid


def _dt_seconds(timestamps_us: np.ndarray) -> np.ndarray:
    """Return the time coordinate (seconds) for gradient computations."""
    t = np.asarray(timestamps_us, dtype=np.float64) / US_PER_S
    return t - t[0]


def _unwrapped_gradient(values: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Central-difference gradient robust to <2 samples."""
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, t_s)


def global_velocities_from_path(
    positions_xy: np.ndarray, timestamps_us: np.ndarray
) -> np.ndarray:
    """Finite-difference global-frame velocities ``(T, 2)`` from an xy path."""
    positions_xy = np.asarray(positions_xy, dtype=np.float64)
    t_s = _dt_seconds(timestamps_us)
    if len(positions_xy) < 2:
        return np.zeros((len(positions_xy), 2), dtype=np.float64)
    vx = _unwrapped_gradient(positions_xy[:, 0], t_s)
    vy = _unwrapped_gradient(positions_xy[:, 1], t_s)
    return np.stack([vx, vy], axis=-1)


def rotate_global_to_body(vectors_xy: np.ndarray, headings: np.ndarray) -> np.ndarray:
    """Rotate global-frame 2D vectors into the per-frame body frame.

    ``v_body = R(-heading) @ v_global`` with x forward, y left.
    """
    vectors_xy = np.asarray(vectors_xy, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)
    cos_h = np.cos(headings)
    sin_h = np.sin(headings)
    vx_body = vectors_xy[:, 0] * cos_h + vectors_xy[:, 1] * sin_h
    vy_body = -vectors_xy[:, 0] * sin_h + vectors_xy[:, 1] * cos_h
    return np.stack([vx_body, vy_body], axis=-1)


def body_velocities_from_path(
    positions_xy: np.ndarray, headings: np.ndarray, timestamps_us: np.ndarray
) -> np.ndarray:
    """Body-frame velocities ``(T, 2)`` from an xy path and per-frame headings."""
    global_vel = global_velocities_from_path(positions_xy, timestamps_us)
    return rotate_global_to_body(global_vel, headings)


def body_accelerations(
    body_velocities: np.ndarray, timestamps_us: np.ndarray
) -> np.ndarray:
    """Body-frame accelerations ``(T, 2)`` from body-frame velocities.

    Accelerations are differentiated component-wise in the body frame, matching
    nuPlan's ``DynamicCarState`` (``acceleration`` stored in the rear-axle
    frame).  For the near-constant-heading horizons PDM scores this is an
    adequate approximation.
    """
    body_velocities = np.asarray(body_velocities, dtype=np.float64)
    t_s = _dt_seconds(timestamps_us)
    if len(body_velocities) < 2:
        return np.zeros((len(body_velocities), 2), dtype=np.float64)
    ax = _unwrapped_gradient(body_velocities[:, 0], t_s)
    ay = _unwrapped_gradient(body_velocities[:, 1], t_s)
    return np.stack([ax, ay], axis=-1)


@dataclasses.dataclass
class TrajectorySample:
    """A trajectory resampled onto a uniform grid, with derived kinematics.

    All arrays are length ``T`` and share ``timestamps_us``.  Positions are the
    rear-axle (or box-center; see caller) xy in the frame the sample was built
    in; velocities/accelerations are in the per-frame body frame.
    """

    timestamps_us: np.ndarray  # (T,) uint64
    positions_xy: np.ndarray  # (T, 2)
    headings: np.ndarray  # (T,)
    velocities_body: np.ndarray  # (T, 2)
    accelerations_body: np.ndarray  # (T, 2)

    def __len__(self) -> int:
        return len(self.timestamps_us)

    @classmethod
    def from_arrays(
        cls,
        timestamps_us: np.ndarray,
        positions_xy: np.ndarray,
        headings: np.ndarray,
    ) -> "TrajectorySample":
        """Build a sample, deriving body velocities/accelerations by finite diff."""
        timestamps_us = np.asarray(timestamps_us, dtype=np.uint64)
        positions_xy = np.asarray(positions_xy, dtype=np.float64)
        headings = np.asarray(headings, dtype=np.float64)
        vel = body_velocities_from_path(positions_xy, headings, timestamps_us)
        acc = body_accelerations(vel, timestamps_us)
        return cls(
            timestamps_us=timestamps_us,
            positions_xy=positions_xy,
            headings=headings,
            velocities_body=vel,
            accelerations_body=acc,
        )


def sample_from_renderable(
    renderable: "RenderableTrajectory", grid_us: np.ndarray
) -> TrajectorySample:
    """Resample a ``RenderableTrajectory`` onto ``grid_us`` and derive kinematics.

    Uses the trajectory's own (SLERP-based) interpolation to stay consistent
    with the rest of the eval codebase, then extracts planar positions/headings
    and finite-differences the body-frame velocities/accelerations.
    """
    grid_us = np.asarray(grid_us, dtype=np.uint64)
    interp = renderable.interpolate(grid_us)
    positions_xy = np.asarray(interp.positions, dtype=np.float64)[:, :2]
    headings = np.asarray(interp.yaws, dtype=np.float64)
    return TrajectorySample.from_arrays(grid_us, positions_xy, headings)
