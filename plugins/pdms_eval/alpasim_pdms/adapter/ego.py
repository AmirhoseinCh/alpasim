# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Build the PDM scorer state array from a resampled ego trajectory.

The PDM scorer consumes an array of ego states with a fixed column layout.  This
mirrors ``StateIndex`` in the vendored ``pdm_planner`` (and nuPlan/NAVSIM):
``[x, y, heading, vx, vy, ax, ay, steering_angle, steering_rate,
angular_velocity, angular_acceleration]``.  We fill pose + body-frame
velocity/acceleration (the fields the scorer's collision/TTC/comfort terms use)
and leave the steering/angular columns at zero, exactly as WorldEngine's
``get_pred_states`` does when scoring already-simulated states.

Kept independent of the vendored (nuplan-tainted) package so it is unit-testable
without the nuplan stack; the constants are asserted against the vendored enum
in tests when that package is importable.
"""

from __future__ import annotations

import numpy as np
from alpasim_pdms.adapter.resample import TrajectorySample


class StateIndex:
    """Column indices for the PDM ego-state array (matches nuPlan/NAVSIM)."""

    X = 0
    Y = 1
    HEADING = 2
    VELOCITY_X = 3
    VELOCITY_Y = 4
    ACCELERATION_X = 5
    ACCELERATION_Y = 6
    STEERING_ANGLE = 7
    STEERING_RATE = 8
    ANGULAR_VELOCITY = 9
    ANGULAR_ACCELERATION = 10


STATE_SIZE = 11


def build_state_array(
    sample: TrajectorySample, include_acceleration: bool = True
) -> np.ndarray:
    """Assemble a ``(T, STATE_SIZE)`` PDM state array from a resampled sample.

    Args:
        sample: Resampled ego trajectory with body-frame kinematics, expressed
            in the nuPlan map frame (transform the sample before calling this).
        include_acceleration: Whether to populate the acceleration columns.

    Returns:
        ``float64`` array of shape ``(T, STATE_SIZE)``.
    """
    n = len(sample)
    states = np.zeros((n, STATE_SIZE), dtype=np.float64)
    states[:, StateIndex.X] = sample.positions_xy[:, 0]
    states[:, StateIndex.Y] = sample.positions_xy[:, 1]
    states[:, StateIndex.HEADING] = sample.headings
    states[:, StateIndex.VELOCITY_X] = sample.velocities_body[:, 0]
    states[:, StateIndex.VELOCITY_Y] = sample.velocities_body[:, 1]
    if include_acceleration:
        states[:, StateIndex.ACCELERATION_X] = sample.accelerations_body[:, 0]
        states[:, StateIndex.ACCELERATION_Y] = sample.accelerations_body[:, 1]
    return states


def stack_proposals(*state_arrays: np.ndarray) -> np.ndarray:
    """Stack per-proposal ``(T, STATE_SIZE)`` arrays into ``(P, T, STATE_SIZE)``.

    All proposals must share the same number of timesteps ``T`` (the PDM scorer
    scores a batch of equal-length proposals; proposal 0 is conventionally the
    reference/expert and proposal 1 the policy).
    """
    if not state_arrays:
        raise ValueError("stack_proposals requires at least one state array")
    lengths = {arr.shape[0] for arr in state_arrays}
    if len(lengths) != 1:
        raise ValueError(f"proposals must share T; got lengths {sorted(lengths)}")
    return np.stack(state_arrays, axis=0)
