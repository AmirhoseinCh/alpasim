# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Adapters converting an Alpasim ``SimulationResult`` into PDM scorer inputs.

The submodules split cleanly into:

* Pure, numpy-only helpers that are unit-tested without the nuplan stack:
  :mod:`alpasim_pdms.adapter.resample`, :mod:`alpasim_pdms.adapter.frame`,
  :mod:`alpasim_pdms.adapter.ego`, and the box-extraction part of
  :mod:`alpasim_pdms.adapter.tracked_objects`.
* Orchestration that imports ``nuplan``/``navsim`` lazily:
  :mod:`alpasim_pdms.adapter.pdm_inputs` and the nuplan-conversion part of
  :mod:`alpasim_pdms.adapter.tracked_objects`.
"""

from __future__ import annotations

from alpasim_pdms.adapter.ego import STATE_SIZE, StateIndex, build_state_array
from alpasim_pdms.adapter.frame import Rigid2D
from alpasim_pdms.adapter.resample import (
    TrajectorySample,
    body_accelerations,
    body_velocities_from_path,
    make_time_grid,
)
from alpasim_pdms.adapter.tracked_objects import ActorBox, extract_actor_boxes

__all__ = [
    "Rigid2D",
    "TrajectorySample",
    "make_time_grid",
    "body_velocities_from_path",
    "body_accelerations",
    "StateIndex",
    "STATE_SIZE",
    "build_state_array",
    "ActorBox",
    "extract_actor_boxes",
]
