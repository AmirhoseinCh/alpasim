# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared fixtures and lightweight fakes for the PDMS plugin tests.

These tests are designed to run **without** the nuplan/navsim stack (and, for
the pure-adapter tests, without ``alpasim_eval``): the adapter modules only
import ``eval``/``nuplan`` lazily or under ``TYPE_CHECKING``.  A small
``FakeRenderable`` duck-types the pieces of ``eval.data.RenderableTrajectory``
the pure adapters use (``timestamps_us``, ``interpolate``, ``positions``,
``yaws``, ``raabb``).
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class FakeRAABB:
    size_x: float = 4.5
    size_y: float = 2.0
    size_z: float = 1.8


class FakeRenderable:
    """Minimal stand-in for ``RenderableTrajectory`` used by the pure adapters.

    Positions are ``(T, 3)`` and quaternions/yaws are derived from a supplied
    heading array.  ``interpolate`` does simple linear interpolation of the xy
    positions and (unwrapped) headings onto the requested integer-µs grid.
    """

    def __init__(
        self,
        timestamps_us: np.ndarray,
        positions_xy: np.ndarray,
        headings: np.ndarray,
        raabb: FakeRAABB | None = None,
    ) -> None:
        self._ts = np.asarray(timestamps_us, dtype=np.uint64)
        xy = np.asarray(positions_xy, dtype=np.float64)
        self._positions = np.column_stack([xy, np.zeros(len(xy))])
        self._yaws = np.asarray(headings, dtype=np.float64)
        self.raabb = raabb or FakeRAABB()

    def __len__(self) -> int:
        return len(self._ts)

    @property
    def timestamps_us(self) -> np.ndarray:
        return self._ts

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def yaws(self) -> np.ndarray:
        return self._yaws

    def interpolate(self, target_timestamps: np.ndarray) -> "FakeRenderable":
        target = np.asarray(target_timestamps, dtype=np.float64)
        src = self._ts.astype(np.float64)
        x = np.interp(target, src, self._positions[:, 0])
        y = np.interp(target, src, self._positions[:, 1])
        unwrapped = np.unwrap(self._yaws)
        h = np.interp(target, src, unwrapped)
        return FakeRenderable(
            np.asarray(target_timestamps, dtype=np.uint64),
            np.column_stack([x, y]),
            h,
            raabb=self.raabb,
        )


def straight_line_renderable(
    n: int = 11,
    dt_us: int = 100_000,
    speed_mps: float = 10.0,
    heading: float = 0.0,
    start_us: int = 1_000_000,
) -> FakeRenderable:
    """A constant-velocity straight-line trajectory along ``heading``."""
    ts = start_us + np.arange(n, dtype=np.uint64) * dt_us
    t_s = np.arange(n) * (dt_us / 1e6)
    dist = speed_mps * t_s
    xs = dist * np.cos(heading)
    ys = dist * np.sin(heading)
    headings = np.full(n, heading, dtype=np.float64)
    return FakeRenderable(ts, np.column_stack([xs, ys]), headings)
