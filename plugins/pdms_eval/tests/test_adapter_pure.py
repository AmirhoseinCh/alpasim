# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for the pure (numpy-only) adapter helpers.

These run without the nuplan stack or ``alpasim_eval``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from alpasim_pdms.adapter.ego import (
    STATE_SIZE,
    StateIndex,
    build_state_array,
    stack_proposals,
)
from alpasim_pdms.adapter.frame import Rigid2D, anchor_gap_m, wrap_to_pi
from alpasim_pdms.adapter.resample import (
    TrajectorySample,
    body_accelerations,
    body_velocities_from_path,
    make_time_grid,
    rotate_global_to_body,
    sample_from_renderable,
)
from alpasim_pdms.adapter.tracked_objects import EGO_ID, extract_actor_boxes
from conftest import straight_line_renderable


# --------------------------------------------------------------------------- #
# resample
# --------------------------------------------------------------------------- #
class TestMakeTimeGrid:
    def test_basic_grid_10hz(self):
        grid = make_time_grid(0, 1_000_000, 10.0)
        assert grid.dtype == np.uint64
        assert grid[0] == 0
        assert grid[-1] == 1_000_000
        assert len(grid) == 11  # inclusive endpoints at 10 Hz over 1 s

    def test_grid_includes_start_only_when_zero_span(self):
        grid = make_time_grid(500, 500, 10.0)
        assert list(grid) == [500]

    def test_invalid_hz(self):
        with pytest.raises(ValueError):
            make_time_grid(0, 100, 0.0)

    def test_invalid_span(self):
        with pytest.raises(ValueError):
            make_time_grid(100, 0, 10.0)


class TestBodyKinematics:
    def test_rotate_global_to_body_identity_heading(self):
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = rotate_global_to_body(vecs, np.zeros(2))
        np.testing.assert_allclose(out, vecs)

    def test_rotate_global_to_body_90deg(self):
        # Heading +90°: global +x maps to body -y (x fwd, y left).
        out = rotate_global_to_body(np.array([[1.0, 0.0]]), np.array([math.pi / 2]))
        np.testing.assert_allclose(out, [[0.0, -1.0]], atol=1e-9)

    def test_body_velocity_constant_speed_forward(self):
        # Moving along +x at 10 m/s with heading 0 -> body vx≈10, vy≈0.
        n = 6
        ts = np.arange(n, dtype=np.uint64) * 100_000
        xs = np.arange(n) * 1.0  # 1 m per 0.1 s = 10 m/s
        pos = np.column_stack([xs, np.zeros(n)])
        vel = body_velocities_from_path(pos, np.zeros(n), ts)
        np.testing.assert_allclose(vel[:, 0], 10.0, atol=1e-6)
        np.testing.assert_allclose(vel[:, 1], 0.0, atol=1e-6)

    def test_acceleration_of_constant_velocity_is_zero(self):
        n = 6
        ts = np.arange(n, dtype=np.uint64) * 100_000
        vel = np.column_stack([np.full(n, 10.0), np.zeros(n)])
        acc = body_accelerations(vel, ts)
        np.testing.assert_allclose(acc, 0.0, atol=1e-6)

    def test_single_sample_yields_zero_kinematics(self):
        sample = TrajectorySample.from_arrays(
            np.array([5], dtype=np.uint64), np.array([[1.0, 2.0]]), np.array([0.3])
        )
        assert sample.velocities_body.shape == (1, 2)
        np.testing.assert_allclose(sample.velocities_body, 0.0)
        np.testing.assert_allclose(sample.accelerations_body, 0.0)


class TestSampleFromRenderable:
    def test_resample_preserves_speed(self):
        renderable = straight_line_renderable(n=11, speed_mps=10.0)
        grid = make_time_grid(1_000_000, 2_000_000, 10.0)
        sample = sample_from_renderable(renderable, grid)
        assert len(sample) == len(grid)
        # Interior body vx ≈ speed.
        np.testing.assert_allclose(sample.velocities_body[1:-1, 0], 10.0, atol=1e-3)


# --------------------------------------------------------------------------- #
# frame
# --------------------------------------------------------------------------- #
class TestRigid2D:
    def test_identity(self):
        t = Rigid2D.identity()
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(t.apply_points(pts), pts)

    def test_from_anchor_maps_src_onto_dst(self):
        src = (0.0, 0.0, 0.0)
        dst = (10.0, -5.0, math.pi / 2)
        t = Rigid2D.from_anchor(src, dst)
        mapped = t.apply_point(np.array(src[:2]))
        np.testing.assert_allclose(mapped, dst[:2], atol=1e-9)
        assert math.isclose(t.apply_heading(src[2]), dst[2], abs_tol=1e-9)
        assert anchor_gap_m(np.array(src), np.array(dst), t) < 1e-9

    def test_inverse_round_trip(self):
        t = Rigid2D(theta=0.7, translation=np.array([3.0, -2.0]))
        pts = np.array([[1.0, 1.0], [-4.0, 2.0], [0.0, 0.0]])
        back = t.inverse().apply_points(t.apply_points(pts))
        np.testing.assert_allclose(back, pts, atol=1e-9)

    def test_compose_matches_sequential_apply(self):
        a = Rigid2D(theta=0.3, translation=np.array([1.0, 0.0]))
        b = Rigid2D(theta=-0.8, translation=np.array([0.0, 2.0]))
        pts = np.array([[2.0, -1.0], [0.5, 0.5]])
        seq = a.apply_points(b.apply_points(pts))
        comp = a.compose(b).apply_points(pts)
        np.testing.assert_allclose(comp, seq, atol=1e-9)

    def test_heading_wrap(self):
        # 2.5π wraps to 0.5π (well inside the (-π, π] range, no boundary ambiguity).
        assert math.isclose(wrap_to_pi(2.5 * math.pi), 0.5 * math.pi, abs_tol=1e-9)
        t = Rigid2D(theta=math.pi, translation=np.zeros(2))
        # +pi added to +pi/2 -> 3pi/2 -> wraps to -pi/2.
        assert math.isclose(t.apply_heading(math.pi / 2), -math.pi / 2, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# ego
# --------------------------------------------------------------------------- #
class TestBuildStateArray:
    def test_layout_and_values(self):
        ts = np.array([0, 100_000], dtype=np.uint64)
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        sample = TrajectorySample.from_arrays(ts, pos, np.zeros(2))
        states = build_state_array(sample)
        assert states.shape == (2, STATE_SIZE)
        np.testing.assert_allclose(states[:, StateIndex.X], pos[:, 0])
        np.testing.assert_allclose(states[:, StateIndex.Y], pos[:, 1])
        # Steering/angular columns remain zero.
        np.testing.assert_allclose(states[:, StateIndex.STEERING_ANGLE :], 0.0)

    def test_exclude_acceleration(self):
        ts = np.array([0, 100_000, 200_000], dtype=np.uint64)
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        sample = TrajectorySample.from_arrays(ts, pos, np.zeros(3))
        states = build_state_array(sample, include_acceleration=False)
        np.testing.assert_allclose(states[:, StateIndex.ACCELERATION_X], 0.0)
        np.testing.assert_allclose(states[:, StateIndex.ACCELERATION_Y], 0.0)

    def test_stack_proposals_shape(self):
        a = np.zeros((5, STATE_SIZE))
        b = np.ones((5, STATE_SIZE))
        stacked = stack_proposals(a, b)
        assert stacked.shape == (2, 5, STATE_SIZE)

    def test_stack_proposals_length_mismatch(self):
        with pytest.raises(ValueError):
            stack_proposals(np.zeros((5, STATE_SIZE)), np.zeros((4, STATE_SIZE)))


# --------------------------------------------------------------------------- #
# tracked_objects (box extraction only — pure)
# --------------------------------------------------------------------------- #
class TestExtractActorBoxes:
    def test_ego_is_excluded(self):
        grid = make_time_grid(1_000_000, 1_500_000, 10.0)
        actors = {
            EGO_ID: straight_line_renderable(),
            "agent_0": straight_line_renderable(heading=0.0),
        }
        frames = extract_actor_boxes(actors, grid)
        assert len(frames) == len(grid)
        # Only the non-ego agent should be present.
        for boxes in frames:
            for box in boxes:
                assert box.token != EGO_ID

    def test_boxes_track_positions_and_extent(self):
        grid = make_time_grid(1_000_000, 1_500_000, 10.0)
        actors = {"agent_0": straight_line_renderable(speed_mps=10.0)}
        frames = extract_actor_boxes(actors, grid)
        # A box present at every in-range frame.
        present = [b for boxes in frames for b in boxes]
        assert len(present) == len(grid)
        assert present[0].length == pytest.approx(4.5)
        assert present[0].width == pytest.approx(2.0)

    def test_out_of_range_actor_absent(self):
        grid = make_time_grid(10_000_000, 10_500_000, 10.0)
        # Actor lives at 1.0–2.0 s, grid is at 10 s -> no boxes.
        actors = {"agent_0": straight_line_renderable()}
        frames = extract_actor_boxes(actors, grid)
        assert all(len(boxes) == 0 for boxes in frames)

    def test_transform_applied_to_boxes(self):
        grid = make_time_grid(1_000_000, 1_000_000, 10.0)
        actors = {"agent_0": straight_line_renderable(n=11, speed_mps=0.0)}
        shift = Rigid2D(theta=0.0, translation=np.array([100.0, 50.0]))
        frames = extract_actor_boxes(actors, grid, transform=shift)
        box = frames[0][0]
        assert box.x == pytest.approx(100.0, abs=1e-6)
        assert box.y == pytest.approx(50.0, abs=1e-6)
