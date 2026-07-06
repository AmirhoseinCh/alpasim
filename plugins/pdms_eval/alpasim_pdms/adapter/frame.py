# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""2D rigid transform between Alpasim ``local`` ENU and the nuPlan map frame.

Alpasim's ``local`` frame is a per-scenario ENU frame (see ``CONTRIBUTING.md``);
nuPlan maps live in a global, city-fixed frame.  For navtest scenes the two are
related by a planar rigid transform that we recover from a single *anchor*
correspondence: the ego pose at policy engagement, known both in ``local``
(from the rollout) and in the nuPlan frame (from the metric cache's anchor ego
state).  This sidesteps deriving the transform from artifact metadata.

This module is pure numpy and unit-tested (round-trip / composition) without the
nuplan stack.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to ``(-pi, pi]``."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _rot_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class Rigid2D:
    """Planar rigid transform ``p_dst = R(theta) @ p_src + t``.

    Rotation is applied first, then translation.  ``theta`` also rotates
    headings additively.
    """

    theta: float
    translation: np.ndarray  # (2,)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "translation", np.asarray(self.translation, dtype=np.float64)
        )

    @property
    def rotation(self) -> np.ndarray:
        return _rot_matrix(self.theta)

    @classmethod
    def identity(cls) -> "Rigid2D":
        return cls(theta=0.0, translation=np.zeros(2))

    @classmethod
    def from_anchor(
        cls,
        src_pose_xytheta: tuple[float, float, float] | np.ndarray,
        dst_pose_xytheta: tuple[float, float, float] | np.ndarray,
    ) -> "Rigid2D":
        """Recover the transform mapping ``src`` pose onto ``dst`` pose.

        Both poses are ``(x, y, heading)``; the returned transform maps the
        source frame to the destination frame such that it takes the source
        anchor exactly onto the destination anchor.
        """
        src = np.asarray(src_pose_xytheta, dtype=np.float64)
        dst = np.asarray(dst_pose_xytheta, dtype=np.float64)
        theta = float(wrap_to_pi(dst[2] - src[2]))
        translation = dst[:2] - _rot_matrix(theta) @ src[:2]
        return cls(theta=theta, translation=translation)

    def apply_points(self, points_xy: np.ndarray) -> np.ndarray:
        """Transform an ``(N, 2)`` array of points."""
        points_xy = np.asarray(points_xy, dtype=np.float64)
        return points_xy @ self.rotation.T + self.translation

    def apply_point(self, point_xy: np.ndarray) -> np.ndarray:
        return self.apply_points(np.asarray(point_xy).reshape(1, 2))[0]

    def apply_headings(self, headings: np.ndarray) -> np.ndarray:
        """Rotate an array of headings."""
        return wrap_to_pi(np.asarray(headings, dtype=np.float64) + self.theta)

    def apply_heading(self, heading: float) -> float:
        return float(wrap_to_pi(heading + self.theta))

    def apply_poses(
        self, positions_xy: np.ndarray, headings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform ``(N, 2)`` positions and ``(N,)`` headings together."""
        return self.apply_points(positions_xy), self.apply_headings(headings)

    def inverse(self) -> "Rigid2D":
        inv_theta = -self.theta
        inv_translation = -_rot_matrix(inv_theta) @ self.translation
        return Rigid2D(theta=float(inv_theta), translation=inv_translation)

    def compose(self, other: "Rigid2D") -> "Rigid2D":
        """Return ``self ∘ other`` (apply ``other`` first, then ``self``)."""
        theta = float(wrap_to_pi(self.theta + other.theta))
        translation = self.rotation @ other.translation + self.translation
        return Rigid2D(theta=theta, translation=translation)


def anchor_gap_m(
    src_pose_xytheta: np.ndarray, dst_pose_xytheta: np.ndarray, transform: Rigid2D
) -> float:
    """Residual translation error (m) after applying ``transform`` to ``src``.

    Used to validate the frame alignment: with a correct anchor this is ~0 by
    construction, so a large value signals a manifest/frame-transform bug.
    """
    mapped = transform.apply_point(np.asarray(src_pose_xytheta)[:2])
    return float(np.linalg.norm(mapped - np.asarray(dst_pose_xytheta)[:2]))
