# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""NAVSIM-style PDM scoring for Alpasim rollouts (eval plugin).

This package computes NAVSIM-style PDM sub-scores (no-at-fault collision,
drivable-area compliance, ego progress, time-to-collision, comfort, driving
direction compliance, lane keeping) and the aggregate PDMS/EPDMS for Alpasim
rollouts, and exposes them as an ``alpasim.scorers`` plugin.

The initial target is the ``navtest`` split, for which NAVSIM metric caches can
be precomputed offline; the scorer reads those caches (via a scene-id manifest)
so that no live nuPlan map API or map-frame transform derivation is required at
scoring time.  See ``README.md`` for the design and limitations.

Importing this package is cheap and never imports ``nuplan``/``navsim``: those
heavy dependencies are imported lazily inside the scorer's scoring path, so the
plugin always registers and its pure-python adapter helpers remain usable even
when the nuplan stack is not installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "PDMSScorer", "PDMSConfig"]


def __getattr__(name: str):
    # Lazily expose the scorer/config without importing them (and their
    # transitive deps) at package import time.
    if name == "PDMSScorer":
        from alpasim_pdms.scorer import PDMSScorer

        return PDMSScorer
    if name == "PDMSConfig":
        from alpasim_pdms.config import PDMSConfig

        return PDMSConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
