#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Feasibility test: can the isolated env import the classes the metric cache
needs and actually deserialize one NAVSIM metric cache?

Self-contained (no alpasim_pdms dependency) so it runs in the throwaway
nuplan/navsim venv.  Reports, in order:

1. Which of the 16 required nuplan/navsim classes import (and which fail).
2. Whether ``MetricCache.load`` and/or a raw lzma+pickle load succeeds.
3. A short summary of the deserialized cache (types + key attributes).

Usage:
    python pdms_load_cache_test.py <manifest.json> [scene_id]
"""
from __future__ import annotations

import importlib
import io
import json
import lzma
import pickle
import sys
from pathlib import Path

REQUIRED = [
    # nuplan
    "nuplan.common.actor_state.agent:Agent",
    "nuplan.common.actor_state.car_footprint:CarFootprint",
    "nuplan.common.actor_state.dynamic_car_state:DynamicCarState",
    "nuplan.common.actor_state.ego_state:EgoState",
    "nuplan.common.actor_state.oriented_box:OrientedBox",
    "nuplan.common.actor_state.scene_object:SceneObjectMetadata",
    "nuplan.common.actor_state.state_representation:StateSE2",
    "nuplan.common.actor_state.static_object:StaticObject",
    "nuplan.common.actor_state.tracked_objects_types:TrackedObjectType",
    "nuplan.common.actor_state.vehicle_parameters:VehicleParameters",
    "nuplan.common.maps.maps_datatypes:SemanticMapLayer",
    "nuplan.planning.simulation.trajectory.interpolated_trajectory:InterpolatedTrajectory",
    # navsim
    "navsim.planning.metric_caching.metric_cache:MetricCache",
    "navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation:PDMObservation",
    "navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map:PDMOccupancyMap",
    "navsim.planning.simulation.planner.pdm_planner.utils.pdm_path:PDMPath",
]


def check_imports() -> tuple[int, int]:
    ok = fail = 0
    print("=== Import check (16 required classes) ===")
    for spec in REQUIRED:
        mod, cls = spec.split(":")
        try:
            m = importlib.import_module(mod)
            getattr(m, cls)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  FAIL {spec}\n       {type(e).__name__}: {str(e)[:180]}")
    print(f"  -> {ok}/{len(REQUIRED)} importable, {fail} failed")
    return ok, fail


def load_cache(path: Path):
    # 1) navsim's own loader (handles the on-disk format + class layout).
    try:
        from navsim.planning.metric_caching.metric_cache import MetricCache  # type: ignore

        if hasattr(MetricCache, "load"):
            print("  trying MetricCache.load() ...")
            return MetricCache.load(path), "MetricCache.load"
    except Exception as e:  # noqa: BLE001
        print(f"  MetricCache.load unavailable/failed: {type(e).__name__}: {str(e)[:160]}")
    # 2) raw lzma + pickle.
    print("  trying raw lzma+pickle ...")
    raw = path.read_bytes()
    data = lzma.decompress(raw) if raw[:1] == b"\xfd" else raw
    return pickle.load(io.BytesIO(data)), "lzma+pickle"


def summarize(obj) -> None:
    print("=== Loaded cache summary ===")
    print(f"  top-level type: {type(obj).__module__}.{type(obj).__name__}")
    attrs = [a for a in dir(obj) if not a.startswith("_")]
    interesting = [
        a
        for a in attrs
        if any(k in a.lower() for k in ("traj", "route", "map", "ego", "centerline", "drivable", "token", "observation"))
    ]
    for a in interesting[:20]:
        try:
            v = getattr(obj, a)
            if callable(v):
                continue
            print(f"  .{a}: {type(v).__module__}.{type(v).__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  .{a}: <err {type(e).__name__}>")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: pdms_load_cache_test.py <manifest.json> [scene_id]", file=sys.stderr)
        return 2
    manifest = json.loads(Path(argv[0]).read_text())
    scene_id = argv[1] if len(argv) > 1 else next(iter(manifest))
    path = Path(manifest[scene_id])
    print(f"Scene: {scene_id}\nCache: {path}\n")

    ok, fail = check_imports()
    print()
    try:
        obj, how = load_cache(path)
        print(f"  LOADED via {how}\n")
        summarize(obj)
        print("\nRESULT: PDMS cache load FEASIBLE in this environment. ✅")
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nRESULT: cache load NOT feasible yet — see error above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
