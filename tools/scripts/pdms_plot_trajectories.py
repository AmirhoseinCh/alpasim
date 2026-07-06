#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot GT vs planner (policy) vs PDM-closed reference for one Alpasim scene.

Reuses the PDMS adapter (``build_inputs_from_cache``) so the three trajectories
are exactly what the scorer sees, all mapped into the common nuPlan frame:

* **PDM**    – the PDM-closed reference trajectory from the NAVSIM metric cache
               (``states[0]``).
* **planner**– the driven (policy) rollout trajectory (``states[policy_index]``).
* **GT**     – the recorded expert trajectory (``states[gt_index]``).

Also overlays the cache route centerline for context and marks the shared start.

Usage:
    python pdms_plot_trajectories.py <run_dir> --scene <id> \
        --manifest <manifest.json> \
        --nuplan-map-root <maps> --nuplan-configs-root <configs> \
        --out <out.png>
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from alpasim_utils.yaml_utils import typed_parse_config
from omegaconf import OmegaConf

from alpasim_pdms.adapter.ego import StateIndex
from alpasim_pdms.adapter.pdm_inputs import build_inputs_from_cache
from alpasim_pdms.config import PDMSConfig
from alpasim_pdms.manifest import SceneManifest, load_metric_cache
from eval.asl_loader import load_scenario_eval_input_from_asl
from eval.data import SimulationResult
from eval.schema import EvalConfig

logger = logging.getLogger("pdms_plot")

X, Y = StateIndex.X, StateIndex.Y


def _pdms_config(manifest, map_root, configs_root) -> PDMSConfig:
    return PDMSConfig(
        enabled=True,
        manifest_path=manifest,
        nuplan_map_root=map_root,
        nuplan_configs_root=configs_root,
        progress_reference="metric_cache",
        human_penalty_filter=True,
    )


def _load_sim_result(asl_path: str, eval_cfg: EvalConfig) -> SimulationResult:
    scenario_input = asyncio.run(
        load_scenario_eval_input_from_asl(
            asl_path,
            eval_cfg,
            artifacts={},
            run_metadata={"run_uuid": "pdms-plot", "run_name": "pdms_plot"},
        )
    )
    return SimulationResult.from_scenario_input(scenario_input, eval_cfg)


def _centerline_xy(inputs) -> np.ndarray | None:
    cl = getattr(inputs, "centerline", None)
    if cl is None:
        return None
    try:
        pts = cl.discrete_path  # List[StateSE2]
        return np.array([[p.x, p.y] for p in pts], dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--nuplan-map-root", required=True)
    ap.add_argument("--nuplan-configs-root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--log-level", default="ERROR")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR))

    eval_cfg = typed_parse_config(str(args.run_dir / "eval-config.yaml"), EvalConfig)
    eval_cfg.enabled_plugin_scorers = []  # we call the adapter directly
    config = _pdms_config(args.manifest, args.nuplan_map_root, args.nuplan_configs_root)

    asls = glob.glob(
        str(args.run_dir / "rollouts" / f"*{args.scene}*" / "**" / "*.asl"),
        recursive=True,
    )
    if not asls:
        print(f"No ASL for scene {args.scene} under {args.run_dir}")
        return 1
    asl = asls[0]
    print(f"Scene: {args.scene}\nASL:   {asl}")

    sim_result = _load_sim_result(asl, eval_cfg)
    manifest = SceneManifest.resolve(config.manifest_path, config.metric_cache_dir)
    cache_path = manifest.get(sim_result.session_metadata.scene_id)
    if cache_path is None:
        print(f"No metric cache for {sim_result.session_metadata.scene_id}")
        return 1
    metric_cache = load_metric_cache(str(cache_path))
    inputs = build_inputs_from_cache(sim_result, metric_cache, config)

    states = np.asarray(inputs.states)  # [P, T, 11]
    pdm = states[0][:, [X, Y]]
    planner = states[inputs.policy_index][:, [X, Y]]
    gt = states[inputs.gt_index][:, [X, Y]] if inputs.gt_index is not None else None

    fig, ax = plt.subplots(figsize=(9, 9))
    cl = _centerline_xy(inputs)
    if cl is not None and len(cl):
        ax.plot(cl[:, 0], cl[:, 1], color="0.8", lw=6, solid_capstyle="round",
                label="route centerline", zorder=1)
    if gt is not None:
        ax.plot(gt[:, 0], gt[:, 1], "-o", color="tab:green", ms=3, lw=2,
                label="GT (expert)", zorder=3)
    ax.plot(pdm[:, 0], pdm[:, 1], "-s", color="tab:blue", ms=3, lw=2,
            label="PDM-closed (cache ref)", zorder=4)
    ax.plot(planner[:, 0], planner[:, 1], "-^", color="tab:red", ms=3, lw=2,
            label="planner (policy)", zorder=5)
    # shared start
    ax.scatter([planner[0, 0]], [planner[0, 1]], c="black", s=80, marker="*",
               zorder=6, label="start")

    # Focus the view on the (short) GT/PDM/planner trajectories, not the full
    # route centerline: a square bbox around those three, plus a margin.
    focus = [pdm, planner] + ([gt] if gt is not None else [])
    pts = np.concatenate(focus, axis=0)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    half = max(xmax - xmin, ymax - ymin) / 2.0
    half = half * 1.15 + 3.0  # 15% + 3 m padding
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    ax.set_title(f"PDMS trajectories — {args.scene}\n"
                 f"anchor gap {inputs.anchor_gap_m:.3f} m")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)

    out = args.out or str(args.run_dir / f"pdms_traj_{args.scene}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved plot -> {out}")
    # brief numeric summary
    def _len(a):
        return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))
    print(f"path lengths [m]: planner={_len(planner):.1f}  "
          f"PDM={_len(pdm):.1f}" + (f"  GT={_len(gt):.1f}" if gt is not None else ""))
    if gt is not None:
        print(f"final planner–GT gap: {np.linalg.norm(planner[-1]-gt[-1]):.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
