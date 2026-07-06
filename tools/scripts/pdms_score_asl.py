#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Option B: score an existing Alpasim rollout with the PDMS plugin, standalone.

Loads a previous run's ASL log(s) into a ``SimulationResult`` (via the same
``asl_loader`` the in-container eval uses), enables *only* the ``pdms`` plugin
scorer, points it at a NAVSIM metric-cache manifest, and prints the resulting
``pdms/*`` sub-scores — no Docker, no re-simulation.

Usage:
    python pdms_score_asl.py <run_dir> --manifest <manifest.json> [--scene <id>]

``run_dir`` is a wizard run directory containing ``eval-config.yaml`` and
``rollouts/**/rollout.asl``.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import os
from pathlib import Path

import polars as pl
from alpasim_utils.yaml_utils import typed_parse_config
from omegaconf import OmegaConf

from eval.asl_loader import load_scenario_eval_input_from_asl
from eval.scenario_evaluator import ScenarioEvaluator
from eval.schema import EvalConfig

logger = logging.getLogger("pdms_score_asl")


def build_cfg(run_dir: Path, manifest: str, nuplan_map_root: str | None,
              nuplan_configs_root: str | None) -> EvalConfig:
    """Load the run's eval-config.yaml and overlay the pdms plugin settings."""
    cfg = typed_parse_config(str(run_dir / "eval-config.yaml"), EvalConfig)
    # Run only the pdms plugin scorer (built-ins still run but are harmless;
    # ScorerGroup isolates per-scorer failures such as missing vec_map).
    cfg.enabled_plugin_scorers = ["pdms"]
    pdms = {
        "enabled": True,
        "manifest_path": manifest,
        "metric_prefix": "pdms",
        "progress_reference": "metric_cache",
        "ego_trajectory_source": "driven",
        "mode": "full_rollout",
        "metric": "epdms",
        "human_penalty_filter": True,
    }
    if nuplan_map_root:
        pdms["nuplan_map_root"] = nuplan_map_root
    if nuplan_configs_root:
        pdms["nuplan_configs_root"] = nuplan_configs_root
    cfg.plugin_scorer_configs = OmegaConf.create({"pdms": pdms})
    cfg.video.render_video = False
    return cfg


def score_asl(asl_path: str, cfg: EvalConfig):
    scenario_input = asyncio.run(
        load_scenario_eval_input_from_asl(
            asl_path,
            cfg,
            artifacts={},  # PDMS gets its map from the metric cache
            run_metadata={"run_uuid": "pdms-standalone", "run_name": "pdms_score_asl"},
        )
    )
    return ScenarioEvaluator(cfg).evaluate(scenario_input)


def _print_pdms(result, scene_id: str) -> None:
    """Print the pdms/* aggregated sub-scores from a ScenarioEvalResult."""
    agg = result.aggregated_metrics or {}
    pdms = {k: v for k, v in agg.items() if k.startswith("pdms/")}
    print(f"\n=== PDMS for {scene_id} ===")
    if not pdms:
        # Fall back to scanning the long-format metrics_df 'name' column.
        df = result.metrics_df
        if df is not None and "name" in df.columns:
            names = [n for n in df["name"].to_list() if str(n).startswith("pdms/")]
            if names:
                print("  (found pdms rows in metrics_df but no aggregate):")
                for n in sorted(set(names)):
                    print(f"    {n}")
                return
        print("  (no pdms/* metrics produced — check warnings above)")
        return
    for k in sorted(pdms):
        print(f"  {k:40s}: {pdms[k]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--manifest", required=True, help="scene_id -> cache JSON manifest.")
    ap.add_argument("--scene", default=None, help="Only score this scene_id.")
    ap.add_argument("--nuplan-map-root", default=None, help="Maps-only nuPlan root (TTC).")
    ap.add_argument(
        "--nuplan-configs-root",
        default=None,
        help="Dir of nuplan-track <scene_id>.yaml configs (for per-scene city).",
    )
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = build_cfg(args.run_dir, args.manifest, args.nuplan_map_root,
                    args.nuplan_configs_root)

    asls = sorted(glob.glob(str(args.run_dir / "rollouts" / "**" / "*.asl"), recursive=True))
    if not asls:
        print(f"No ASL files under {args.run_dir}/rollouts")
        return 1
    print(f"Found {len(asls)} ASL file(s).")

    rc = 0
    for asl in asls:
        scene_id = os.path.basename(os.path.dirname(os.path.dirname(asl)))
        if args.scene and args.scene not in scene_id:
            continue
        print(f"\n--- scoring {scene_id} ({asl}) ---")
        try:
            result = score_asl(asl, cfg)
            _print_pdms(result, scene_id)
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
