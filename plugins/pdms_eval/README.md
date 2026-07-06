# alpasim-pdms — NAVSIM-style PDM scoring for Alpasim rollouts

This plugin computes NAVSIM-style **PDM sub-scores** (no-at-fault collision,
drivable-area compliance, ego progress, time-to-collision, comfort, driving
direction compliance, lane keeping) and the aggregate **PDMS** for Alpasim
rollouts, and exposes them through the standard `alpasim.scorers` plugin hook so
the numbers flow into the existing `metrics_results.txt` / aggregation outputs
alongside the built-in scene score.

The initial target is the **navtest** split, for which NAVSIM metric caches can
be precomputed offline — the scorer reads those caches (via a scene-id manifest)
so no live nuPlan map API or map-frame transform derivation is required at
scoring time.

> **These numbers are not comparable to the NAVSIM leaderboard.** Alpasim runs a
> reactive closed-loop sim over a different horizon, has no traffic-light
> channel, and (by default) measures progress relative to a reference
> trajectory. Use PDMS here for *relative* policy comparison on navtest scenes.

## How it fits together

```
EvalConfig ──▶ PDMSScorer (alpasim.scorers: "pdms")
                  │
                  ├─ manifest.py         scene_id ─▶ NAVSIM metric-cache path
                  ├─ adapter/            SimulationResult ─▶ PDM scorer inputs
                  │    ├─ resample.py    10 Hz grid + body-frame kinematics   (pure)
                  │    ├─ frame.py       local ENU ⇄ nuPlan map (anchor)      (pure)
                  │    ├─ ego.py         PDM state array layout               (pure)
                  │    ├─ tracked_objects.py  actor boxes ─▶ DetectionsTracks
                  │    └─ pdm_inputs.py  orchestration (lazy nuplan imports)
                  └─ vendored/pdm_planner/    WorldEngine PDM scoring path
```

* The **pure** adapter helpers (`resample`, `frame`, `ego`, and box extraction in
  `tracked_objects`) are numpy-only and fully unit-tested without the nuplan
  stack.
* Everything that touches `nuplan`/`navsim` or the vendored scorer is imported
  **lazily**, so importing the plugin is cheap and it always registers — even
  when the nuplan stack is not installed (in which case the scorer degrades to a
  logged no-op instead of breaking the eval run).

## Installation

The plugin is a uv workspace member (via the `plugins/*` glob). Install it
alongside the core packages:

```bash
uv sync --extra all --extra pdms
uv run alpasim-info    # should list  Scorers: ..., pdms  and  Tools: ..., pdms-manifest
```

### Installing the nuplan stack

The default `pdms` extra deliberately **does not** pull `nuplan-devkit` /
`navsim`: the devkit is old and its pinned dependencies conflict with the modern
workspace, so forcing it into `--extra all` would break unrelated installs. Only
the actual *scoring* run needs it. Install it into the same environment
separately, e.g.:

```bash
uv pip install -e "plugins/pdms_eval[nuplan]"
# or, if the pinned deps fight the workspace, install with --no-deps and add the
# handful of real deps (numpy, shapely 2.x, pandas, geopandas) by hand — only
# nuplan.common.actor_state, nuplan.common.maps and a few nuplan.planning utils
# are imported (see alpasim_pdms/vendored/NOTICE).
```

You also need a **maps-only** nuPlan download for the TTC intersection test
(`nuplan-maps-v1.0`); point `pdms.nuplan_map_root` at it (or rely on a map
carried by the metric cache).

## Preparing metric caches (navtest)

Precompute NAVSIM metric caches once with navsim's `run_metric_caching.py`
(outside this repo), then build a manifest:

```bash
uv run alpasim-pdms-manifest \
    --metric-cache-dir /data/navtest/metric_cache \
    --output          /data/navtest/pdms_manifest.json
```

The cache directory is expected to be laid out as
`<metric_cache_dir>/<token>/metric_cache.pkl` (navsim default); the Alpasim
`scene_id` is used directly as the token. The scorer can also read the by-token
directory without a manifest (`pdms.metric_cache_dir`).

## Running eval with PDMS

Compose the shipped overlay onto an eval run and point it at your data:

```bash
uv run alpasim-eval ... +pdms=navtest \
    +eval.plugin_scorer_configs.pdms.metric_cache_dir=/data/navtest/metric_cache \
    +eval.plugin_scorer_configs.pdms.nuplan_map_root=/data/nuplan/maps
```

`+pdms=navtest` enables the `pdms` plugin scorer (via
`eval.enabled_plugin_scorers`) and populates
`eval.plugin_scorer_configs.pdms`. Emitted metrics are named
`pdms/no_at_fault_collision`, `pdms/drivable_area_compliance`,
`pdms/ego_progress`, `pdms/ttc_within_bound`, `pdms/comfort`,
`pdms/driving_direction_compliance`, `pdms/lane_keeping`, `pdms/score`, plus a
`pdms/first_violation_timestamp` diagnostic. Each metric's `info` field records
the progress reference used and the anchor-alignment gap.

## Configuration

All keys live under `eval.plugin_scorer_configs.pdms` (see
`alpasim_pdms/config.py::PDMSConfig` for the authoritative list and defaults):

| Key | Meaning |
|---|---|
| `enabled` | Master switch for the computation. |
| `metric_cache_dir` / `manifest_path` | Scene → metric-cache resolution. |
| `nuplan_map_root` | Maps-only nuPlan root for the TTC intersection test. |
| `progress_reference` | `metric_cache` (canonical NAVSIM EP; falls back to `gt`) or `gt`. |
| `ego_trajectory_source` | `driven` (achieved states) or `driver_estimated`. |
| `mode` | `full_rollout` (shipped) or `windowed` (planned). |
| `human_penalty_filter` | Un-penalize the policy for gates the GT rollout also fails. Default on. |
| `proposal_sampling_*`, `resample_hz`, `window_s` | Sampling/horizon. |
| `anchor_tolerance_m` | Max allowed rollout↔cache anchor gap before warning. |

## Deviations from NAVSIM (deliberate)

* **No re-simulation.** NAVSIM re-simulates predicted trajectories through an LQR
  tracker before scoring because it scores open-loop predictions. Alpasim
  rollouts are already closed-loop and dynamically feasible, so we score the
  achieved states directly (matching WorldEngine's direct state-array
  injection).
* **Reference / progress.** By default proposal 0 is the cached PDM-closed
  trajectory (canonical EP), falling back to the recorded GT when the cache
  lacks it. The used reference is reported per scene.
* **Observation from the rollout.** Background agents come from Alpasim's
  trafficsim and diverge from the log, so the observation is built from the
  rollout's *actual* actor tracks — never the cache's detection tracks. The
  cache's centerline / route / drivable-area map remain valid.
* **No traffic lights.** Alpasim has no traffic-light channel, so per-frame
  traffic-light lists are empty (mainly affects intersections).
* **Human-penalty filter.** On original scenes the policy is un-penalized for any
  binary gate the human trajectory also fails — robust against MTGS/navtest map
  annotation noise. Scenes where GT itself fails NC/DAC are flagged as candidate
  manifest/frame-transform bugs.

## Why only the scoring path is vendored

`alpasim_pdms/vendored/pdm_planner/` contains **only** the modules needed to
*score* already-simulated ego states (scorer, comfort metrics, scorer utils,
observation + occupancy map, and the geometry/array/path/enum utilities they
import). The PDM-Closed planner, the LQR/kinematic simulator, IDM proposal
generation and the emergency-brake logic are intentionally omitted — in Alpasim
the scorer evaluates the achieved closed-loop states and never needs to plan or
re-simulate.

The subtree is produced reproducibly by
`python -m alpasim_pdms.tools.vendor_pdm --worldengine-root ./WorldEngine`, which
copies the files and rewrites the internal import prefix. See
`alpasim_pdms/vendored/NOTICE` for provenance (WorldEngine → NAVSIM /
tuplan_garage, all Apache-2.0).

> **Note on metric set.** The vendored WorldEngine scorer implements the classic
> PDMS aggregate `NC × DAC × Σ(wᵢ·subscoreᵢ)/Σwᵢ` with weights
> `{progress:5, ttc:5, comfort:2}` and no traffic-light gate. Targeting NAVSIM v2
> **EPDMS** (adds a traffic-light gate, lane-keeping and two-frame comfort) is a
> drop-in future step: re-vendor the scorer from `autonomousvision/navsim` `main`
> and widen the metric readout — the adapter is unchanged.

## Testing

```bash
uv run pytest plugins/pdms_eval/tests -v
```

The suite covers entry-point registration, config resolution, the pure adapter
helpers on synthetic trajectories, manifest resolution, the scorer's aggregate /
human-penalty-filter math, and graceful degradation when the nuplan stack is
absent. The end-to-end golden test against navsim's own `pdm_score.py` requires
the nuplan stack + a real metric cache and is documented but skipped by default.
