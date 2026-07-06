# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Reproducible vendoring of the WorldEngine PDM scoring path.

This script copies *only the scoring path* of WorldEngine's ``pdm_planner``
subtree into ``alpasim_pdms/vendored/pdm_planner`` and rewrites its internal
imports.  The planner/simulator/proposal generation are intentionally excluded:
in Alpasim the scorer evaluates already-simulated closed-loop states, so it does
not need PDM-Closed, the LQR tracker, the kinematic bicycle model, IDM proposals
or the emergency-brake logic (see README "Why only the scoring path").

Both WorldEngine and Alpasim are Apache-2.0, so vendoring with attribution is
clean; provenance is recorded in ``alpasim_pdms/vendored/NOTICE``.

Usage (from the repo root, after cloning WorldEngine next to this repo)::

    python -m alpasim_pdms.tools.vendor_pdm --worldengine-root ./WorldEngine

Re-run it whenever the upstream subtree is updated; keep the resulting change in
its own commit for reviewability.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("alpasim_pdms.vendor")

# Path of the pdm_planner subtree within a WorldEngine checkout.
WORLDENGINE_PDM_SUBPATH = (
    "projects/SimEngine/worldengine/components/agents/policy/pdm_planner"
)

OLD_IMPORT_PREFIX = "worldengine.components.agents.policy.pdm_planner"
NEW_IMPORT_PREFIX = "alpasim_pdms.vendored.pdm_planner"

# The minimal import closure of ``scoring/pdm_scorer.py`` (verified by scanning
# intra-package imports).  Excludes the planner/simulator/proposal modules and
# ``pdm_observation_utils`` (only used by the planner).
VENDOR_FILES = [
    "scoring/pdm_scorer.py",
    "scoring/pdm_comfort_metrics.py",
    "scoring/pdm_scorer_utils.py",
    "observation/pdm_observation.py",
    "observation/pdm_object_manager.py",
    "observation/pdm_occupancy_map.py",
    "utils/pdm_enums.py",
    "utils/pdm_array_representation.py",
    "utils/pdm_geometry_utils.py",
    "utils/pdm_path.py",
]

# Package levels that need an ``__init__.py``.
PACKAGE_DIRS = ["", "scoring", "observation", "utils"]

PROVENANCE_BANNER = (
    "# Vendored from OpenDriveLab/WorldEngine (Apache-2.0):\n"
    "#   {subpath}/{rel}\n"
    "# Imports rewritten {old} -> {new} by alpasim_pdms.tools.vendor_pdm.\n"
    "# Do not edit by hand; see alpasim_pdms/vendored/NOTICE.\n\n"
)

NOTICE_TEXT = """\
PDMS eval plugin — vendored code NOTICE
=======================================

alpasim_pdms/vendored/pdm_planner/ contains the *scoring path* of the PDM
planner vendored from:

    OpenDriveLab/WorldEngine
    projects/SimEngine/worldengine/components/agents/policy/pdm_planner/
    License: Apache-2.0

WorldEngine's PDM scorer is itself derived from NAVSIM
(autonomousvision/navsim) and tuplan_garage (autonomousvision/tuplan_garage),
both Apache-2.0.  Original copyright and license notices are retained.

Only the modules required to *score* already-simulated ego states are vendored
(scorer, comfort metrics, scorer utils, observation + occupancy map, and the
geometry/array/path/enum utilities they import).  The PDM-Closed planner, the
LQR/kinematic simulator, IDM proposal generation and the emergency-brake logic
are intentionally omitted.

The only local modification is rewriting the intra-package import prefix
``worldengine.components.agents.policy.pdm_planner`` to
``alpasim_pdms.vendored.pdm_planner`` (performed by
``alpasim_pdms/tools/vendor_pdm.py``).  Re-run that script to refresh the subtree.
"""

INIT_DOC = '"""Vendored WorldEngine PDM scoring path. See ../NOTICE."""\n'


def _default_worldengine_root() -> Path:
    # Repo root is five parents up from this file:
    # plugins/pdms_eval/alpasim_pdms/tools/vendor_pdm.py -> repo root.
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "WorldEngine"


def _vendored_root() -> Path:
    # alpasim_pdms/tools/vendor_pdm.py -> alpasim_pdms/vendored/pdm_planner
    return Path(__file__).resolve().parents[1] / "vendored" / "pdm_planner"


def vendor(worldengine_root: Path, dest_root: Path) -> int:
    """Copy + rewrite the scoring path. Returns the number of files written."""
    src_subtree = worldengine_root / WORLDENGINE_PDM_SUBPATH
    if not src_subtree.is_dir():
        raise FileNotFoundError(
            f"WorldEngine pdm_planner subtree not found at {src_subtree}. "
            "Pass --worldengine-root pointing at a WorldEngine checkout."
        )

    dest_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for rel in VENDOR_FILES:
        src = src_subtree / rel
        if not src.is_file():
            raise FileNotFoundError(f"Expected vendored source missing: {src}")
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        rewritten = text.replace(OLD_IMPORT_PREFIX, NEW_IMPORT_PREFIX)
        banner = PROVENANCE_BANNER.format(
            subpath=WORLDENGINE_PDM_SUBPATH,
            rel=rel,
            old=OLD_IMPORT_PREFIX,
            new=NEW_IMPORT_PREFIX,
        )
        dst.write_text(banner + rewritten, encoding="utf-8")
        logger.info("vendored %s", rel)
        written += 1

    for pkg in PACKAGE_DIRS:
        init_path = dest_root / pkg / "__init__.py"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.write_text(INIT_DOC, encoding="utf-8")

    notice_path = dest_root.parent / "NOTICE"
    notice_path.write_text(NOTICE_TEXT, encoding="utf-8")
    logger.info("wrote %s", notice_path)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worldengine-root",
        type=Path,
        default=_default_worldengine_root(),
        help="Path to a WorldEngine checkout (default: <repo>/WorldEngine).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=_vendored_root(),
        help="Destination pdm_planner directory (default: the vendored subtree).",
    )
    args = parser.parse_args()
    count = vendor(args.worldengine_root, args.dest)
    logger.info("Vendored %d files into %s", count, args.dest)


if __name__ == "__main__":
    main()
