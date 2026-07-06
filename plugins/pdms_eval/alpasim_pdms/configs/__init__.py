# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Hydra configuration overlays for the PDMS eval plugin.

The wizard discovers this package via the ``alpasim.configs`` entry point and
adds ``pkg://alpasim_pdms.configs`` to Hydra's search path, so the overlays here
can be selected on the command line (e.g. ``+pdms=navtest``).
"""
