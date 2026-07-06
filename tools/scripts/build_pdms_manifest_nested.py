#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a PDMS scene_id -> metric-cache manifest for the *nested* nuPlan-devkit
cache layout, and probe what stack a cache pickle needs.

The alpasim-pdms plugin's built-in manifest builder assumes the flat NAVSIM
layout ``<dir>/<token>/metric_cache.pkl`` where ``token == scene_id``.  The
cache at /data/sungyeonpark/nuplan/metric_cache instead uses the nuPlan devkit
layout::

    <metric_cache_dir>/<log_segment>/unknown/<token_hash>/metric_cache.pkl

and the alpasim navtest scene_id is ``<log_segment>-<token_hash>``.  This script
walks that structure and writes ``{scene_id: cache_path}`` JSON that the plugin
can consume via ``pdms.manifest_path``.

Usage:
    python build_pdms_manifest_nested.py <metric_cache_dir> <output.json> \
        [--probe]
"""
from __future__ import annotations

import argparse
import json
import pickletools
import sys
from pathlib import Path

CACHE_BASENAME = "metric_cache.pkl"


def build(metric_cache_dir: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    # <cache_dir>/<log_segment>/unknown/<token_hash>/metric_cache.pkl
    for pkl in metric_cache_dir.glob(f"*/unknown/*/{CACHE_BASENAME}"):
        token_hash = pkl.parent.name
        log_segment = pkl.parent.parent.parent.name
        scene_id = f"{log_segment}-{token_hash}"
        entries[scene_id] = str(pkl.resolve())
    return dict(sorted(entries.items()))


def _decompress_maybe(raw: bytes) -> bytes:
    """Transparently handle gzip / xz(lzma) / zstd wrapped pickles."""
    if raw[:2] == b"\x1f\x8b":  # gzip
        import gzip

        return gzip.decompress(raw)
    if raw[:6] == b"\xfd7zXZ\x00" or raw[:1] == b"\xfd":  # xz / lzma
        import lzma

        return lzma.decompress(raw)
    if raw[:4] == b"\x28\xb5\x2f\xfd":  # zstd
        import zstandard  # type: ignore

        return zstandard.ZstdDecompressor().decompress(raw)
    return raw


def probe_stack(pkl_path: Path) -> tuple[set[str], str]:
    """Return (modules referenced, detected compression) without importing."""
    raw = pkl_path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        comp = "gzip"
    elif raw[:1] == b"\xfd":
        comp = "xz/lzma"
    elif raw[:4] == b"\x28\xb5\x2f\xfd":
        comp = "zstd"
    else:
        comp = "none"
    data = _decompress_maybe(raw)
    modules: set[str] = set()
    import io

    for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
        if op.name in ("GLOBAL", "STACK_GLOBAL") and isinstance(arg, str):
            modules.add(arg.split(" ")[0].split(".")[0])
        elif op.name in ("SHORT_BINUNICODE", "BINUNICODE") and isinstance(arg, str):
            if arg.startswith(("nuplan", "navsim")):
                modules.add(arg.split(".")[0])
    return modules, comp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metric_cache_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument(
        "--probe",
        action="store_true",
        help="Also report the module stack referenced by the first cache pickle.",
    )
    ap.add_argument(
        "--verify",
        nargs="*",
        default=[],
        help="Scene ids expected to resolve (asserts they are present).",
    )
    args = ap.parse_args(argv)

    if not args.metric_cache_dir.is_dir():
        print(f"ERROR: not a dir: {args.metric_cache_dir}", file=sys.stderr)
        return 2

    entries = build(args.metric_cache_dir)
    print(f"Discovered {len(entries)} nested metric caches.")
    if entries:
        first_id, first_path = next(iter(entries.items()))
        print(f"  example: {first_id}\n        -> {first_path}")

    missing = [s for s in args.verify if s not in entries]
    for s in args.verify:
        status = "OK" if s in entries else "MISSING"
        print(f"  verify {s}: {status}")
    if missing:
        print(f"ERROR: {len(missing)} verify scene(s) missing", file=sys.stderr)

    if args.probe and entries:
        probe_path = Path(next(iter(entries.values())))
        try:
            mods, comp = probe_stack(probe_path)
            rel = sorted(m for m in mods if m in ("nuplan", "navsim"))
            print(f"  probe {probe_path.name}: compression={comp}")
            print(f"  probe references: {sorted(mods)}")
            print(f"  -> required heavy stacks: {rel or 'none detected'}")
        except Exception as e:  # noqa: BLE001
            print(f"  probe failed: {e}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2, sort_keys=True))
    print(f"Wrote manifest with {len(entries)} entries to {args.output}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
