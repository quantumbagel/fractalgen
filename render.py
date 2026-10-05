#!/usr/bin/env python3
"""Parallel ultra-quality wallpaper CLI.

Imports RECIPES from generate.py; does not modify that file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bg.engine import (
    PRESETS,
    ROOT,
    default_format,
    default_jobs,
    default_out_dir,
    render_batch,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render math wallpapers in parallel at 4K / 8K / 16K (desktop + iPhone).",
    )
    p.add_argument(
        "--preset",
        choices=list(PRESETS),
        help="Size pack: desktop-4k/8k/16k or iphone / iphone-8k / iphone-16k",
    )
    p.add_argument("--width", type=int, default=None, help="Override preset width")
    p.add_argument("--height", type=int, default=None, help="Override preset height")
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Process-pool workers (default: 2 at 4K/iPhone native, 1 at 8K/16K)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip wallpapers whose output file already exists",
    )
    p.add_argument("--only", nargs="+", help="Render only these recipe names")
    p.add_argument("--out", type=Path, default=None, help="Output directory")
    p.add_argument("--list", action="store_true", help="List wallpaper names")
    p.add_argument(
        "--format",
        dest="fmt",
        choices=("png", "jpg"),
        default=None,
        help="png (default at 8K and below) or jpg quality 95 (default at 16K)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    from generate import RECIPES

    if args.list:
        for name in RECIPES:
            print(name)
        return 0

    if args.preset is None and (args.width is None or args.height is None):
        print("Need --preset, or both --width and --height.", file=sys.stderr)
        return 2

    if args.preset is not None:
        spec = PRESETS[args.preset]
        width = args.width if args.width is not None else int(spec["width"])
        height = args.height if args.height is not None else int(spec["height"])
    else:
        width = int(args.width)
        height = int(args.height)

    if args.preset is None and args.out is None:
        print("Need --out when --preset is omitted.", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out is not None else default_out_dir(args.preset, ROOT)
    jobs = args.jobs if args.jobs is not None else default_jobs(args.preset, width, height)
    fmt = args.fmt if args.fmt is not None else default_format(args.preset, width, height)

    names = list(RECIPES)
    if args.only:
        unknown = [n for n in args.only if n not in RECIPES]
        if unknown:
            print(f"Unknown: {unknown}", file=sys.stderr)
            return 2
        names = list(args.only)

    jpeg_quality = 95
    return render_batch(
        names,
        width,
        height,
        out_dir,
        jobs=jobs,
        fmt=fmt,
        skip_existing=args.skip_existing,
        jpeg_quality=jpeg_quality,
    )


if __name__ == "__main__":
    raise SystemExit(main())
