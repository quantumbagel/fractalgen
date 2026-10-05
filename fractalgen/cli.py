"""Command-line interface: ``python -m fractalgen <command> ...``"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .params import parse_literal


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().replace("×", "x").split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must look like 1920x1080, got {text!r}") from None


def _kv(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    k, v = text.split("=", 1)
    return k.strip(), parse_literal(v)


def _add_override_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("overrides (applied on top of each recipe)")
    g.add_argument("--set", dest="sets", type=_kv, action="append", default=[], metavar="KEY=VALUE",
                   help="Generator parameter, e.g. --set max_iter=2000 --set c=-0.8+0.156j")
    g.add_argument("--palette", help="Palette name (append _r to reverse) or comma-separated hex stops")
    g.add_argument("--center", help="View center, e.g. -0.745+0.11j or -0.745,0.11")
    g.add_argument("--half-width", type=float, help="View horizontal half-extent")
    g.add_argument("--zoom", type=float, help="View zoom (= 1 / half-width)")
    g.add_argument("--rotate", help="View rotation: degrees, or '0.5rad', 'pi/6'")
    g.add_argument("--fit", choices=("fill", "cover", "contain"), help="How the view meets the frame aspect")


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    ov: dict[str, Any] = {}
    if args.sets:
        ov["params"] = dict(args.sets)
    if args.palette:
        ov["palette"] = args.palette.split(",") if "," in args.palette else args.palette
    view: dict[str, Any] = {}
    if args.center:
        c = args.center
        view["center"] = [float(x) for x in c.split(",")] if "," in c else c
    if args.half_width is not None:
        view["half_width"] = args.half_width
    if args.zoom is not None:
        view["zoom"] = args.zoom
    if args.rotate is not None:
        view["rotation"] = parse_literal(args.rotate)
    if args.fit:
        view["fit"] = args.fit
    if view:
        ov["view"] = view
    return ov


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fractalgen", description="Extensible math / fractal wallpaper generator.")
    p.add_argument("--load", action="append", default=[], metavar="PATH",
                   help="Extra .toml data file, .py plugin, or directory (repeatable). Also $FRACTALGEN_PATH.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="List recipes")
    s.add_argument("--tag", action="append", help="Only recipes with this tag")
    s.add_argument("--exclude-tag", action="append")
    s.add_argument("--generator", action="append")
    s.add_argument("-l", "--long", action="store_true", help="Show generator, palette, and tags")

    s = sub.add_parser("show", help="Print a recipe's resolved settings")
    s.add_argument("name")
    s.add_argument("--size", type=_size, default=(3840, 2160))

    s = sub.add_parser("generators", help="List generators, or show one generator's parameters")
    s.add_argument("name", nargs="?")

    s = sub.add_parser("palettes", help="List palettes; optionally render a swatch sheet")
    s.add_argument("--swatch", type=Path, metavar="PNG", help="Write a swatch image of every palette")

    sub.add_parser("presets", help="List size presets")

    s = sub.add_parser("render", help="Render recipes")
    s.add_argument("names", nargs="*", help="Recipe names (default: preset's tag selection, or all)")
    s.add_argument("--preset", nargs="+", help="One or more size presets (see `presets`)")
    s.add_argument("--size", type=_size, help="Explicit WxH instead of a preset")
    s.add_argument("--tag", action="append", help="Select recipes with this tag (repeatable, OR)")
    s.add_argument("--exclude-tag", action="append", help="Drop recipes with this tag")
    s.add_argument("--all", action="store_true", help="Ignore the preset's default tag selection")
    s.add_argument("--out", type=Path, help="Output directory (default: backgrounds/<preset>)")
    s.add_argument("--jobs", type=int, default=0, help="Parallel workers (0 = preset default; capped by preset)")
    s.add_argument("--format", choices=("png", "jpg"))
    s.add_argument("--skip-existing", action="store_true")
    s.add_argument("--timeout", type=float, default=20 * 60, help="Per-recipe timeout in seconds")
    s.add_argument("--strips", type=int, help="Row strips for per-pixel kernels (default by size)")
    s.add_argument("--no-isolate", action="store_true", help="Render in-process (no timeout, easier debugging)")
    s.add_argument("--no-lock", action="store_true", help="Allow concurrent batches")
    _add_override_args(s)

    s = sub.add_parser("try", help="Render a generator ad hoc (no recipe) and optionally print it as a recipe")
    s.add_argument("generator")
    s.add_argument("-o", "--output", type=Path, default=Path("try.png"))
    s.add_argument("--size", type=_size, default=(1920, 1080))
    s.add_argument("--emit", metavar="NAME", help="Print a TOML recipe named NAME for these settings")
    _add_override_args(s)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from .loader import bootstrap

    loads = bootstrap(args.load)
    return {
        "list": _cmd_list,
        "show": _cmd_show,
        "generators": _cmd_generators,
        "palettes": _cmd_palettes,
        "presets": _cmd_presets,
        "render": _cmd_render,
        "try": _cmd_try,
    }[args.cmd](args, loads)


def _cmd_list(args, _loads) -> int:
    from .recipe import RECIPES, select

    for name in select(None, args.tag, args.exclude_tag, args.generator):
        if args.long:
            r = RECIPES.get(name)
            pal = r.palette if isinstance(r.palette, str) or r.palette is None else "<inline>"
            print(f"{name:28s} {r.generator:20s} {str(pal or '-'):12s} {','.join(r.tags)}")
        else:
            print(name)
    return 0


def _cmd_show(args, _loads) -> int:
    from .recipe import RECIPES

    r = RECIPES.get(args.name)
    w, h = args.size
    spec, params, view = r.resolve(w, h)
    print(f"recipe     {r.name}  ({r.source})")
    print(f"generator  {spec.name}  -- {spec.description}")
    print(f"tags       {', '.join(r.tags) or '-'}")
    if spec.uses_view:
        print(f"view       center={view.center} half_width={view.half_width} rotation={view.rotation} fit={view.fit}")
    print("params")
    for k, v in params.items():
        mark = " " if k in r.params or k == "palette" else "~"
        print(f"  {mark} {k} = {v!r}")
    print("(~ = generator default)")
    return 0


def _cmd_generators(args, _loads) -> int:
    from .generators.base import GENERATORS

    if args.name:
        g = GENERATORS.get(args.name)
        print(f"{g.name}  [{g.category}]  {g.description}")
        if g.uses_view:
            print("  uses a view: center, half_width, rotation, fit")
        for p in g.params.values():
            default = "(required)" if p.required else repr(p.default)
            print(f"  {p.name:18s} = {default}" + (f"   # {p.doc}" if p.doc else ""))
        if g.portrait_params:
            print(f"  portrait defaults: {g.portrait_params}")
        return 0
    by_cat: dict[str, list] = {}
    for g in (GENERATORS.get(n) for n in GENERATORS):
        by_cat.setdefault(g.category, []).append(g)
    for cat in sorted(by_cat):
        print(f"{cat}:")
        for g in by_cat[cat]:
            print(f"  {g.name:24s} {g.description}")
    return 0


def _cmd_palettes(args, _loads) -> int:
    import numpy as np

    from .color import save_image
    from .palette import PALETTES

    for name in PALETTES:
        pal = PALETTES.get(name)
        print(f"{name:12s} {' '.join(pal.hex_stops())}  {pal.description}")
    if args.swatch:
        names = PALETTES.names()
        row_h, w = 48, 768
        t = np.linspace(0.0, 1.0, w)
        rows = [np.broadcast_to(PALETTES.get(n).sample(t), (row_h, w, 3)) for n in names]
        save_image(args.swatch, np.concatenate(rows, axis=0))
        print(f"swatch ({' / '.join(names)}) -> {args.swatch}")
    return 0


def _cmd_presets(args, _loads) -> int:
    from .engine import PRESETS

    for name, p in PRESETS.items():
        if name != p.name:
            continue
        sel = ",".join(p.tags) or "all"
        alias = f"  (alias: {', '.join(p.aliases)})" if p.aliases else ""
        print(f"{name:11s} {p.width:>6d}x{p.height:<6d} {p.format:4s} jobs={p.jobs}/{p.max_jobs}  tags={sel:5s} {p.description}{alias}")
    return 0


def _cmd_render(args, loads) -> int:
    from . import engine
    from .recipe import select

    if not args.preset and not args.size:
        print("Need --preset or --size.", file=sys.stderr)
        return 2
    targets: list[tuple[str, int, int, engine.Preset | None]] = []
    if args.size:
        targets.append((f"{args.size[0]}x{args.size[1]}", args.size[0], args.size[1], None))
    for name in args.preset or []:
        p = engine.PRESETS.get(name)
        targets.append((p.name, p.width, p.height, p))
    overrides = _overrides(args)

    if not args.no_lock:
        engine.acquire_lock()
    failed = 0
    try:
        for label, w, h, preset in targets:
            tags = args.tag or (None if args.all or preset is None else list(preset.tags) or None)
            exclude = args.exclude_tag or (list(preset.exclude_tags) if preset else None)
            names = select(args.names or None, tags, exclude)
            if not names:
                print(f"{label}: nothing selected", file=sys.stderr)
                continue
            if args.out:
                out = args.out if len(targets) == 1 else args.out / label
            else:
                out = engine.OUT_ROOT / (preset.subdir if preset else "")
            fmt = args.format or (preset.format if preset else engine.default_format(w, h))
            jobs = args.jobs or (preset.jobs if preset else engine.default_jobs(w, h))
            if preset:
                jobs = min(jobs, preset.max_jobs)
            failed += engine.render_batch(
                names, w, h, out, jobs=jobs, fmt=fmt, skip_existing=args.skip_existing,
                n_strips=args.strips, overrides=overrides, loads=loads, timeout_s=args.timeout,
                isolate=not args.no_isolate, label=label,
            )
    finally:
        if not args.no_lock:
            engine.release_lock()
    return 1 if failed else 0


def _cmd_try(args, _loads) -> int:
    import time

    from .color import save_image
    from .recipe import Recipe

    ov = _overrides(args)
    recipe = Recipe(name=args.emit or "try", generator=args.generator)
    w, h = args.size
    t0 = time.perf_counter()
    rgb = recipe.render(w, h, overrides=ov)
    save_image(args.output, rgb)
    print(f"{args.generator} {w}x{h} -> {args.output} ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)
    if args.emit:
        print(_to_toml(args.emit, args.generator, ov))
    return 0


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, complex):
        return f"[{v.real!r}, {v.imag!r}]"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _to_toml(name: str, generator: str, ov: dict[str, Any]) -> str:
    lines = [f"[recipe.{name}]", f'generator = "{generator}"']
    if "palette" in ov:
        lines.append(f"palette = {_toml_value(ov['palette'])}")
    for key in ("view", "params"):
        if ov.get(key):
            lines.append(f"{key} = {_toml_value(ov[key])}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
