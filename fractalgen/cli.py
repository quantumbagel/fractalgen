"""Command-line interface: ``python -m fractalgen <command> ...``"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import platform
import random
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .params import deep_merge, parse_literal

from .loader import CUSTOM_DIR

_QUICKSTART = """\
fractalgen — math / fractal wallpapers

Quick start:
  python -m fractalgen list                    # recipes by category
  python -m fractalgen render julia_dragon --size 1920x1080
  python -m fractalgen render --preset 8k      # batch at preset size + tags
  python -m fractalgen gui                     # live browser UI
  python -m fractalgen doctor                  # versions + smoke render

Try: python -m fractalgen --help
"""

_EPILOG = """\
examples:
  python -m fractalgen list --search dragon
  python -m fractalgen show julia_dragon
  python -m fractalgen render clifford_gold --size 960x540 --open
  python -m fractalgen new generator ripples --dir ./custom
"""

_RENDER_EPILOG = """\
examples:
  python -m fractalgen render julia_dragon --size 3840x2160
  python -m fractalgen render --preset 8k iphone julia_dragon
  python -m fractalgen render --preset 4k --random 3 --size 1280x720
  python -m fractalgen presets
"""

_SUBCOMMANDS = (
    "list",
    "show",
    "generators",
    "palettes",
    "presets",
    "post",
    "tone",
    "mask",
    "fork",
    "render",
    "try",
    "gui",
    "new",
    "doctor",
    "completion",
)


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().replace("×", "x").split("x")
        w, h = int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must look like 1920x1080, got {text!r}") from None
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError(f"size width and height must be >= 1, got {text!r}")
    return w, h


def _timeout_s(text: str) -> float:
    t = float(text)
    if not t > 0:  # also rejects NaN (NaN comparisons are false)
        raise argparse.ArgumentTypeError(f"timeout must be > 0, got {text!r}")
    return t


def _port(text: str) -> int:
    p = int(text)
    if not 0 <= p <= 65535:
        raise argparse.ArgumentTypeError(f"port must be 0-65535, got {text!r}")
    return p


_SWEEP_VIEW_KEYS = frozenset({"zoom", "center", "half_width", "rotation", "fit", "ref_aspect"})
_SWEEP_TABLES = frozenset({"view", "post", "tone", "params", "mask"})
_SWEEP_TAG_SAFE = re.compile(r"[^A-Za-z0-9._=@+-]")


def _kv(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    k, v = text.split("=", 1)
    return k.strip(), parse_literal(v)


def _kv_raw(text: str) -> tuple[str, str]:
    """key=value with the value kept as typed (mask text, fonts and hex colors stay strings)."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    k, v = text.split("=", 1)
    return k.strip(), v


def _suggest(kind: str, name: str, known: Iterable[str], n: int = 5) -> str:
    hits = difflib.get_close_matches(name, list(known), n=n, cutoff=0.5)
    if not hits:
        return f"Unknown {kind} {name!r}."
    return f"Unknown {kind} {name!r}. Did you mean: {', '.join(hits)}?"


def _require_name(registry: Any, name: str) -> None:
    if name not in registry:
        raise KeyError(_suggest(registry.kind, name, registry.names()))


def _require_names(registry: Any, names: Sequence[str]) -> None:
    missing = [n for n in names if n not in registry]
    if not missing:
        return
    if len(missing) == 1:
        raise KeyError(_suggest(registry.kind, missing[0], registry.names()))
    raise KeyError(f"Unknown {registry.kind}(s): {missing}")


def _add_override_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("overrides (applied on top of each recipe)")
    g.add_argument(
        "--set",
        dest="sets",
        type=_kv,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help='Generator parameter, e.g. --set max_iter=2000 --set c="-0.8+0.156j"',
    )
    g.add_argument(
        "--palette",
        help='Palette name (append _r to reverse) or comma hex stops, e.g. ember or 000000,ff5500,ffffff',
    )
    g.add_argument("--center", help='View center, e.g. -0.745+0.11j or -0.745,0.11')
    zh = g.add_mutually_exclusive_group()
    zh.add_argument("--half-width", type=float, metavar="FLOAT", help="View horizontal half-extent, e.g. 1.45")
    zh.add_argument("--zoom", type=float, help="View zoom (= 1 / half-width), e.g. 2.0")
    g.add_argument("--rotate", help='View rotation in degrees or radians, e.g. 30 or pi/6')
    g.add_argument("--fit", choices=("fill", "cover", "contain"), help="How the view meets the frame aspect")
    g.add_argument(
        "--post",
        dest="posts",
        type=_kv,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Finishing op, e.g. --post saturation=1.3 --post vignette=0.35 (see `post`)",
    )
    g.add_argument(
        "--tone",
        dest="tones",
        type=_kv,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Auto-exposure tweak, e.g. --tone white=0.8 --tone curve=1.4 (see `tone`)",
    )
    g.add_argument(
        "--mask",
        dest="masks",
        type=_kv_raw,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help='Cut the render out with text / a shape / an image, e.g. --mask text=HELLO '
             '--mask background=self --mask glow=0.5 (see `mask`)',
    )


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    ov: dict[str, Any] = {}
    if args.sets:
        ov["params"] = dict(args.sets)
    if args.palette:
        ov["palette"] = args.palette.split(",") if "," in args.palette else args.palette
    view: dict[str, Any] = {}
    if args.half_width is not None and args.zoom is not None:
        raise ValueError("cannot use --zoom and --half-width together")
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
    if args.posts:
        ov["post"] = dict(args.posts)
    if args.tones:
        ov["tone"] = dict(args.tones)
    if args.masks:
        from .mask import check_mask

        ov["mask"] = check_mask(dict(args.masks))
    return ov


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fractalgen",
        description="Extensible math / fractal wallpaper generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    from . import __version__

    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--load",
        action="append",
        default=[],
        metavar="PATH",
        help="Extra .toml, .py plugin, or directory (repeatable); also $FRACTALGEN_PATH",
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    s = sub.add_parser("list", help="List recipes (grouped by generator category)")
    s.add_argument("--tag", action="append", help='Only recipes with this tag, e.g. vip')
    s.add_argument("--exclude-tag", action="append", help='Drop recipes with this tag, e.g. draft')
    s.add_argument("--generator", action="append", help='Only this generator, e.g. escape_time')
    s.add_argument("-l", "--long", action="store_true", help="Also show generator, palette, and tags")
    s.add_argument("--search", metavar="TEXT", help='Filter: name, description, tags, or generator')
    s.add_argument("--json", action="store_true", help="Machine-readable JSON array of recipe records")
    s.add_argument("--names", action="store_true", help="One recipe name per line (for shell completion)")

    s = sub.add_parser("show", help="Print a recipe's resolved settings")
    s.add_argument("name", help="Recipe name, e.g. julia_dragon")
    s.add_argument("--size", type=_size, default=(3840, 2160), help="Frame used to resolve view, e.g. 3840x2160")

    s = sub.add_parser("generators", help="List generators, or show one generator's parameters")
    s.add_argument("name", nargs="?", help="Generator name, e.g. escape_time")

    s = sub.add_parser("palettes", help="List palettes; optionally render a swatch sheet")
    s.add_argument("--swatch", type=Path, metavar="PNG", help="Write a swatch image, e.g. palettes.png")

    sub.add_parser("presets", help="List output size presets (use with render --preset)")
    sub.add_parser("post", help="List post-processing ops for recipe post = {...} / --post")
    sub.add_parser("tone", help="List tone keys for recipe tone = {...} / --tone")
    sub.add_parser("mask", help="List mask keys for recipe mask = {...} / --mask (fractal text, shapes, images)")

    s = sub.add_parser("fork", help="Copy a recipe into an editable TOML with every parameter spelled out")
    s.add_argument("name", help="Recipe to copy (e.g. julia_dragon), or a PNG/JPEG fractalgen rendered")
    s.add_argument("new_name", nargs="?", help="Name of the copy (default: NAME_custom)")
    s.add_argument("-o", "--output", type=Path, help=f"TOML file (default: {CUSTOM_DIR.name}/<new_name>.toml)")
    s.add_argument("--stdout", action="store_true", help="Print TOML to stdout instead of writing")
    s.add_argument("--force", action="store_true", help="Overwrite an existing file")

    s = sub.add_parser(
        "render",
        help="Render recipes to image files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_RENDER_EPILOG,
    )
    s.add_argument("names", nargs="*", help="Recipe names (default: preset tag selection, or all)")
    s.add_argument(
        "--preset",
        dest="preset_groups",
        action="append",
        nargs="+",
        metavar="NAME",
        help="Size preset(s); repeat flag or list several, e.g. --preset 8k or --preset 8k iphone",
    )
    s.add_argument("--size", type=_size, help="Explicit WxH instead of a preset, e.g. 1920x1080")
    s.add_argument("--tag", action="append", help="Select recipes with this tag (OR)")
    s.add_argument("--exclude-tag", action="append", help="Drop recipes with this tag")
    s.add_argument("--all", action="store_true", help="Ignore the preset's default tag selection")
    s.add_argument("--random", nargs="?", const=1, type=int, metavar="N", help="Pick N random non-draft recipes")
    s.add_argument("--out", type=Path, help="Output directory (default: backgrounds/<preset>)")
    s.add_argument("--open", action="store_true", help="Open each output with the OS image viewer")
    s.add_argument("--jobs", type=int, default=0, help="Parallel workers (0 = preset default)")
    s.add_argument("--format", choices=("png", "jpg"), help="Output format (default: by size)")
    s.add_argument("--skip-existing", action="store_true", help="Skip when the output file already exists")
    s.add_argument("--timeout", type=_timeout_s, default=20 * 60, help="Per-recipe timeout in seconds, e.g. 1200")
    s.add_argument("--strips", type=int, help="Row strips for per-pixel kernels")
    s.add_argument("--no-isolate", action="store_true", help="Render in-process (no timeout; easier debugging)")
    s.add_argument("--no-lock", action="store_true", help="Allow concurrent batches")
    s.add_argument(
        "--sweep",
        type=_kv,
        metavar="KEY=V1,V2,...",
        help="One image per value, e.g. --sweep max_iter=200,400,800",
    )
    _add_override_args(s)

    s = sub.add_parser("gui", help="Open the live-tweaking GUI in your browser")
    s.add_argument("--port", type=_port, default=8765, help="HTTP port, e.g. 8765")
    s.add_argument("--host", default="127.0.0.1", help="Bind address, e.g. 127.0.0.1")
    s.add_argument("--no-browser", action="store_true", help="Do not open a browser tab automatically")

    s = sub.add_parser("try", help="Render a generator ad hoc (no recipe)")
    s.add_argument("generator", help="Generator name, e.g. escape_time")
    s.add_argument("-o", "--output", type=Path, default=Path("try.png"), help="Output image path")
    s.add_argument("--size", type=_size, default=(1920, 1080), help="Frame size, e.g. 960x540")
    s.add_argument("--emit", metavar="NAME", help="Print a TOML recipe named NAME for these settings")
    _add_override_args(s)

    s = sub.add_parser("new", help="Scaffold a custom generator, post op, or palette")
    s.add_argument("kind", choices=("generator", "post", "palette"), help="What to create")
    s.add_argument("name", help="Plugin name (snake_case)")
    s.add_argument("--category", default="custom", help="Generator category for `list generators`, e.g. fields")
    s.add_argument("--view", action="store_true", help="Generator uses ctx.plane() / view")
    s.add_argument("--stops", help="Palette hex stops for `new palette`, e.g. 000000,ff5500,ffffff")
    s.add_argument("--dir", type=Path, default=CUSTOM_DIR, help="Output directory (default: custom/)")
    s.add_argument("--force", action="store_true", help="Overwrite existing files")

    sub.add_parser("doctor", help="Environment report for bug reports")

    s = sub.add_parser("completion", help="Print a shell completion script")
    s.add_argument("shell", choices=("bash", "zsh", "fish"), help="Shell family")

    return p


def _print_quickstart() -> None:
    print(_QUICKSTART.rstrip())


def _user_errors() -> tuple[type[BaseException], ...]:
    return (KeyError, ValueError, TypeError, FileNotFoundError, FileExistsError)


def _extract_load(argv: list[str]) -> tuple[list[str], list[str]]:
    """Pull global ``--load PATH`` out of argv (works after the subcommand too)."""
    out: list[str] = []
    loads: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--load":
            if i + 1 >= len(argv):
                raise ValueError("--load requires a path")
            loads.append(argv[i + 1])
            i += 2
            continue
        if argv[i].startswith("--load="):
            path = argv[i].split("=", 1)[1]
            if not path:
                raise ValueError("--load requires a path")
            loads.append(path)
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out, loads


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    try:
        if not argv:
            _print_quickstart()
            return 0
        if argv == ["-h"] or argv == ["--help"]:
            _parser().print_help()
            return 0
        argv, extra_loads = _extract_load(argv)
        try:
            args = _parser().parse_args(argv)
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 2
        if extra_loads:
            args.load = list(args.load) + extra_loads
        if args.cmd is None:
            _print_quickstart()
            return 0
        from .loader import bootstrap

        loads = bootstrap(args.load)
        return {
            "list": _cmd_list,
            "show": _cmd_show,
            "generators": _cmd_generators,
            "palettes": _cmd_palettes,
            "presets": _cmd_presets,
            "post": _cmd_post,
            "tone": _cmd_tone,
            "mask": _cmd_mask,
            "fork": _cmd_fork,
            "gui": _cmd_gui,
            "render": _cmd_render,
            "try": _cmd_try,
            "new": _cmd_new,
            "doctor": _cmd_doctor,
            "completion": _cmd_completion,
        }[args.cmd](args, loads)
    except _user_errors() as exc:
        if os.environ.get("FRACTALGEN_DEBUG") == "1":
            raise
        # str(KeyError) repr-quotes its message; show the message itself.
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {msg}", file=sys.stderr)
        return 2


def _filtered_recipes(args: argparse.Namespace) -> list[str]:
    from .recipe import RECIPES, select

    names = select(None, args.tag, args.exclude_tag, args.generator)
    if args.search:
        q = args.search.lower()
        out = []
        for n in names:
            r = RECIPES.get(n)
            hay = " ".join(
                (
                    n,
                    r.description,
                    r.generator,
                    " ".join(r.tags),
                )
            ).lower()
            if q in hay:
                out.append(n)
        return out
    return names


def _cmd_list(args, _loads) -> int:
    from .generators.base import GENERATORS
    from .recipe import RECIPES

    names = _filtered_recipes(args)
    if args.names:
        for name in names:
            print(name)
        return 0
    if args.json:
        rows = []
        for name in names:
            r = RECIPES.get(name)
            rows.append(
                {
                    "name": name,
                    "generator": r.generator,
                    "description": r.description,
                    "tags": list(r.tags),
                    "palette": r.palette if isinstance(r.palette, str) or r.palette is None else "<inline>",
                }
            )
        print(json.dumps(rows, indent=2))
        return 0

    width = shutil.get_terminal_size((100, 20)).columns
    by_cat: dict[str, list[str]] = {}
    for name in names:
        cat = GENERATORS.get(RECIPES.get(name).generator).category
        by_cat.setdefault(cat, []).append(name)
    for cat in sorted(by_cat):
        print(f"{cat}:")
        for name in by_cat[cat]:
            r = RECIPES.get(name)
            desc = r.description or "—"
            if args.long:
                pal = r.palette if isinstance(r.palette, str) or r.palette is None else "<inline>"
                extra = f"  {r.generator}  {str(pal or '-')}  {','.join(r.tags)}"
                line = f"  {name:28s} {desc}"
                if len(line) + len(extra) > width:
                    print(line[: max(40, width - len(extra) - 1)] + "…" + extra)
                else:
                    print(f"{line}{extra}")
            else:
                line = f"  {name:28s} {desc}"
                print(line if len(line) <= width else line[: width - 1] + "…")
    return 0


def _cmd_show(args, _loads) -> int:
    from .recipe import RECIPES

    _require_name(RECIPES, args.name)
    r = RECIPES.get(args.name)
    w, h = args.size
    spec, params, view = r.resolve(w, h)
    print(f"recipe     {r.name}  ({r.source})")
    print(f"description {r.description or '-'}")
    print(f"generator  {spec.name}  -- {spec.description}")
    print(f"tags       {', '.join(r.tags) or '-'}")
    if r.tone:
        print(f"tone       {r.tone}")
    if r.post:
        print(f"post       {r.post}")
    if r.mask:
        print(f"mask       {r.mask}")
    if r.portrait:
        print(f"portrait   {r.portrait}")
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
        _require_name(GENERATORS, args.name)
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

    rows = [(name, p) for name, p in PRESETS.items() if name == p.name]
    name_w = max((len(name) for name, _ in rows), default=0)
    for name, p in rows:
        sel = ",".join(p.tags) or "all"
        alias = f"  (alias: {', '.join(p.aliases)})" if p.aliases else ""
        print(
            f"{name:{name_w}s} {p.width:>6d}x{p.height:<6d} {p.format:4s} "
            f"jobs={p.jobs}/{p.max_jobs}  tags={sel:5s} {p.description}{alias}"
        )
    return 0


def _split_preset_groups(groups: list[list[str]] | None, preset_keys: set[str]) -> tuple[list[str], list[str]]:
    """Return (preset names, extra recipe names) from --preset nargs groups."""
    presets: list[str] = []
    extra: list[str] = []
    for group in groups or []:
        if not group:
            continue
        i = 0
        while i < len(group) and group[i] in preset_keys:
            presets.append(group[i])
            i += 1
        if i == 0:
            raise KeyError(_suggest("preset", group[0], preset_keys))
        extra.extend(group[i:])
    return presets, extra


def _open_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        if platform.system() == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        elif platform.system() == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)


def _cmd_render(args, loads) -> int:
    from . import engine
    from .engine import PRESETS
    from .recipe import RECIPES, select

    preset_keys = set(PRESETS.names())
    preset_names, from_presets = _split_preset_groups(args.preset_groups, preset_keys)
    recipe_names = list(args.names) + from_presets
    if preset_names:
        _require_names(PRESETS, preset_names)

    if not preset_names and not args.size:
        print(
            "render needs --size WxH or --preset NAME.\n"
            "  python -m fractalgen render julia_dragon --size 1920x1080\n"
            "  python -m fractalgen presets",
            file=sys.stderr,
        )
        return 2

    targets: list[tuple[str, int, int, engine.Preset | None]] = []
    if args.size and preset_names:
        w, h = args.size
        for name in preset_names:
            p = PRESETS.get(name)
            targets.append((p.name, w, h, p))
    elif args.size:
        targets.append(("", args.size[0], args.size[1], None))
    else:
        for name in preset_names:
            p = PRESETS.get(name)
            targets.append((p.name, p.width, p.height, p))
    overrides = _overrides(args)
    sweeps = _sweep_overrides(args.sweep, overrides) if args.sweep else [("", overrides)]

    if recipe_names:
        _require_names(RECIPES, recipe_names)

    if recipe_names and sweeps:
        w0, h0 = targets[0][1], targets[0][2]
        for suffix, ov in sweeps:
            RECIPES.get(recipe_names[0]).resolve(w0, h0, ov)

    if not args.no_lock:
        engine.acquire_lock()
    failed = 0
    written: list[Path] = []
    try:
        for label, w, h, preset in targets:
            tags = args.tag or (None if args.all or preset is None else list(preset.tags) or None)
            exclude = args.exclude_tag or (list(preset.exclude_tags) if preset else None)
            names = select(recipe_names or None, tags, exclude)
            if args.random is not None:
                pool = [n for n in names if "draft" not in RECIPES.get(n).tags]
                if not pool:
                    print(f"{label}: no non-draft recipes in selection", file=sys.stderr)
                    continue
                k = min(max(1, args.random), len(pool))
                names = random.sample(pool, k)
            if not names:
                print(f"{label}: nothing selected", file=sys.stderr)
                continue
            if args.out:
                out = args.out if len(targets) == 1 else args.out / (label or f"{w}x{h}")
            else:
                out = engine.OUT_ROOT / (preset.subdir if preset else "")
            fmt = args.format or (preset.format if preset else engine.default_format(w, h))
            jobs = args.jobs or (preset.jobs if preset else engine.default_jobs(w, h))
            if preset:
                jobs = min(jobs, preset.max_jobs)
            ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"
            for suffix, ov in sweeps:
                for n in names:
                    written.append(out / f"{n}{'@' + suffix if suffix else ''}.{ext}")
                failed += engine.render_batch(
                    names,
                    w,
                    h,
                    out,
                    jobs=jobs,
                    fmt=fmt,
                    skip_existing=args.skip_existing,
                    n_strips=args.strips,
                    overrides=ov,
                    loads=loads,
                    timeout_s=args.timeout,
                    isolate=not args.no_isolate,
                    label=" ".join(filter(None, [label, suffix])),
                    suffix=suffix,
                )
    finally:
        if not args.no_lock:
            engine.release_lock()
    for path in written:
        if path.exists():
            print(path)
    if args.open:
        _open_paths([p for p in written if p.exists()])
    return 1 if failed else 0


def _sweep_overrides(sweep: tuple[str, Any], base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    key, values = sweep
    if isinstance(values, str):
        values = [parse_literal(v) for v in values.split(",")]
    elif not isinstance(values, (list, tuple)):
        values = [values]
    out = []
    for v in values:
        if key == "palette":
            patch: dict[str, Any] = {"palette": v}
        elif key in _SWEEP_VIEW_KEYS:
            patch = {"view": {key: v}}
        elif "." in key:
            table, sub = key.split(".", 1)
            if table not in _SWEEP_TABLES:
                raise ValueError(
                    f"unknown sweep key {key!r}; use a view./post./tone./params./mask. prefix "
                    f"(or bare view keys: {', '.join(sorted(_SWEEP_VIEW_KEYS))})"
                )
            patch = {table: {sub: v}}
        else:
            patch = {"params": {key: v}}
        tag = _SWEEP_TAG_SAFE.sub("_", f"{key.split('.')[-1]}={v}")
        out.append((tag, deep_merge(base, patch)))
    return out


def _cmd_post(args, _loads) -> int:
    from .post import POST_OPS

    for name, (_, doc) in POST_OPS.items():
        print(f"  {name:12s} {doc}")
    return 0


def _cmd_tone(args, _loads) -> int:
    from .tone import TONE_KEYS

    for name, key in TONE_KEYS.items():
        print(f"  {name:12s} {key.doc} (default {key.neutral})")
    return 0


def _cmd_mask(args, _loads) -> int:
    from .mask import MASK_KEYS

    print("Off until text, shape or image is set. Lengths are pixels at 4K.")
    for name, key in MASK_KEYS.items():
        extra = f" [{', '.join(c for c in key.choices if c)}]" if key.choices else ""
        default = f" (default {key.neutral!r})" if key.neutral not in ("", None) else ""
        print(f"  {name:16s} {key.doc}{extra}{default}")
    return 0


def _cmd_fork(args, _loads) -> int:
    from .recipe import RECIPES
    from .tomlout import recipe_to_toml

    from .recipe import Recipe

    image = Path(args.name)
    if image.suffix.lower() in (".png", ".jpg", ".jpeg") and image.is_file():
        from .tomlout import recipe_from_image

        name, table = recipe_from_image(image)  # the look the image was rendered with
        src = Recipe.from_table(name, table, str(image))
        args.name = name
    else:
        _require_name(RECIPES, args.name)
        src = RECIPES.get(args.name)
    new = args.new_name or f"{args.name.replace('@', '_')}_custom"
    if new in RECIPES and not args.force:
        print(
            f"Recipe {new!r} already exists ({RECIPES.get(new).source}); pick another name or --force",
            file=sys.stderr,
        )
        return 2
    text = recipe_to_toml(src, new)
    if args.stdout:
        print(text)
        return 0
    path = args.output or CUSTOM_DIR / f"{new}.toml"
    if path.exists() and not args.force:
        print(f"{path} exists (use --force)", file=sys.stderr)
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"{args.name} -> {new}  written to {path}")
    print(f"edit it, then: python -m fractalgen render {new} --size 1920x1080")
    return 0


def _cmd_gui(args, loads) -> int:
    from .gui.server import serve

    serve(args.host, args.port, open_browser=not args.no_browser, loads=loads)
    return 0


def _cmd_try(args, _loads) -> int:
    import time

    from . import progress
    from .color import save_image
    from .engine import RENDER_SHARE, live_status
    from .generators.base import GENERATORS
    from .recipe import Recipe

    _require_name(GENERATORS, args.generator)
    ov = _overrides(args)
    if ov.get("palette"):
        from .palette import PALETTES

        pal = ov["palette"]
        if isinstance(pal, str) and pal not in PALETTES and not pal.endswith("_r"):
            _require_name(PALETTES, pal)
    recipe = Recipe(name=args.emit or "try", generator=args.generator)
    w, h = args.size
    t0 = time.perf_counter()
    with live_status(args.generator) as tracker, progress.tracking(tracker):
        with progress.span(0.0, RENDER_SHARE):
            rgb = recipe.render(w, h, overrides=ov)
        with progress.span(RENDER_SHARE, 1.0, "saving"):
            from .tomlout import image_meta

            save_image(args.output, rgb, meta=image_meta(recipe.with_overrides(ov), w, h))
    print(f"{args.generator} {w}x{h} -> {args.output} ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)
    if args.emit:
        from .tomlout import overrides_to_toml

        print(overrides_to_toml(args.emit, args.generator, ov))
    return 0


def _sanitize_plugin_name(name: str) -> str:
    if not name.replace("_", "").isalnum() or not name.isidentifier():
        raise ValueError(f"Name must be a snake_case identifier, got {name!r}")
    return name


def _cmd_new(args, _loads) -> int:
    name = _sanitize_plugin_name(args.name)
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.kind == "generator":
        py_path = out_dir / f"{name}.py"
        toml_path = out_dir / f"{name}.toml"
        for path in (py_path, toml_path):
            if path.exists() and not args.force:
                raise FileExistsError(f"{path} exists (use --force)")
        plane = "    re, im = ctx.plane()\n" if args.view else "    re, im = np.meshgrid(np.linspace(-1, 1, ctx.width), np.linspace(-1, 1, ctx.height))\n"
        py_path.write_text(
            textwrap.dedent(
                f'''\
"""Custom generator `{name}` — loaded from {out_dir.name}/."""

import numpy as np

import fractalgen as fg


@fg.generator(
    "{name}",
    category="{args.category}",
    uses_view={args.view!s},
    param_docs={{
        "palette": "gradient mapped over equalized field",
        "seed": "noise seed",
    }},
    ranges={{"seed": (0, 1000, 1)}},
)
def {name}(ctx, *, palette="abyss", seed: int = 1):
    """Smooth radial waves with histogram tone."""
{plane}    rng = np.random.default_rng(seed)
    field = np.sin(3.0 * np.hypot(re, im) + rng.random() * 6.28)
    t = fg.histogram_index(field, np.ones_like(field, dtype=bool))
    return fg.sample_palette(palette, t)
'''
            )
        )
        toml_path.write_text(
            textwrap.dedent(
                f'''\
[recipe.{name}]
description = "Custom {name} look"
generator = "{name}"
palette = "abyss"
tags = ["custom"]
params = {{ seed = 1 }}
'''
            )
        )
        print(f"Wrote {py_path} and {toml_path}")
        print(f"Try: python -m fractalgen --load {out_dir} render {name} --size 640x360")
        return 0
    if args.kind == "post":
        py_path = out_dir / f"{name}.py"
        if py_path.exists() and not args.force:
            raise FileExistsError(f"{py_path} exists (use --force)")
        py_path.write_text(
            textwrap.dedent(
                f'''\
"""Custom post op `{name}`."""

import numpy as np

import fractalgen as fg


@fg.post_op("{name}", "Blend toward mid-gray by amount", neutral=0.0, range=(0.0, 1.0))
def {name}(rgb, amount, ctx):
    amount = float(amount)
    return rgb * (1.0 - amount) + 0.5 * amount
'''
            )
        )
        print(f"Wrote {py_path}")
        print(f"Try: python -m fractalgen --load {py_path} render julia_dragon --size 64x36 --post {name}=0.2")
        return 0
    if not args.stops:
        raise ValueError("--stops is required for `new palette`, e.g. --stops 000000,ff5500,ffffff")
    stops = [s.strip().lstrip("#") for s in args.stops.split(",")]
    toml_path = out_dir / f"{name}_palette.toml"
    if toml_path.exists() and not args.force:
        raise FileExistsError(f"{toml_path} exists (use --force)")
    stops_line = ", ".join(f'"{s}"' for s in stops)
    toml_path.write_text(
        f'[palette.{name}]\n'
        f'description = "Custom palette {name}"\n'
        f"stops = [{stops_line}]\n"
    )
    print(f"Wrote {toml_path}")
    print(f"Try: python -m fractalgen --load {toml_path} render julia_dragon --size 64x36 --palette {name}")
    return 0


def _cmd_doctor(args, loads) -> int:
    import numpy as np

    from .engine import PRESETS
    from .generators.base import GENERATORS
    from .loader import CUSTOM_DIR, DATA_DIR
    from .palette import PALETTES
    from .recipe import RECIPES

    from . import __version__

    print(f"fractalgen {__version__}")
    print(f"python     {platform.python_version()} ({sys.executable})")
    print(f"numpy      {np.__version__}")
    try:
        import numba

        layer = getattr(numba.config, "THREADING_LAYER", "?")
        print(f"numba      {numba.__version__}  threading_layer={layer}")
    except ImportError:
        print("numba      (not installed)")
    try:
        from PIL import Image

        print(f"pillow     {Image.__version__}")
    except ImportError:
        print("pillow     (not installed)")
    try:
        import scipy

        print(f"scipy      {scipy.__version__}")
    except ImportError:
        print("scipy      (not installed)")
    print(f"cpu_count  {os.cpu_count()}")
    print(f"custom_dir {CUSTOM_DIR.resolve()}")
    env_path = os.environ.get("FRACTALGEN_PATH", "")
    print(f"FRACTALGEN_PATH {env_path or '(unset)'}")
    print(f"extra --load {loads or '(none)'}")
    print(f"data_dir   {DATA_DIR}")
    print(f"counts     recipes={len(RECIPES)} generators={len(GENERATORS)} palettes={len(PALETTES)} presets={len([n for n, p in PRESETS.items() if n == p.name])}")

    from .recipe import Recipe

    t0 = time.perf_counter()
    recipe = Recipe(name="_doctor", generator="escape_time")
    recipe.render(
        64,
        36,
        overrides={"params": {"formula": "mandelbrot", "max_iter": 32}, "palette": "ember"},
    )
    print(f"smoke      escape_time 64x36 in {time.perf_counter() - t0:.2f}s (includes numba compile)")
    return 0


def _cmd_completion(args, _loads) -> int:
    shell = args.shell
    cmds = " ".join(_SUBCOMMANDS)
    if shell == "bash":
        script = textwrap.dedent(
            f"""\
# bash completion for fractalgen — eval "$(python -m fractalgen completion bash)"
_fractalgen() {{
  local cur prev words cword
  _init_completion || return
  local recipes
  recipes=$(python -m fractalgen list --names 2>/dev/null)
  if [[ $cword -eq 1 ]]; then
    COMPREPLY=($(compgen -W "{cmds}" -- "$cur"))
    return
  fi
  case "${{words[1]}}" in
    render|show|fork)
      COMPREPLY=($(compgen -W "$recipes" -- "$cur"))
      ;;
    completion)
      COMPREPLY=($(compgen -W "bash zsh fish" -- "$cur"))
      ;;
  esac
}}
complete -F _fractalgen fractalgen
complete -F _fractalgen python
"""
        )
    elif shell == "zsh":
        script = textwrap.dedent(
            f"""\
# zsh completion for fractalgen — eval "$(python -m fractalgen completion zsh)"
#compdef fractalgen
local -a commands recipes
commands=({' '.join(_SUBCOMMANDS)})
recipes=("${{@(${{(f)"$(python -m fractalgen list --names 2>/dev/null)"}})}}")
if (( CURRENT == 2 )); then
  _describe 'command' commands
elif [[ $words[2] == render || $words[2] == show || $words[2] == fork ]]; then
  _describe 'recipe' recipes
elif [[ $words[2] == completion ]]; then
  _describe 'shell' bash zsh fish
fi
"""
        )
    else:
        script = textwrap.dedent(
            f"""\
# fish completion for fractalgen — python -m fractalgen completion fish | source
function __fractalgen_recipes
  python -m fractalgen list --names 2>/dev/null
end
complete -c fractalgen -f -n "not __fish_seen_subcommand_from {_SUBCOMMANDS}" -a "{cmds}"
complete -c fractalgen -f -n "__fish_seen_subcommand_from render show fork" -a "(__fractalgen_recipes)"
complete -c fractalgen -f -n "__fish_seen_subcommand_from completion" -a "bash zsh fish"
"""
        )
    print(script.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
