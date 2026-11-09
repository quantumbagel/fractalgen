#!/usr/bin/env python3
"""Regenerate README images: header (and optional gallery thumbs)."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"

HEADER_SIZE = (3840, 960)
HEADER_MASK = {
    "text": "fractalgen",
    "font": "Avenir Next Heavy",
    "glow": 0.9,
    "glow_radius": 64,
    "bevel": 0.55,
    "shadow": 0.7,
    "background": "self",
    "background_dim": 0.14,
    "background_blur": 48,
    "size": 0.9,
    "feather": 1.2,
}
HEADER_POST = {"vignette": 0.3}

GALLERY = (
    ("backgrounds/8k/julia_dragon.png", "julia_dragon.jpg"),
    ("backgrounds/gui/I-have-nailed-it_7680x4320.png", "julia_zoom.jpg"),
    ("backgrounds/8k/newton_sextic_dusk.png", "newton_sextic.jpg"),
    ("backgrounds/8k/julia_snowflake.png", "julia_snowflake.jpg"),
    ("backgrounds/8k/clifford_gold.png", "clifford_gold.jpg"),
    ("backgrounds/gui/THISISFIRE_3840x2160.png", "flow_ember.jpg"),
    ("backgrounds/8k/lyapunov_baba.png", "lyapunov_baba.jpg"),
    ("backgrounds/8k/mandelbrot_seahorse.png", "mandelbrot_seahorse.jpg"),
    ("backgrounds/8k/burning_ship_cathedral.png", "burning_ship_cathedral.jpg"),
    ("backgrounds/gui/paint_3840x2160.png", "attractor_burst.jpg"),
    ("backgrounds/8k/magnetic_pendulum_tri.png", "magnetic_pendulum_tri.jpg"),
    ("backgrounds/8k/domain_exp.png", "domain_exp.jpg"),
)

# One look per family for the README's "What it draws" grid, rendered fresh from the
# current recipes (1080p, scaled down for clean edges).
LOOKS = (
    "mandelbrot_spiral", "buddhabrot_nebula", "newton_star", "lyapunov_aurum",
    "domain_sin_inv", "flame_bloom", "lorenz_butterfly", "hopalong",
    "barnsley_fern", "gray_scott_coral", "physarum_biolum", "sandpile_ice",
    "poincare_ember", "harmonograph_gold", "caustics_pool", "standard_map_golden",
)


def render_looks(size: tuple[int, int] = (480, 270)) -> None:
    import fractalgen as fg
    from fractalgen.color import to_image
    from PIL import Image

    fg.bootstrap()
    out = ASSETS / "looks"
    out.mkdir(parents=True, exist_ok=True)
    for name in LOOKS:
        rgb = fg.render(name, size[0] * 4, size[1] * 4)
        im = to_image(rgb).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        dest = out / f"{name}.jpg"
        im.save(dest, "JPEG", quality=84, optimize=True, progressive=True)
        print(f"wrote {dest.relative_to(ROOT)}")


def render_header() -> Path:
    import fractalgen as fg

    fg.bootstrap()
    ASSETS.mkdir(parents=True, exist_ok=True)
    w, h = HEADER_SIZE
    rgb = fg.render("julia_dragon", w, h, mask=HEADER_MASK, post=HEADER_POST)
    png = ASSETS / "header.png"
    jpg = ASSETS / "header.jpg"
    fg.save_image(png, rgb)
    from PIL import Image

    Image.open(png).convert("RGB").save(jpg, "JPEG", quality=92, optimize=True, progressive=True)
    png.unlink()
    return jpg


def render_gallery(width: int = 960) -> None:
    from PIL import Image

    ASSETS.mkdir(parents=True, exist_ok=True)
    for rel, dest_name in GALLERY:
        src = ROOT / rel
        if not src.exists():
            alts = {
                "newton_sextic.jpg": (
                    "backgrounds/8k/newton_octic.png",
                    "backgrounds/8k/newton_sextic.png",
                ),
            }
            for alt in alts.get(dest_name, ()):
                if (ROOT / alt).exists():
                    src = ROOT / alt
                    break
            else:
                print(f"skip missing {rel}")
                continue
        im = Image.open(src).convert("RGB")
        nh = max(1, int(round(im.size[1] * (width / im.size[0]))))
        im = im.resize((width, nh), Image.Resampling.LANCZOS)
        dest = ASSETS / dest_name
        im.save(dest, "JPEG", quality=88, optimize=True, progressive=True)
        print(f"wrote {dest.relative_to(ROOT)} ({im.size[0]}x{im.size[1]})")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Regenerate the README header (julia_dragon cut into "
        "'fractalgen' type). Pass --gallery to also rebuild thumbs from backgrounds/."
    )
    p.add_argument("--gallery", action="store_true", help="Also rebuild gallery thumbs")
    p.add_argument("--looks", action="store_true", help="Only rebuild the per-family look thumbs")
    args = p.parse_args()
    if args.looks:
        render_looks()
        return 0
    path = render_header()
    print(f"wrote {path.relative_to(ROOT)}")
    if args.gallery:
        render_gallery()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
