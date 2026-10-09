# Changelog

## 0.1.0 — 2026-10-08

First public release. A recipe-driven rewrite of the original hard-coded
wallpaper scripts (`generate.py`, `ultra.py`, `bg/` — still on the `old-code`
branch).

- **129 built-in looks** across escape-time fractals, Newton basins, Lyapunov
  space, domain coloring, strange attractors, IFS, reaction-diffusion, flow
  fields, slime mold, sandpiles, hyperbolic tilings, harmonographs, fractal
  flames, Buddhabrots, KAM islands, caustics, and more.
- **Live GUI** (`python -m fractalgen gui`): sliders for every parameter, drag
  to pan, wheel to zoom, save as a recipe, export at full size.
- **Data-driven looks.** A recipe is generator + params + palette + view +
  tone + post + mask, all TOML. Fork a built-in, extend it, or write one from
  scratch.
- **Masks.** Cut any render out with text, a shape, or an image (fractal-filled
  lettering, portholes, stars).
- **The same picture at every size.** Looks are defined at a 4K reference
  frame, so a 720p preview and a 16K export agree.
- **Batch presets** for 4K / 8K / 16K desktops, iPhone, iPad, ultrawide, and
  multi-monitor spans.
- **Extension hooks** in TOML or Python: generators, palettes, formulas, 2D
  maps, 3D flows, vector fields, domain functions, post ops. Pip packages can
  register via the `fractalgen.plugins` entry point.
- **Embedded recipes.** Every PNG / JPEG fractalgen writes carries the full
  look, so you can fork a wallpaper back from the file.
