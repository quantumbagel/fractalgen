# Changelog

## 0.1.0 — 2026-11-09

First public release. A recipe-driven rewrite of the original hard-coded
wallpaper scripts (`generate.py`, `ultra.py`, `bg/` — still on the `old-code`
branch).

- **129 built-in looks** across escape-time fractals, Newton basins, Lyapunov
  space, domain coloring, strange attractors, IFS, reaction-diffusion, flow
  fields, slime mold, sandpiles, hyperbolic tilings, harmonographs, fractal
  flames, Buddhabrots, KAM islands, caustics, and more.
- **GUI** (`python -m fractalgen gui`): a task-first local page with a look gallery,
  Variations, and tabs for Look / Frame / Color / Layers / Effects. It previews in your
  screen's exact shape, has Compare, undo / redo, keyboard shortcuts and a phone layout,
  and a Save wallpaper dialog with per-screen sizes, supersampling, and either a direct
  browser download or the `backgrounds/gui/` folder. No build step: small ES modules
  with a vendored Preact + htm.
- **Data-driven looks.** A recipe is generator + params + palette + view +
  tone + effects + layers, all TOML. Fork a built-in, extend it, or write one from
  scratch.
- **Layers**: any number of text, shape and image layers, each painted on top or cut
  out of the render, with its own position, opacity, blend mode, outline, glow, shadow
  and bevel. The older single `mask = {...}` still loads and renders identically.
- **Ordered effects**: `[[post]]` tables that repeat, reorder, switch off and blend:
  bloom, blur, sharpen, grain, chromatic aberration, gradient, posterize and more.
- **The same picture at every size.** Looks are defined at a 4K reference
  frame, so a 720p preview and a 16K export agree.
- **Batch presets** for 4K / 8K / 16K desktops, iPhone, iPad, ultrawide, and
  multi-monitor spans.
- **Fast renders.** Numba kernels throughout; escape-time orbits that exactly repeat
  stop early (Brent cycle detection), magnetic pendulum and polynomial Newton run as
  parallel kernels.
- **Extension hooks** in TOML or Python: generators, palettes, formulas, 2D
  maps, 3D flows, vector fields, domain functions, effects, shapes and layer kinds.
  Pip packages can register via the `fractalgen.plugins` entry point.
- CLI: `render`, `list`, `fork`, `new`, `presets`, `doctor`, with `--layer`,
  `--effect`, `--background`, `--no-layers` and `--no-post` overrides.
- **Embedded recipes.** Every PNG / JPEG fractalgen writes carries the full
  look, so you can fork a wallpaper back from the file.
