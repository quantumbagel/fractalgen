# Changelog

## Unreleased

- **New GUI.** A rebuilt, task-first interface: a look gallery, Variations, tabs for
  Look / Frame / Color / Layers / Effects, previews in your screen's exact shape, Compare,
  a Save wallpaper dialog with per-screen sizes, supersampling, and a choice between a
  direct browser download and the `backgrounds/gui/` folder, image upload, keyboard shortcuts and a phone layout. No build step: small ES
  modules with a vendored Preact + htm.
- **Layers** replace the single mask: any number of text, shape and image layers, each
  painted on top or cut out of the render, with its own position, opacity, blend mode,
  outline, glow, shadow and bevel. `mask = {...}` still loads and renders identically.
- **Ordered effects.** `post` is a list (`[[post]]` tables): effects repeat, reorder,
  switch off and blend. New: bloom, blur, sharpen, grain, chromatic aberration, gradient,
  posterize. The table form still works.
- **Plugins** can add shapes (`fg.shape`), layer kinds (`fg.layer_type`) and
  multi-parameter effects (`fg.effect` / `fg.Key`); the GUI picks them up.
- CLI: `--layer`, `--effect`, `--background`, `--no-layers`, `--no-post`.
- **Faster renders.** Escape-time kernels stop orbits that exactly repeat (Brent cycle
  detection: same results, far less work inside the set): Buddhabrots up to 8× faster.
  Magnetic pendulum and polynomial Newton run as parallel kernels (5–8×), and attractor
  window detection uses a parallel histogram (Rössler 2.3×).

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
