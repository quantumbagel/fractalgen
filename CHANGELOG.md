# Changelog

## 0.1.0 (2026-11-09)

Initial public release, completely rewriting the original standalone wallpaper scripts (`generate.py`, `ultra.py`, and `bg/`, preserved in the git history at commit `1bdc817`) into an extensible, recipe-driven engine.

- Ships 129 built-in looks across escape-time fractals, Newton basins, Lyapunov parameter space, complex domain coloring, strange attractors, iterated function systems, reaction-diffusion simulations, flow fields, slime mold networks, abelian sandpiles, hyperbolic tilings, harmonographs, fractal flames, Buddhabrot densities, KAM islands, and caustics.
- Adds an interactive local web GUI (`python -m fractalgen gui`) with real-time viewport adjustments, parameter sliders, variation generators, side-by-side comparisons, undo and redo history, keyboard navigation, and full-resolution wallpaper export with supersampling. The frontend runs as ES modules using bundled Preact and htm without build steps.
- Introduces data-driven TOML recipes combining generator choices, parameters, palettes, view coordinates, tone mapping, composite layers, and post-processing stacks.
- Supports multi-layer compositing for text, geometric shapes, and external images, offering independent opacity, blend modes, strokes, glows, drop shadows, and bevels. Legacy single-mask definitions remain compatible.
- Implements an ordered post-processing pipeline using repeatable `[[post]]` tables for adjustments including bloom, blur, sharpening, film grain, chromatic aberration, gradients, and posterization.
- Standardizes framing and exposure against a 4K reference canvas so preview windows and high-resolution exports produce matching composition and brightness.
- Includes render presets for standard desktop resolutions, phone displays, ultrawides, and multi-monitor configurations up to 16K.
- Accelerates core routines using parallel Numba kernels, Brent cycle detection for escape-time iterations, and multi-threaded simulation solvers.
- Exposes plugin hooks for custom generators, palettes, formulas, 2D maps, 3D flows, vector fields, domain functions, and custom layers in Python or TOML, including discovery via `fractalgen.plugins` entry points.
- Provides a unified CLI for rendering, cataloging, scaffolding extensions, and diagnosing dependencies.
- Embeds complete recipe definitions directly into generated PNG and JPEG files, allowing any exported image to be restored or forked into a source recipe.
