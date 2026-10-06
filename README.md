# fractalgen

An extensible math / fractal wallpaper generator. 95 built-in looks across escape-time
fractals, Newton basins, Lyapunov space, domain coloring, strange attractors, IFS,
reaction-diffusion, flow fields, cellular automata, black holes, raymarched 3D fractals
and more. Everything that shapes a picture is data you can tweak: generator parameters,
palettes, framing (center / zoom / **rotation**), camera angles, and finishing.

The original hard-coded scripts (`generate.py`, `ultra.py`, `bg/`) live on the
`old-code` branch. Every recipe here reproduces their 4K output pixel-for-pixel;
other sizes are now consistent with 4K (see "Same picture at every size").

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: pip install -e .
python -m fractalgen gui                         # live-tweaking GUI in your browser
python -m fractalgen list -l                     # recipes, generator, palette, tags
python -m fractalgen render julia_dragon --size 3840x2160
python -m fractalgen render --preset 8k iphone   # batch: presets pick recipes by tag
```

## Concepts

| Thing | What it is | Where it lives |
|---|---|---|
| **generator** | an algorithm: `fn(ctx, *, **params) -> RGB`. Its keyword args *are* its schema | `fractalgen/generators/*.py` |
| **recipe** | a named look: generator + params + palette + view + post + tags | `fractalgen/data/recipes/*.toml` |
| **palette** | gradient stops (optionally positioned) | `fractalgen/data/palettes.toml` |
| **view** | complex-plane window: `center`, `half_width`, `rotation` (deg), `fit` | recipe `view = {...}` |
| **post** | finishing that works on *any* render: exposure, contrast, gamma, saturation, hue shift, tint, vignette, palette remap, flips | recipe `post = {...}` |
| **formula registries** | escape-time formulas, 2D maps, 3D flows, vector fields, domain functions | Python or TOML |
| **preset** | output size + default recipe selection + memory caps | `fractalgen/data/presets.toml` |

`python -m fractalgen generators NAME` lists every parameter of a generator with its
default and documentation; `python -m fractalgen show RECIPE` shows a recipe's resolved
values (`~` marks generator defaults).

## Customizing a render

From quickest to most permanent:

```sh
# 1. One-off overrides on any render
python -m fractalgen render julia_dragon --size 1920x1080 \
    --set max_iter=900 --set c=-0.79+0.16j --palette aurora_r \
    --center -0.1+0.2j --zoom 2 --rotate 30 \
    --post saturation=1.3 --post vignette=0.35

# 2. Explore one knob: renders julia_dragon@max_iter=200.png, ...
python -m fractalgen render julia_dragon --size 960x540 --sweep max_iter=200,400,800
python -m fractalgen render clifford_gold --size 960x540 --sweep post.hue_shift=0,60,120

# 3. Live: sliders for every parameter, drag to pan, wheel to zoom, save as recipe
python -m fractalgen gui

# 4. Fork into an editable file with EVERY parameter spelled out and documented
python -m fractalgen fork mandelbulb_grotto my_grotto     # -> custom/my_grotto.toml
python -m fractalgen render my_grotto --size 3840x2160
```

`custom/` is loaded automatically. In a recipe file, small tweaks don't need a full copy:

```toml
[recipe.dragon_night]
extends = "julia_dragon"            # inherit everything but tags/description, override a little
palette = "abyss"
post = { exposure = -0.3, vignette = 0.4 }

[recipe.dragon_night.variants.tilted]   # renders as dragon_night@tilted
view = { rotation = 25, half_width = 0.9 }
post = { hue_shift = 40 }

[recipe.dragon_night.portrait]          # merged in when height > width (phones)
view = { center = [-0.2, 0.25], half_width = 0.6, fit = "fill" }
```

Palettes anywhere: a name (`"ember"`), reversed (`"ember_r"`), inline stops
(`["000000", "ff6a00", "fff3d6"]`), or `{ stops = [...], positions = [0, 0.3, 1] }`.
Angles anywhere: degrees (`30`), or strings like `"0.72rad"`, `"pi/6"`.
Complex numbers: `[re, im]` or `"-0.8+0.156j"`.

### The GUI

`python -m fractalgen gui` opens a local page with a control for every parameter of
the current recipe. Each control shows what it does, its unit, default and typical
range (toggle with "show help"). Drag the image to pan, scroll to zoom, Shift+scroll
or Alt+drag to rotate, `R` resets the view; Undo/Redo, Reset, and the URL keep your
edits (bookmark or reload it). **Save recipe** writes `custom/NAME.toml` with only
your edits layered over the original recipe, so its phone (portrait) framing still
works; **Export** renders at full size into `backgrounds/gui/`.

### Same picture at every size

Every look is defined at a 4K reference frame (3840×2160's pixel count, at your
frame's aspect), so a 720p preview, a 4K wallpaper and a 16K export show the same
picture, only sharper:

- simulations (reaction-diffusion, slime mold, cellular automata, sandpiles) run on
  a grid sized from the reference and are upsampled; repeated renders with the same
  simulation parameters reuse the result in-process, so recoloring is instant;
- particle and line renders put their hit counts on the reference's scale before tone
  mapping, so brightness doesn't drift with resolution;
- lengths documented as "pixels at 4K" scale with the output.

Output at exactly 3840×2160 is unchanged by this. Simulations cost the same at every
output size (a preview takes as long to simulate as a 4K render).

## Extending

### Without Python (TOML in `custom/`, `--load PATH`, or `$FRACTALGEN_PATH`)

```toml
[palette.sunset]
stops = ["000000", "ff5500", "ffd27f", "ffffff"]
positions = [0, 0.3, 0.8, 1]

[formula.sine_julia]          # escape-time formula, compiled to numba
z = "c * sin(z)"              # names: z, c, zp (previous z), k, sin/cos/exp/log/sqrt/abs/conj/re/im
julia = true                  # pixel is z0 (else pixel is c)
degree = 1.0                  # polynomial degree for smooth coloring (<= 1: plain counts)
bailout = 50.0

[map.my_map]                  # 2D iterated map for map_attractor
x = "sin(a*y) - cos(b*x)"
y = "sin(c*x) - cos(d*y)"
params = { a = 1.4, b = -2.3, c = 2.4, d = -2.1 }

[flow.my_flow]                # 3D ODE for flow_attractor
dx = "s*(y - x)"
dy = "x*(r - z) - y"
dz = "x*y - b*z"
params = { s = 10.0, r = 28.0, b = 2.667 }

[vectorfield.my_field]        # 2D field for flow_field (variables px, py)
vx = "sin(a*py) + cos(px)"
vy = "cos(px*py)"
params = { a = 1.7 }

[domain.my_fn]                # complex function for domain_color (variable z)
w = "(z**3 - a) / (z**2 + 1)"
params = { a = 1.0 }

[recipe.sine_glow]
generator = "escape_time"
palette = "sunset"
view = { center = [0, 0], half_width = 3.5, rotation = 30 }
params = { formula = "sine_julia", c = [1.0, 0.1], max_iter = 120 }
```

See `examples/extensions.toml` for a runnable version.

### With Python (a `.py` file in `custom/`, or `--load plugin.py`)

```python
import numpy as np
import fractalgen as fg

@fg.generator("ripples", category="fields", uses_view=True,
              param_docs={"waves": "number of wave sources"}, ranges={"waves": (1, 20, 1)})
def ripples(ctx, *, waves: int = 5, freq: float = 12.0, palette="abyss", seed: int = 1):
    """Interfering circular waves."""
    re, im = ctx.plane()                    # honors center / zoom / rotation / fit
    rng = np.random.default_rng(seed)
    amp = sum(np.sin(freq * np.hypot(re - x, im - y)) for x, y in rng.uniform(-1, 1, (waves, 2)))
    return fg.sample_palette(palette, fg.histogram_index(amp, np.ones_like(amp, bool)))

@fg.post_op("posterize", "Quantize each channel to N levels", range=(2, 32))
def posterize(rgb, levels, ctx):
    return np.round(rgb * levels) / levels
```

Other hooks: `fg.register_palette`, `fg.register_formula` (numba step
`f(z, c, zp, k) -> complex`), `fg.register_map`, `fg.register_flow`,
`fractalgen.generators.fields.register_vector_field`, `register_domain`,
`fractalgen.loader.section("name")` for brand-new TOML tables, and
`fg.Recipe(...)` / `fg.register_recipe` to build looks in code.
New generators automatically appear in the CLI, `fork`, and the GUI (with sliders
from `ranges` and dropdowns from `choices`).

## Batch rendering

```sh
python -m fractalgen presets
python -m fractalgen render --preset 4k                 # everything at 3840x2160
python -m fractalgen render --preset 8k iphone-8k       # tag "keep"
python -m fractalgen render --preset 16k                # tag "vip", JPEG
python -m fractalgen render --preset 8k --tag vip --skip-existing --jobs 1
```

Each recipe renders in its own process with a timeout (`--timeout`), so an OOM or a
runaway render skips that wallpaper instead of the batch. Per-pixel kernels render in
row strips at 8K/16K to bound memory. Output: `backgrounds/<preset>/<recipe>.png`.

| old command | new command |
|---|---|
| `python generate.py` | `python -m fractalgen render --preset 4k` |
| `python generate.py --only a b` | `python -m fractalgen render a b --size 3840x2160` |
| `python ultra.py --preset 8k` (tier keep) | `python -m fractalgen render --preset 8k` |
| `python ultra.py --tier vip` | `... --tag vip` |
| `python ultra.py --tier all` | `... --all --exclude-tag draft` |
| `python render.py --preset desktop-8k` | `python -m fractalgen render --preset desktop-8k` |

Tags: `keep` (8K-worthy), `vip` (16K-worthy), `draft` (weaker looks skipped by `--tier all`).

## Development

```sh
python -m unittest discover tests        # unit tests + smoke-render every recipe
python tools/parity.py BASELINE_DIR      # compare against old-code renders
python tools/gui_check.py --all          # drive the GUI in headless Chromium (needs playwright)
```

`tools/parity.py` compares small renders; since the resolution-independence work,
particle and simulation recipes match the old code at 3840×2160 only (smaller sizes
now look like the 4K render instead of differing from it).

Numba kernels intentionally skip numba's on-disk cache: with `fastmath`, a
cache-loaded kernel can round differently from a fresh compile and flip chaotic
boundary pixels, so identical recipes rendered differently between runs. Compiling
costs a second or two per process.
