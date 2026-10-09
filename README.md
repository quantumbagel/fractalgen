<p align="center">
  <img src="docs/assets/header.jpg" alt="fractalgen" width="100%">
</p>

<p align="center">
  <b>Math, as wallpaper.</b><br>
  An extensible fractal / dynamical-systems generator with a live GUI,<br>
  129 built-in looks, and recipes you can fork.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/looks-129-ff6a00?style=flat-square" alt="129 looks">
  <img src="https://img.shields.io/badge/license-MIT-2ea44f?style=flat-square" alt="MIT">
</p>

The header above is the engine rendering its own name: `julia_dragon` cut into
type with a glow, bevel, and a dimmed copy of itself behind the letters.

```sh
python -m venv .venv && .venv/bin/pip install -e .
fractalgen gui                                   # live-tweaking GUI in your browser
fractalgen list -l                               # recipes, generator, palette, tags
fractalgen render julia_dragon --size 3840x2160
fractalgen render --preset 8k iphone             # batch: presets pick recipes by tag
```

(`python -m fractalgen` is the same command.)

Escape-time fractals, Newton basins, Lyapunov space, domain coloring, strange
attractors, IFS, reaction-diffusion, flow fields, slime mold, sandpiles,
hyperbolic tilings, harmonographs, fractal flames, Buddhabrots, KAM islands,
caustics, and more. Everything that shapes a picture is data you can tweak:
generator parameters, palettes, framing (center / zoom / **rotation**), camera
angles, finishing, and masks (fractal-filled text and shapes).

The original hard-coded scripts (`generate.py`, `ultra.py`, `bg/`) live on the
`old-code` branch. Every recipe reproduces their 4K output pixel-for-pixel except
`physarum_dark_matter` (now simulated on all cores); other sizes are consistent
with 4K (see [Same picture at every size](#same-picture-at-every-size)).

## Gallery

<p align="center">
  <img src="docs/assets/julia_dragon.jpg" width="49%" alt="julia_dragon">
  <img src="docs/assets/julia_zoom.jpg" width="49%" alt="Julia zoom">
</p>
<p align="center"><sub><code>julia_dragon</code> · Julia zoom (GUI)</sub></p>
<p align="center">
  <img src="docs/assets/newton_sextic.jpg" width="49%" alt="newton_sextic">
  <img src="docs/assets/julia_snowflake.jpg" width="49%" alt="julia_snowflake">
</p>
<p align="center"><sub><code>newton_sextic</code> · <code>julia_snowflake</code></sub></p>
<p align="center">
  <img src="docs/assets/clifford_gold.jpg" width="49%" alt="clifford_gold">
  <img src="docs/assets/flow_ember.jpg" width="49%" alt="flow attractor">
</p>
<p align="center"><sub><code>clifford_gold</code> · flow attractor (GUI)</sub></p>
<p align="center">
  <img src="docs/assets/lyapunov_baba.jpg" width="49%" alt="lyapunov_baba">
  <img src="docs/assets/mandelbrot_seahorse.jpg" width="49%" alt="mandelbrot_seahorse">
</p>
<p align="center"><sub><code>lyapunov_baba</code> · <code>mandelbrot_seahorse</code></sub></p>
<p align="center">
  <img src="docs/assets/burning_ship_cathedral.jpg" width="49%" alt="burning_ship_cathedral">
  <img src="docs/assets/attractor_burst.jpg" width="49%" alt="attractor burst">
</p>
<p align="center"><sub><code>burning_ship_cathedral</code> · attractor burst (GUI)</sub></p>
<p align="center">
  <img src="docs/assets/magnetic_pendulum_tri.jpg" width="49%" alt="magnetic_pendulum_tri">
  <img src="docs/assets/domain_exp.jpg" width="49%" alt="domain_exp">
</p>
<p align="center"><sub><code>magnetic_pendulum_tri</code> · <code>domain_exp</code></sub></p>

Rebuild the header (and gallery thumbs, if you have local 8K renders) with
`python docs/render_assets.py`.

## Concepts

| Thing | What it is | Where it lives |
|---|---|---|
| **generator** | an algorithm: `fn(ctx, *, **params) -> RGB`. Its keyword args *are* its schema | `fractalgen/generators/*.py` |
| **recipe** | a named look: generator + params + palette + view + tone + post + tags | `fractalgen/data/recipes/*.toml` |
| **palette** | gradient stops (optionally positioned) | `fractalgen/data/palettes.toml` |
| **view** | complex-plane window (Lyapunov: the (a, b) plane): `center`, `half_width`, `rotation` (deg), `fit` | recipe `view = {...}` |
| **tone** | adjusts a generator's automatic exposure before the palette: `black`, `white`, `curve`, `per_frame` (see below) | recipe `tone = {...}` |
| **post** | finishing that works on *any* render: exposure, contrast, gamma, saturation, hue shift, tint, vignette, palette remap, flips | recipe `post = {...}` |
| **mask** | cuts any render out with text, a shape or an image, over a background (see below) | recipe `mask = {...}` |
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
    --tone white=0.85 --post saturation=1.3 --post vignette=0.35

# 2. Explore one knob: renders julia_dragon@max_iter=200.png, ...
python -m fractalgen render julia_dragon --size 960x540 --sweep max_iter=200,400,800
python -m fractalgen render clifford_gold --size 960x540 --sweep post.hue_shift=0,60,120

# 3. Live: sliders for every parameter, drag to pan, wheel to zoom, save as recipe
python -m fractalgen gui

# 4. Fork into an editable file with EVERY parameter spelled out and documented
python -m fractalgen fork physarum_dark_matter my_slime   # -> custom/my_slime.toml
python -m fractalgen render my_slime --size 3840x2160

# 5. Every PNG / JPEG fractalgen writes embeds the full look; fork it back from the image
python -m fractalgen fork backgrounds/gui/julia_dragon_3840x2160.png dragon_again
```

The embedded recipe (`fractalgen.recipe` text chunk, or a JSON comment in JPEGs) includes
every override: palette, params, view, post, tone and mask. A look saved from a pan or zoom
of another recipe records `tone_from = "that_recipe"`, so it keeps that recipe's exposure
rather than re-measuring on the new view.

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

### Tone

Most generators stretch their raw values (iteration counts, orbit hits, densities) to
the screen automatically: a histogram or percentile range measured from the pixels.
That measurement is taken once, on the recipe's own view, and frozen as the look's
default tone. Panning, zooming or resizing (in the GUI or with `--center` / `--zoom`)
then keeps the same exposure instead of re-measuring every frame. `tone` adjusts that
stretch before the palette is applied (`post` works afterwards, on the final colors):

```toml
tone = { black = 0.05, white = 0.9, curve = 1.3 }   # omitted keys keep the default
```

`black` / `white` clip the stretch (0..1), `curve` bends the midtones (>1 brighter), and
`per_frame = true` brings back re-measuring on every frame, which is handy for deep zooms
past where the default's range runs out. `python -m fractalgen tone` lists them;
`--sweep tone.white=1,0.8,0.6` explores one. Generators with no automatic stretch
(domain coloring, for instance) ignore `tone`.

### Masks: fractal text, shapes and cut-outs

Any recipe can be cut out with text, a shape or an image; the fractal shows through
and `background` fills the rest:

```toml
[recipe.dragon_lettering]
extends = "julia_dragon"
mask = { text = "FRACTAL", glow = 0.7, bevel = 0.5, background = "self", background_dim = 0.12 }
portrait.mask = { text = "FRAC\nTAL" }          # phones get two lines
```

```sh
python -m fractalgen render julia_dragon --size 3840x2160 \
    --mask text=HELLO --mask font="Avenir Next Heavy" --mask outline=6 --mask background=self
python -m fractalgen render julia_dragon --size 2048x2048 \
    --mask image=logo.png --mask background=transparent      # PNG with alpha
python -m fractalgen mask                                     # every key, documented
```

- **what**: `text` (`\n` breaks lines; `font` takes a family name like `"Arial Black"` or
  `"Avenir Next Heavy"`, or a font file), `shape` (circle, ring, heart, star, hexagon,
  rounded), or `image` (its alpha, or its brightness). Several combine.
- **where**: `size` (fraction of the frame it fills), `x` / `y` (0..1, from the top left),
  `rotate`, `invert` (fractal outside instead of inside), `feather`.
- **finish**: `outline` + `outline_color`, `glow` (the fractal's own colors bleeding past
  the edge) + `glow_radius`, `shadow`, `bevel`.
- **background**: a hex color, `transparent`, `self` (the fractal itself, dimmed by
  `background_dim`, softened by `background_blur`), `recipe:NAME` (another look), or
  `palette:NAME` (a vertical gradient).

The mask runs before `post`, so post finishes the whole frame. Lengths are pixels at 4K, so
previews match exports. In the GUI the Mask section switches on as soon as text, a shape
or an image is set, and dragging the preview moves the fractal inside the letters.
Built-in examples: `seahorse_porthole`, `snowflake_star`, `dragon_lettering`.

Palettes anywhere: a name (`"ember"`), reversed (`"ember_r"`), inline stops
(`["000000", "ff6a00", "fff3d6"]`), or `{ stops = [...], positions = [0, 0.3, 1] }`.
Angles anywhere: degrees (`30`), or strings like `"0.72rad"`, `"pi/6"`.
Complex numbers: `[re, im]` or `"-0.8+0.156j"`.

### The GUI

`python -m fractalgen gui` opens a local page with a control for every parameter of
the current recipe. Each control shows what it does, its unit, default and typical
range (toggle with "show help"). Drag the image to pan, scroll to zoom, Shift+scroll
or Alt+drag to rotate, `R` resets the view; Undo/Redo, Reset, and the URL keep your
edits (bookmark or reload it). **Save recipe** writes `custom/NAME.toml` as a
complete, self-contained recipe (every parameter spelled out at the 4K reference,
including your edits and the original's portrait overrides). A pan or zoom made on a
desktop frame carries over to the phone framing as the same shift and zoom factor (and
vice versa), and the saved look keeps the original's exposure. **Export** renders at full
size into `backgrounds/gui/`, with a progress bar and the recipe embedded in the file.

### Same picture at every size

Every look is defined at a 4K reference frame (3840×2160's pixel count, at your
frame's aspect), so a 720p preview, a 4K wallpaper and a 16K export show the same
picture, only sharper:

- simulations (reaction-diffusion, slime mold, sandpiles) run on
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
Long generators can call `fractalgen.progress.update(fraction)` inside their loops
so export and batch progress bars move (see `fractalgen/progress.py`).
New generators automatically appear in the CLI, `fork`, and the GUI (with sliders
from `ranges` and dropdowns from `choices`).

Scaffold without hand-copying:

```sh
python -m fractalgen new generator my_waves --view    # -> custom/my_waves.py + .toml
python -m fractalgen new post my_grade
python -m fractalgen new palette sunset --stops 000000,ff5500,ffd700,ffffff
```

A third-party package can register hooks with a `fractalgen.plugins` entry point
(see `examples/pip_plugin/`).

## Batch rendering

```sh
python -m fractalgen presets
python -m fractalgen render --preset 4k                 # everything at 3840x2160
python -m fractalgen render --preset 8k iphone-8k       # tag "keep"
python -m fractalgen render --preset 16k                # tag "vip", JPEG
python -m fractalgen render --preset 8k --tag vip --skip-existing --jobs 1
python -m fractalgen render --preset super-ultrawide julia_dragon   # 5120x1440, one picture
```

Besides 4K / 8K / 16K desktops and iPhones there are `5k`, `macbook`, `ipad`, `ultrawide`
(3440×1440), `ultrawide-5k`, `super-ultrawide` (32:9), and `dual-4k` / `triple-4k` for one
picture spanning several monitors. `--size` overrides a preset's size but keeps its tags and
format. The GUI's aspect menu lists every preset.

Each recipe renders in its own process with a timeout (`--timeout`), so an OOM or a
runaway render skips that wallpaper instead of the batch. Per-pixel kernels render in
row strips at 8K/16K to bound memory. Output: `backgrounds/<preset>/<recipe>.png`.
On a terminal, a live line shows each running recipe's progress and elapsed time.

| old command | new command |
|---|---|
| `python generate.py` | `python -m fractalgen render --preset 4k` |
| `python generate.py --only a b` | `python -m fractalgen render a b --size 3840x2160` |
| `python ultra.py --preset 8k` (tier keep) | `python -m fractalgen render --preset 8k` |
| `python ultra.py --tier vip` | `... --tag vip` |
| `python ultra.py --tier all` | `... --all --exclude-tag draft` |
| `python render.py --preset desktop-8k` | `python -m fractalgen render --preset desktop-8k` |

Tags: `keep` (8K-worthy), `vip` (16K-worthy), `draft` (weaker looks skipped by `--tier all`).

## Python API

```python
import fractalgen as fg
fg.bootstrap()
rgb = fg.render("julia_dragon", 1920, 1080)
fg.save_image("dragon.png", rgb)
```

`overrides` on `fg.render` may hold `params`, `view`, `palette`, `tone`, `post`,
and `mask`, the same keys as the CLI flags.

## Development

```sh
python -m unittest discover tests        # unit tests + smoke-render every recipe
python tools/parity.py BASELINE_DIR      # compare against old-code renders
python tools/gui_check.py --all          # drive the GUI in headless Chromium (needs playwright)
python -m fractalgen doctor              # versions + a tiny smoke render
```

`tools/parity.py` compares small renders; since the resolution-independence work,
particle and simulation recipes match the old code at 3840×2160 only (smaller sizes
now look like the 4K render instead of differing from it).

Numba kernels intentionally skip numba's on-disk cache: with `fastmath`, a
cache-loaded kernel can round differently from a fresh compile and flip chaotic
boundary pixels, so identical recipes rendered differently between runs. Compiling
costs a second or two per process.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a look or a generator.

## License

[MIT](LICENSE). See [CHANGELOG.md](CHANGELOG.md) for the 0.1.0 notes.
