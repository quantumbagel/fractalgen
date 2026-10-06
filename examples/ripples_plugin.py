"""Example plugin: a new generator and a new post op, in plain Python.

    python -m fractalgen --load examples/ripples_plugin.py try ripples -o ripples.png
    python -m fractalgen --load examples/ripples_plugin.py --load examples/extensions.toml \
        render ripple_pool --size 1920x1080

Drop a file like this into custom/ to have it loaded automatically.
"""

import numpy as np

import fractalgen as fg


@fg.generator(
    "ripples",
    category="fields",
    uses_view=True,
    param_docs={
        "waves": "number of point sources",
        "freq": "ripple frequency (rings per unit)",
        "decay": "amplitude falloff with distance",
        "palette": "gradient mapped over equalized amplitude",
        "seed": "source placement seed",
    },
    ranges={"waves": (1, 24, 1), "freq": (1.0, 60.0), "decay": (0.0, 2.0), "seed": (0, 1000, 1)},
)
def ripples(ctx, *, waves: int = 5, freq: float = 12.0, decay: float = 0.3, palette="abyss", seed: int = 1):
    """Interfering circular waves from random point sources."""
    re, im = ctx.plane()  # honors center / zoom / rotation / fit
    rng = np.random.default_rng(seed)
    amp = np.zeros_like(re)
    for x, y in rng.uniform(-1.0, 1.0, (waves, 2)):
        r = np.hypot(re - x, im - y)
        amp += np.sin(freq * r) / (1.0 + decay * r)
    t = fg.histogram_index(amp, np.ones_like(amp, dtype=bool))
    return fg.sample_palette(palette, t)


@fg.post_op("posterize", "Quantize each channel to N levels", neutral=0, range=(0, 32))
def posterize(rgb, levels, ctx):
    levels = int(levels)
    return rgb if levels < 2 else np.round(rgb * (levels - 1)) / (levels - 1)
