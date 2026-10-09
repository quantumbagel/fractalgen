"""Example: type-hint metadata (Range, Doc, Literal), auto_tone, and a post op."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np

import fractalgen as fg


@fg.generator("annotated_waves", category="fields", uses_view=True)
def annotated_waves(
    ctx,
    *,
    mode: Literal["sine", "cosine"] = "sine",
    rings: Annotated[int, fg.Range(2, 24, 1), fg.Doc("number of radial rings")] = 8,
    scale: Annotated[float, fg.Range(0.5, 20.0), fg.Doc("spatial frequency")] = 6.0,
    palette: str = "cosmic",
):
    """Radial waves with annotation-driven GUI hints."""
    re, im = ctx.plane()
    r = np.hypot(re, im)
    wave = np.sin if mode == "sine" else np.cos
    amp = wave(scale * r * (rings / 8.0))
    t = fg.auto_tone(amp)
    return fg.sample_palette(palette, t)


@fg.post_op("soft_clip", "Gentle shoulder on highlights", neutral=0.0, range=(0.0, 1.0))
def soft_clip(rgb, amount, ctx):
    a = float(amount)
    if a <= 0:
        return rgb
    return np.clip(rgb * (1.0 + a) / (1.0 + a * rgb), 0.0, 1.0)
