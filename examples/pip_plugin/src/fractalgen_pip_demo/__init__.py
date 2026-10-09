"""Minimal installable plugin: ``pip install -e examples/pip_plugin`` then ``bootstrap()``."""

from __future__ import annotations

import numpy as np

import fractalgen as fg


def register() -> None:
    """Entry-point target: register a tiny generator."""

    @fg.generator("pip_demo", category="test", replace=True)
    def pip_demo(ctx, *, palette: str = "ember"):
        """Flat field from pip entry point."""
        h, w = ctx.height, ctx.width
        t = np.linspace(0, 1, w)[None, :].repeat(h, axis=0)
        return fg.sample_palette(palette, t)
