"""Human-readable labels for parameter and option names (GUI, docs).

``label_for("max_iter") == "Detail (iterations)"``; names without an entry are
humanized ("glow_radius" -> "Glow radius").
"""

from __future__ import annotations

LABELS: dict[str, str] = {
    # generators, common
    "palette": "Colors",
    "max_iter": "Detail (iterations)",
    "c": "Julia constant",
    "formula": "Formula",
    "power": "Power",
    "bailout": "Escape radius",
    "coloring": "Coloring",
    "interior": "Inside color",
    "seed": "Random seed",
    "samples": "Samples",
    "n_points": "Points",
    "steps": "Steps",
    "dt": "Time step",
    "coeffs": "Coefficients",
    "symmetry": "Symmetry",
    "gamma": "Gamma",
    "exposure": "Exposure",
    "saturation": "Saturation",
    "blur": "Softness",
    "density": "Density",
    "trap": "Orbit trap",
    "tol": "Tolerance",
    "Du": "Diffusion U",
    "Dv": "Diffusion V",
    "freq": "Frequency",
    "n": "Count",
    "m": "Mode m",
    "map": "Map",
    "flow": "Flow",
    "field": "Field",
    "expr": "Expression",
    "xforms": "Transforms",
    "n_xforms": "Transforms",
    "colors": "Colors",
    "speed": "Speed",
    "zoom": "Zoom",
    "rotation": "Rotation",
    "angle_deg": "Angle",
    "color_by": "Color by",
    "hue_mix": "Hue mix",
    "glow_color": "Glow color",
    "glow_strength": "Glow strength",
    "background_color": "Background color",
    "percentiles": "Clip percentiles",
    "tight": "Tight framing",
    "fill": "Fill",
    # view
    "center": "Center",
    "half_width": "Zoom (half width)",
    "fit": "Fit to screen",
    # tone
    "black": "Black point",
    "white": "White point",
    "curve": "Midtones",
    "per_frame": "Re-expose every frame",
    # layers / background / effects
    "text": "Text",
    "font": "Font",
    "shape": "Shape",
    "image": "Image",
    "image_channel": "Mask from",
    "size": "Size",
    "x": "Horizontal position",
    "y": "Vertical position",
    "rotate": "Rotation",
    "opacity": "Opacity",
    "blend": "Blend mode",
    "invert": "Invert",
    "feather": "Soft edge",
    "outline": "Outline",
    "outline_color": "Outline color",
    "glow": "Glow",
    "glow_radius": "Glow size",
    "shadow": "Shadow",
    "bevel": "Bevel",
    "dim": "Brightness",
    "hue_shift": "Hue shift",
    "contrast": "Contrast",
    "tint": "Tint",
    "vignette": "Vignette",
    "flip_x": "Mirror horizontally",
    "flip_y": "Mirror vertically",
    "chromatic_aberration": "Chromatic aberration",
    "mix": "Strength",
    "value": "Amount",
}

_WORDS = {"n": "number of", "iter": "iterations", "hi": "high", "lo": "low", "pct": "percent", "freq": "frequency",
          "amp": "amplitude", "deg": "(degrees)", "pow": "power", "rad": "radius", "sat": "saturation",
          "val": "value", "k": "k", "dt": "time step", "num": "number of", "init": "initial", "max": "max",
          "min": "min", "dist": "distance", "coc": "blur circle"}


def label_for(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    words = [w for w in name.split("_") if w]
    out = [_WORDS.get(w, w) for w in words]
    text = " ".join(out).strip()
    return text[:1].upper() + text[1:] if text else name
