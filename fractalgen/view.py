"""Framing: 2D complex-plane views, 3D projections, and angle parsing.

All angles in recipes are degrees unless written with a unit string:
``30``, ``"30deg"``, ``"0.72rad"``, ``"pi/4"`` (radians when ``pi`` appears).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .params import to_complex

REF_ASPECT = 16.0 / 9.0
FITS = ("fill", "cover", "contain")


def angle(value: Any) -> float:
    """Parse an angle spec and return radians."""
    if isinstance(value, (int, float)):
        return math.radians(float(value))
    s = str(value).strip().lower().replace(" ", "")
    if s.endswith("rad"):
        return float(_eval_pi(s[:-3]))
    if s.endswith("deg"):
        return math.radians(float(_eval_pi(s[:-3])))
    if s.endswith("°"):
        return math.radians(float(s[:-1]))
    if "pi" in s:
        return float(_eval_pi(s))
    return math.radians(float(s))


def _eval_pi(s: str) -> float:
    if not re.fullmatch(r"[0-9.eE+\-*/()pi]+", s):
        raise ValueError(f"Bad angle expression {s!r}")
    return float(eval(s, {"__builtins__": {}}, {"pi": math.pi}))  # noqa: S307 - charset-restricted


@dataclass(frozen=True)
class View:
    """A window onto the complex plane.

    ``half_width`` is the horizontal half-extent of the authored 16:9 window
    for ``cover`` / ``contain``, or of *this* frame for ``fill``.
    ``rotation`` (degrees, counter-clockwise) spins the window about ``center``.

    fit:
      fill    -- half_width is for this frame's aspect
      cover   -- same vertical zoom as the reference window, crop/extend sideways
      contain -- show the whole reference window, pad along the long axis
    """

    center: complex = 0j
    half_width: float = 2.0
    rotation: float = 0.0
    fit: str = "fill"
    ref_aspect: float = REF_ASPECT

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "View":
        d = dict(d or {})
        kw: dict[str, Any] = {}
        if "center" in d:
            kw["center"] = to_complex(d.pop("center"))
        if "zoom" in d:  # zoom = 1 / half_width, handy on the CLI
            kw["half_width"] = 1.0 / float(d.pop("zoom"))
        if "half_width" in d:
            kw["half_width"] = float(d.pop("half_width"))
        if "rotation" in d:
            kw["rotation"] = math.degrees(angle(d.pop("rotation")))
        if "fit" in d:
            fit = str(d.pop("fit"))
            if fit not in FITS:
                raise ValueError(f"view.fit must be one of {FITS}, got {fit!r}")
            kw["fit"] = fit
        if "ref_aspect" in d:
            kw["ref_aspect"] = float(d.pop("ref_aspect"))
        if d:
            raise ValueError(f"Unknown view keys: {sorted(d)}")
        return cls(**kw)

    def with_(self, **kw: Any) -> "View":
        return replace(self, **kw)

    def extents(self, width: int, height: int) -> tuple[float, float]:
        """(half_width, half_height) of this frame in plane units."""
        aspect = width / float(height)
        hw = self.half_width
        if self.fit == "cover":
            hh = hw / self.ref_aspect
            hw = hh * aspect
        elif self.fit == "contain":
            if aspect >= self.ref_aspect:
                hh = hw / self.ref_aspect
                hw = hh * aspect
            else:
                hh = hw / aspect
        else:
            hh = hw / aspect
        return hw, hh

    def plane(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (real, imag) meshgrids, row 0 at the top."""
        hw, hh = self.extents(width, height)
        c = self.center
        if not self.rotation:
            re_ = np.linspace(c.real - hw, c.real + hw, width, dtype=np.float64)
            im_ = np.linspace(c.imag + hh, c.imag - hh, height, dtype=np.float64)
            return np.meshgrid(re_, im_)
        u = np.linspace(-hw, hw, width, dtype=np.float64)
        v = np.linspace(hh, -hh, height, dtype=np.float64)
        U, V = np.meshgrid(u, v)
        th = math.radians(self.rotation)
        ct, st = math.cos(th), math.sin(th)
        return c.real + U * ct - V * st, c.imag + U * st + V * ct


def fit_window(
    xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, fill: str = "auto"
) -> tuple[float, float, float, float]:
    """Grow (contain) or crop (cover) a data window to the frame aspect."""
    aspect = width / float(height)
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    if fill == "auto":
        fill = "cover" if aspect < 1.0 else "contain"
    if fill == "cover":
        if dx / dy > aspect:
            new_dx = dy * aspect
            mid = 0.5 * (xmin + xmax)
            xmin, xmax = mid - new_dx / 2, mid + new_dx / 2
        else:
            new_dy = dx / aspect
            mid = 0.5 * (ymin + ymax)
            ymin, ymax = mid - new_dy / 2, mid + new_dy / 2
    else:
        if dx / dy > aspect:
            extra = dx / aspect - dy
            ymin -= extra / 2
            ymax += extra / 2
        else:
            extra = dy * aspect - dx
            xmin -= extra / 2
            xmax += extra / 2
    return xmin, xmax, ymin, ymax


def oblique_projection(x, y, z, angle_deg: float = 32.0, z_scale: float = 0.7, tilt: float = 0.2):
    """Cheap isometric-style 3D -> 2D used by the trajectory attractors.

    ``angle_deg`` spins the x/y plane, ``z_scale`` sets vertical stretch and
    ``tilt`` how much the spun plane leans into the vertical axis.
    """
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    px = x * ca + y * sa
    py = z * z_scale - tilt * x * sa + tilt * y * ca
    return px, py


def rotate_yaw_pitch(x, y, z, yaw: float, pitch: float):
    """Rotate points by ``yaw`` about Y then ``pitch`` about X (radians).

    Returns (x, y, depth) in camera space.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x1 = x * cy + z * sy
    z1 = -x * sy + z * cy
    y1 = y * cp - z1 * sp
    z2 = y * sp + z1 * cp
    return x1, y1, z2


def orbit_camera(distance: float, azimuth_deg: float, elevation_deg: float, target=(0.0, 0.0, 0.0)):
    """Camera position orbiting ``target`` (Y up). Handy for raymarchers."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    tx, ty, tz = target
    return (
        tx + distance * math.cos(el) * math.sin(az),
        ty + distance * math.sin(el),
        tz + distance * math.cos(el) * math.cos(az),
    )
