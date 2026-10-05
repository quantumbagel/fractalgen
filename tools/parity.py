#!/usr/bin/env python3
"""Compare new recipe renders against baselines rendered by the old code.

    python tools/parity.py BASELINE_DIR [names...]

BASELINE_DIR holds ``<name>.land.npy`` (384x216) and ``<name>.port.npy``
(216x468) uint8 renders. Prints per-recipe max / mean abs difference and the
fraction of pixels that moved by more than 2/255.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fractalgen import RECIPES, bootstrap  # noqa: E402
from fractalgen.color import to_uint8  # noqa: E402

SIZES = {"land": (384, 216), "port": (216, 468)}


def main() -> int:
    base = Path(sys.argv[1])
    bootstrap()
    names = sys.argv[2:] or RECIPES.names()
    bad = 0
    for name in names:
        for tag, (w, h) in SIZES.items():
            ref_path = base / f"{name}.{tag}.npy"
            if not ref_path.exists():
                print(f"{name:28s} {tag}  (no baseline)")
                continue
            ref = np.load(ref_path).astype(np.int16)
            t0 = time.perf_counter()
            try:
                new = to_uint8(RECIPES.get(name).render(w, h)).astype(np.int16)
            except Exception as exc:  # report and keep going
                print(f"{name:28s} {tag}  ERROR {type(exc).__name__}: {exc}")
                bad += 1
                continue
            d = np.abs(new - ref)
            frac = float((d > 2).mean())
            ok = frac < 0.001
            bad += not ok
            print(f"{name:28s} {tag}  max={int(d.max()):3d} mean={d.mean():7.4f} moved={frac:7.4%}  "
                  f"{'ok' if ok else 'DIFF'}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    print(f"{bad} mismatching render(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
