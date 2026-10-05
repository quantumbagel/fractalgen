"""Parallel ultra-quality wallpaper renderer.

Process-pool across recipes; optional same-process row-strip splitting for
escape-time / per-pixel fractal kernels so 16K frames do not keep several
full-size float64 working arrays live at once.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# 16:9 desktop + ~9:19.5 iPhone 14/15/16 Pro portrait.
PRESETS: dict[str, dict[str, Any]] = {
    "desktop-4k": {"width": 3840, "height": 2160, "subdir": "desktop_4k"},
    "desktop-8k": {"width": 7680, "height": 4320, "subdir": "desktop_8k"},
    "desktop-16k": {"width": 15360, "height": 8640, "subdir": "desktop_16k"},
    "iphone": {"width": 1290, "height": 2796, "subdir": "iphone"},
    "iphone-8k": {"width": 3840, "height": 8320, "subdir": "iphone_8k"},
    "iphone-16k": {"width": 7680, "height": 16640, "subdir": "iphone_16k"},
}

# ~33 MP (8K class) and ~128 MP (16K class).
_PX_8K = 20_000_000
_PX_16K = 80_000_000

_STRIP_FNS = (
    "escape_time",
    "orbit_trap",
    "newton_basins",
    "newton_shifted",
    "lyapunov",
    "domain_color",
    "popcorn_field",
)

_strips_installed = False


def default_out_dir(preset: str | None, root: Path | None = None) -> Path:
    base = root if root is not None else ROOT
    if preset is None:
        return base / "backgrounds"
    spec = PRESETS[preset]
    return base / "backgrounds" / spec["subdir"]


def default_jobs(preset: str | None, width: int, height: int) -> int:
    """One live recipe on 16 GB; two only for small iPhone-native / 4K frames."""
    if preset is not None:
        if "16k" in preset or "8k" in preset:
            return 1
        return 2
    px = width * height
    if px >= _PX_8K:
        return 1
    return 2


def default_format(preset: str | None, width: int, height: int) -> str:
    """JPG at 16K to save disk; PNG otherwise (including 8K)."""
    if preset is not None:
        return "jpg" if "16k" in preset else "png"
    return "jpg" if width * height >= _PX_16K else "png"


def default_n_strips(width: int, height: int) -> int:
    """Horizontal strips inside a worker; keep ~1/4–1/8 of a huge frame live."""
    px = width * height
    if px >= _PX_16K:
        return 8
    if px >= _PX_8K:
        return 4
    return 1


def _strip_bounds(height: int, n_strips: int) -> list[tuple[int, int]]:
    n_strips = max(1, min(int(n_strips), height))
    base, rem = divmod(height, n_strips)
    bounds: list[tuple[int, int]] = []
    y = 0
    for i in range(n_strips):
        span = base + (1 if i < rem else 0)
        bounds.append((y, y + span))
        y += span
    return bounds


def _leading_hw_count(args: tuple[Any, ...]) -> int:
    if not args or not isinstance(args[0], np.ndarray) or args[0].ndim < 2:
        return 0
    h, w = args[0].shape[:2]
    n = 0
    for a in args:
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[0] == h and a.shape[1] == w:
            n += 1
        else:
            break
    return n


def _alloc_full(height: int, piece: Any, y0: int, y1: int) -> Any:
    def one(p: np.ndarray) -> np.ndarray:
        out = np.empty((height,) + p.shape[1:], dtype=p.dtype)
        out[y0:y1] = p
        return out

    if isinstance(piece, tuple):
        return tuple(one(p) for p in piece)
    return one(piece)


def _place(full: Any, y0: int, y1: int, piece: Any) -> None:
    if isinstance(piece, tuple):
        for dst, src in zip(full, piece):
            dst[y0:y1] = src
    else:
        full[y0:y1] = piece


def _row_strip_wrap(fn: Callable[..., Any], n_strips: int) -> Callable[..., Any]:
    """Same-process numpy row slices; no nested process pool."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        n_arr = _leading_hw_count(args)
        if n_arr == 0:
            return fn(*args, **kwargs)
        height = int(args[0].shape[0])
        bounds = _strip_bounds(height, n_strips)
        if len(bounds) <= 1:
            return fn(*args, **kwargs)

        full: Any = None
        rest = args[n_arr:]
        for y0, y1 in bounds:
            sliced = tuple(a[y0:y1] for a in args[:n_arr]) + rest
            piece = fn(*sliced, **kwargs)
            if full is None:
                full = _alloc_full(height, piece, y0, y1)
            else:
                _place(full, y0, y1, piece)
            del piece
        return full

    wrapped.__name__ = getattr(fn, "__name__", "stripped")
    wrapped.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapped


def install_row_strips(n_strips: int) -> None:
    """Patch generate + bg.fractals kernels in this worker process."""
    global _strips_installed
    if n_strips <= 1 or _strips_installed:
        return

    import generate as gen
    import bg.fractals as fractals

    for name in _STRIP_FNS:
        orig = getattr(fractals, name)
        wrapped = _row_strip_wrap(orig, n_strips)
        setattr(fractals, name, wrapped)
        if hasattr(gen, name):
            setattr(gen, name, wrapped)
    _strips_installed = True


def save_rgb(path: str | Path, rgb: np.ndarray, fmt: str, jpeg_quality: int = 95) -> None:
    from PIL import Image

    from bg.color import to_image

    Image.MAX_IMAGE_PIXELS = None
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = to_image(rgb)
    fmt = fmt.lower()
    if fmt in ("jpg", "jpeg"):
        img.save(dest, format="JPEG", quality=jpeg_quality, optimize=False)
    else:
        # optimize=True is RAM-heavy on 8K/16K; keep it for smaller frames only.
        optimize = img.size[0] * img.size[1] < 8_000_000
        img.save(dest, format="PNG", optimize=optimize, compress_level=6)


def render_wallpaper(job: dict[str, Any]) -> dict[str, Any]:
    """Top-level picklable worker. Import RECIPES inside for macOS spawn."""
    name = job["name"]
    t0 = time.perf_counter()
    try:
        from generate import RECIPES

        n_strips = int(job.get("n_strips") or 1)
        install_row_strips(n_strips)
        rgb = RECIPES[name](int(job["width"]), int(job["height"]))
        save_rgb(job["out_path"], rgb, job["fmt"], jpeg_quality=int(job.get("jpeg_quality") or 95))
        del rgb
        return {
            "ok": True,
            "name": name,
            "path": job["out_path"],
            "sec": time.perf_counter() - t0,
        }
    except Exception:
        return {
            "ok": False,
            "name": name,
            "path": job.get("out_path"),
            "sec": time.perf_counter() - t0,
            "error": traceback.format_exc(),
        }


def render_batch(
    names: list[str],
    width: int,
    height: int,
    out_dir: Path,
    jobs: int,
    fmt: str,
    skip_existing: bool = False,
    jpeg_quality: int = 95,
    n_strips: int | None = None,
) -> int:
    """Render *names* in a process pool. Returns 0 if every job succeeded."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"
    strips = default_n_strips(width, height) if n_strips is None else n_strips
    cpus = os.cpu_count() or 1
    workers = max(1, min(jobs, cpus, max(1, len(names))))
    if width * height >= _PX_8K:
        workers = min(workers, 1)
    else:
        workers = min(workers, 2)

    pending: list[dict[str, Any]] = []
    skipped = 0
    for name in names:
        path = out_dir / f"{name}.{ext}"
        if skip_existing and path.exists():
            skipped += 1
            print(f"  {name}  (skip existing)", flush=True)
            continue
        pending.append(
            {
                "name": name,
                "width": width,
                "height": height,
                "out_path": str(path),
                "fmt": ext,
                "jpeg_quality": jpeg_quality,
                "n_strips": strips,
            }
        )

    total = len(pending)
    print(
        f"Rendering {total} wallpaper(s) at {width}x{height} → {out_dir}  "
        f"jobs={workers}  format={ext}  strips={strips}"
        + (f"  skipped={skipped}" if skipped else ""),
        flush=True,
    )
    if total == 0:
        return 0

    failures = 0
    done = 0
    # Recycle each worker after one wallpaper so 16K allocations return to the OS.
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(render_wallpaper, job): job for job in pending}
        for fut in as_completed(futures):
            done += 1
            job = futures[fut]
            try:
                result = fut.result()
            except Exception:
                failures += 1
                print(
                    f"[{done}/{total}] {job['name']} ... FAILED ({0.0:.1f}s)",
                    flush=True,
                )
                traceback.print_exc()
                continue
            sec = result.get("sec", 0.0)
            if result.get("ok"):
                saved = Path(result["path"]).name
                print(
                    f"[{done}/{total}] {result['name']} ... saved {saved} ({sec:.1f}s)",
                    flush=True,
                )
            else:
                failures += 1
                print(
                    f"[{done}/{total}] {result['name']} ... FAILED ({sec:.1f}s)",
                    flush=True,
                )
                err = result.get("error") or ""
                if err:
                    print(err, end="" if err.endswith("\n") else "\n", flush=True)

    return 1 if failures else 0
