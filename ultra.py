#!/usr/bin/env python3
"""Parallel ultra / iPhone wallpaper renderer.

Presets
-------
4k          3840 x 2160   16:9 desktop
8k          7680 x 4320   16:9 desktop
16k        15360 x 8640   16:9 desktop
iphone      1290 x 2796   ~9:19.5  (iPhone 14/15/16 Pro)
iphone-8k   3840 x 8320   ~9:19.5  8K-class long edge
iphone-16k  7680 x 16640  ~9:19.5  16K-class long edge

Tiers skip failed / empty 4K looks. 16K always uses the VIP subset.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from collections import deque
from pathlib import Path

# Pin BLAS before any numpy import in this process or spawned workers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "backgrounds"

PRESETS: dict[str, tuple[int, int]] = {
    "4k": (3840, 2160),
    "8k": (7680, 4320),
    "16k": (15360, 8640),
    "iphone": (1290, 2796),
    "iphone-8k": (3840, 8320),
    "iphone-16k": (7680, 16640),
}

# Skip a recipe if it exceeds this wall time (seconds).
RECIPE_TIMEOUT_S = 20 * 60

# 16 GB machine: at most two live recipes. 16K stays at 1.
DEFAULT_JOBS: dict[str, int] = {
    "4k": 1,
    "iphone": 2,
    "8k": 2,
    "iphone-8k": 1,
    "16k": 1,
    "iphone-16k": 1,
}
JOB_CAP: dict[str, int] = {
    "4k": 2,
    "iphone": 2,
    "8k": 2,
    "iphone-8k": 1,
    "16k": 1,
    "iphone-16k": 1,
}
PIDFILE = Path("/tmp/cool-backgrounds-ultra.pid")

# 4K looks that are empty, noisy, toy-RGB, or mostly black.
SKIP_ULTRA: frozenset[str] = frozenset(
    {
        "tinkerbell",
        "ikeda",
        "ikeda_whirl",
        "julia_cauliflower",
        "flow_silk",
        "flow_swirl",
        "magnetic_pendulum",
        "magnetic_pendulum_close",
        "newton_cubic",
        "newton_shifted",
        "rossler",
        "apollonian",
        "harmonograph",
        "gray_scott_spots",
        "cyclic_cellular",
        "mandelbrot_elephant",
    }
)

# Dense, high-detail keepers worth 8K / iPhone-8K.
KEEP: frozenset[str] = frozenset(
    {
        "mandelbrot_cosmic",
        "mandelbrot_seahorse",
        "mandelbrot_spiral",
        "mandelbrot_antenna",
        "mandelbrot_minibrot",
        "mandelbrot_tendril",
        "mandelbrot_orbit_trap",
        "mandelbrot_orbit_cross",
        "burning_ship",
        "burning_ship_cathedral",
        "buffalo",
        "celtic",
        "tricorn",
        "multibrot3",
        "multibrot4",
        "phoenix",
        "phoenix_close",
        "julia_rabbit",
        "julia_swirl",
        "julia_dragon",
        "julia_snowflake",
        "julia_golden",
        "julia_dendrite",
        "julia_spirals",
        "julia_siegel",
        "julia_san_marco",
        "julia_airplane",
        "newton_octic",
        "newton_sextic_dusk",
        "newton_cubic_dusk",
        "newton_quartic",
        "newton_quintic",
        "newton_sextic",
        "lyapunov_logistic",
        "lyapunov_ab",
        "lyapunov_baba",
        "domain_rational",
        "domain_poly",
        "domain_sin",
        "domain_exp",
        "popcorn",
        "clifford_gold",
        "clifford_ice",
        "clifford_fire",
        "dejong_ember",
        "dejong_ice",
        "hopalong",
        "hopalong_alt",
        "gumowski_mira",
        "gumowski_alt",
        "svensson",
        "bedhead",
        "fractal_dream",
        "lorenz_butterfly",
        "aizawa",
        "thomas",
        "flow_silk_open",
        "flow_swirl_open",
        "gray_scott_coral",
        "gray_scott_worms",
        "gray_scott_maze",
        "gray_scott_bubbles",
        "cymatics",
        "cymatics_dense",
        "magnetic_pendulum_tri",
        "chladni",
        "chladni_high",
        "worley_cells",
        "worley_blobs",
        "barnsley_fern",
        "sandpile",
        "kerr_einstein_rings",
        "gray_scott_morphogenesis",
        "physarum_dark_matter",
        "poincare_mirage",
        "halvorsen_ribbon",
        "hydrogen_orbitals",
        "ferrofluid_rosettes",
        "mandelbulb_grotto",
    }
)

# 16K / iPhone-16K: finest structure, skip heavy-but-soft RD upsamples except maze/coral.
VIP: frozenset[str] = frozenset(
    {
        "mandelbrot_cosmic",
        "mandelbrot_seahorse",
        "mandelbrot_spiral",
        "mandelbrot_antenna",
        "mandelbrot_minibrot",
        "mandelbrot_tendril",
        "julia_swirl",
        "julia_dragon",
        "julia_snowflake",
        "julia_spirals",
        "julia_golden",
        "burning_ship_cathedral",
        "phoenix_close",
        "buffalo",
        "celtic",
        "newton_octic",
        "lyapunov_baba",
        "lyapunov_ab",
        "domain_exp",
        "popcorn",
        "hopalong_alt",
        "fractal_dream",
        "flow_silk_open",
        "gray_scott_maze",
        "gray_scott_coral",
        "magnetic_pendulum_tri",
        "sandpile",
        "barnsley_fern",
        "kerr_einstein_rings",
        "gray_scott_morphogenesis",
        "physarum_dark_matter",
        "poincare_mirage",
        "mandelbulb_grotto",
    }
)


def _select_names(all_names: list[str], tier: str, preset: str, only: list[str] | None) -> list[str]:
    if only:
        return [n for n in only if n in all_names]
    if preset in ("16k", "iphone-16k") or tier == "vip":
        return [n for n in all_names if n in VIP]
    if tier == "keep":
        return [n for n in all_names if n in KEEP]
    if tier == "all":
        return [n for n in all_names if n not in SKIP_ULTRA]
    return [n for n in all_names if n in KEEP]


def _render_one(payload: tuple) -> tuple[str, str, float, str]:
    """Worker: (name, width, height, out_dir, skip) -> (name, status, seconds, detail)."""
    name, width, height, out_dir, skip = payload
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "4")

    from PIL import Image
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from generate import RECIPES

    path = Path(out_dir) / f"{name}.png"
    if skip and path.exists():
        return name, "skip", 0.0, str(path)
    t0 = time.perf_counter()
    try:
        rgb = RECIPES[name](width, height)
        rgb8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(rgb8, mode="RGB").save(path, format="PNG", compress_level=1)
    except MemoryError:
        return name, "oom", time.perf_counter() - t0, "MemoryError"
    return name, "ok", time.perf_counter() - t0, str(path)


def _unlink_partial(out_dir: str, name: str) -> None:
    path = Path(out_dir) / f"{name}.png"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _subprocess_entry(payload: tuple, result_q) -> None:
    try:
        result_q.put(_render_one(payload))
    except Exception as exc:
        result_q.put((payload[0], "fail", 0.0, f"{type(exc).__name__}: {exc}"))


def _render_one_isolated(payload: tuple) -> tuple[str, str, float, str]:
    """Own process per recipe so timeout/OOM can skip without killing the batch."""
    ctx = multiprocessing.get_context("spawn")
    result_q = ctx.Queue(1)
    proc = ctx.Process(target=_subprocess_entry, args=(payload, result_q), daemon=True)
    t0 = time.perf_counter()
    proc.start()
    proc.join(RECIPE_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join(8)
        if proc.is_alive():
            proc.kill()
            proc.join(4)
        _unlink_partial(payload[3], payload[0])
        return payload[0], "timeout", time.perf_counter() - t0, "exceeded 20 minutes"
    if proc.exitcode not in (0, None):
        code = proc.exitcode
        reason = "OOM" if code in (-9, 9, -6, 137) else f"worker exit {code}"
        _unlink_partial(payload[3], payload[0])
        return payload[0], "fail", time.perf_counter() - t0, reason
    try:
        return result_q.get_nowait()
    except Exception:
        _unlink_partial(payload[3], payload[0])
        return payload[0], "fail", time.perf_counter() - t0, "no result (possible OOM)"


def _capped_jobs(preset: str, jobs: int) -> int:
    cap = JOB_CAP.get(preset, 1)
    return max(1, min(jobs, cap))


def _acquire_pidfile() -> None:
    if PIDFILE.exists():
        try:
            old = int(PIDFILE.read_text().strip())
        except ValueError:
            old = None
        if old and _pid_alive(old):
            raise SystemExit(
                f"Another ultra.py is already running (pid {old}). "
                "Stop it before starting a second batch."
            )
    PIDFILE.write_text(str(os.getpid()))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _release_pidfile() -> None:
    try:
        if PIDFILE.exists() and PIDFILE.read_text().strip() == str(os.getpid()):
            PIDFILE.unlink()
    except OSError:
        pass


def _run_preset(
    preset: str,
    names: list[str],
    jobs: int,
    skip_existing: bool,
    out_root: Path,
) -> int:
    width, height = PRESETS[preset]
    out_dir = out_root / preset
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = _capped_jobs(preset, jobs)
    print(
        f"\n=== {preset}  {width}x{height}  {len(names)} wallpaper(s)  jobs={jobs}  → {out_dir}",
        flush=True,
    )
    failed = 0
    done = 0
    t_all = time.perf_counter()
    pending: deque[tuple] = deque()
    for n in names:
        path = out_dir / f"{n}.png"
        if skip_existing and path.exists():
            done += 1
            print(f"[{done}/{len(names)}] {n}  (skip existing)", flush=True)
            continue
        pending.append((n, width, height, str(out_dir), skip_existing))
    live: list[tuple[multiprocessing.Process, multiprocessing.Queue, tuple, float]] = []
    ctx = multiprocessing.get_context("spawn")

    def harvest(proc, q, payload, t0, timeout=0.0) -> bool:
        nonlocal done, failed
        proc.join(timeout)
        if proc.is_alive():
            return False
        done += 1
        name = payload[0]
        if proc.exitcode not in (0, None):
            failed += 1
            _unlink_partial(payload[3], name)
            reason = "OOM" if proc.exitcode in (-9, 9, -6, 137) else f"worker exit {proc.exitcode}"
            print(f"[{done}/{len(names)}] {name}  SKIP fail  {reason}  ({time.perf_counter() - t0:.1f}s)", flush=True)
            return True
        try:
            n, status, dt, detail = q.get_nowait()
        except Exception:
            failed += 1
            _unlink_partial(payload[3], name)
            print(f"[{done}/{len(names)}] {name}  SKIP fail  no result  ({time.perf_counter() - t0:.1f}s)", flush=True)
            return True
        if status == "skip":
            print(f"[{done}/{len(names)}] {n}  (skip existing)", flush=True)
        elif status == "ok":
            print(f"[{done}/{len(names)}] {n}  saved  ({dt:.1f}s)", flush=True)
        else:
            failed += 1
            _unlink_partial(str(out_dir), n)
            print(f"[{done}/{len(names)}] {n}  SKIP {status}  {detail}  ({dt:.1f}s)", flush=True)
        return True

    while pending or live:
        while pending and len(live) < jobs:
            payload = pending.popleft()
            q = ctx.Queue(1)
            proc = ctx.Process(target=_subprocess_entry, args=(payload, q), daemon=True)
            proc.start()
            live.append((proc, q, payload, time.perf_counter()))
        still = []
        for item in live:
            proc, q, payload, t0 = item
            elapsed = time.perf_counter() - t0
            if elapsed > RECIPE_TIMEOUT_S and proc.is_alive():
                proc.terminate()
                proc.join(8)
                if proc.is_alive():
                    proc.kill()
                    proc.join(4)
                done += 1
                failed += 1
                _unlink_partial(payload[3], payload[0])
                print(
                    f"[{done}/{len(names)}] {payload[0]}  SKIP timeout  exceeded 20 minutes  ({elapsed:.1f}s)",
                    flush=True,
                )
                continue
            if not harvest(proc, q, payload, t0, timeout=0.4):
                still.append(item)
        live = still
    print(
        f"=== {preset} done in {time.perf_counter() - t_all:.1f}s  failed={failed}",
        flush=True,
    )
    return failed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Parallel 8K/16K + iPhone math wallpapers.")
    p.add_argument(
        "--preset",
        nargs="+",
        default=["iphone", "8k", "iphone-8k"],
        choices=list(PRESETS),
        help="One or more size presets (default: iphone, then 8k, then iphone-8k)",
    )
    p.add_argument(
        "--tier",
        choices=("keep", "vip", "all"),
        default="keep",
        help="keep=cool 4K looks, vip=16K subset, all=everything except known duds",
    )
    p.add_argument("--only", nargs="+", help="Generate only these recipe names")
    p.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Workers (0 = safe default). Hard-capped: 2 for iPhone/4K/8K, 1 for 16K",
    )
    p.add_argument("--out", type=Path, default=OUT_ROOT)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    from generate import RECIPES

    all_names = list(RECIPES)
    if args.list:
        for preset in args.preset:
            names = _select_names(all_names, args.tier, preset, args.only)
            print(f"{preset} ({len(names)}):")
            for n in names:
                print(f"  {n}")
        return 0

    _acquire_pidfile()
    failed = 0
    try:
        for preset in args.preset:
            names = _select_names(all_names, args.tier, preset, args.only)
            if not names:
                print(f"{preset}: nothing to render", file=sys.stderr)
                continue
            jobs = args.jobs if args.jobs > 0 else DEFAULT_JOBS[preset]
            failed += _run_preset(preset, names, jobs, args.skip_existing, args.out)
    finally:
        _release_pidfile()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
