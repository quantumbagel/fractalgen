"""Batch rendering: size presets, process isolation, timeouts, row strips."""

from __future__ import annotations

import multiprocessing
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import Registry

# Pin BLAS in this process and (via inheritance) in spawned workers.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "backgrounds"
PIDFILE = Path("/tmp/fractalgen-batch.pid")
DEFAULT_TIMEOUT_S = 20 * 60

_PX_8K = 20_000_000
_PX_16K = 80_000_000


@dataclass(frozen=True)
class Preset:
    name: str
    width: int
    height: int
    subdir: str = ""
    jobs: int = 1  # default workers
    max_jobs: int = 2  # hard memory cap
    format: str = "png"
    tags: tuple[str, ...] = ()  # default recipe selection when none is given
    exclude_tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


PRESETS: Registry[Preset] = Registry("preset")


def register_preset(name: str, table: dict[str, Any], *, replace: bool = False) -> Preset:
    t = dict(table)
    p = Preset(
        name=name,
        width=int(t.pop("width")),
        height=int(t.pop("height")),
        subdir=str(t.pop("subdir", name)),
        jobs=int(t.pop("jobs", 1)),
        max_jobs=int(t.pop("max_jobs", 2)),
        format=str(t.pop("format", "png")),
        tags=tuple(t.pop("tags", ())),
        exclude_tags=tuple(t.pop("exclude_tags", ())),
        aliases=tuple(t.pop("aliases", ())),
        description=str(t.pop("description", "")),
    )
    if t:
        raise ValueError(f"Unknown preset keys: {sorted(t)}")
    PRESETS.register(name, p, replace=replace)
    for alias in p.aliases:
        PRESETS.register(alias, p, replace=True)
    return p


def default_n_strips(width: int, height: int) -> int:
    """Row strips inside a worker; keep ~1/4-1/8 of a huge frame live."""
    px = width * height
    if px >= _PX_16K:
        return 8
    if px >= _PX_8K:
        return 4
    return 1


def default_format(width: int, height: int) -> str:
    return "jpg" if width * height >= _PX_16K else "png"


def default_jobs(width: int, height: int) -> int:
    return 1 if width * height >= _PX_8K else 2


@dataclass
class Job:
    name: str
    width: int
    height: int
    out_path: str
    fmt: str = "png"
    jpeg_quality: int = 95
    n_strips: int = 1
    overrides: dict[str, Any] = field(default_factory=dict)
    loads: list[str] = field(default_factory=list)


def run_job(job: Job) -> dict[str, Any]:
    """Render and save one recipe in *this* process."""
    from .color import save_image
    from .loader import bootstrap
    from .recipe import RECIPES

    t0 = time.perf_counter()
    try:
        bootstrap(job.loads)
        rgb = RECIPES.get(job.name).render(
            job.width, job.height, overrides=job.overrides, n_strips=job.n_strips
        )
        save_image(job.out_path, rgb, job.fmt, jpeg_quality=job.jpeg_quality)
        return {"status": "ok", "sec": time.perf_counter() - t0}
    except MemoryError:
        return {"status": "oom", "sec": time.perf_counter() - t0, "detail": "MemoryError"}
    except Exception:
        return {"status": "fail", "sec": time.perf_counter() - t0, "detail": traceback.format_exc()}


def _worker(job: Job, q) -> None:
    os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))
    q.put(run_job(job))


def _unlink(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def render_batch(
    names: list[str],
    width: int,
    height: int,
    out_dir: Path,
    *,
    jobs: int = 1,
    fmt: str = "png",
    skip_existing: bool = False,
    jpeg_quality: int = 95,
    n_strips: int | None = None,
    overrides: dict[str, Any] | None = None,
    loads: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    isolate: bool = True,
    label: str = "",
) -> int:
    """Render ``names`` to ``out_dir``. Returns the number of failures.

    With ``isolate`` each recipe runs in its own spawned process so a timeout
    or OOM kill skips that wallpaper instead of the whole batch.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"
    strips = default_n_strips(width, height) if n_strips is None else n_strips
    pending: list[Job] = []
    total = len(names)
    done = 0
    for n in names:
        path = out_dir / f"{n}.{ext}"
        if skip_existing and path.exists():
            done += 1
            print(f"[{done}/{total}] {n}  (skip existing)", flush=True)
            continue
        pending.append(Job(n, width, height, str(path), ext, jpeg_quality, strips, dict(overrides or {}), list(loads or [])))

    head = f"=== {label + '  ' if label else ''}{width}x{height}  {len(pending)} wallpaper(s)"
    print(f"{head}  jobs={jobs}  format={ext}  strips={strips}  -> {out_dir}", flush=True)
    t_all = time.perf_counter()
    failed = 0

    def report(job: Job, res: dict[str, Any]) -> None:
        nonlocal done, failed
        done += 1
        if res["status"] == "ok":
            print(f"[{done}/{total}] {job.name}  saved {Path(job.out_path).name}  ({res['sec']:.1f}s)", flush=True)
            return
        failed += 1
        _unlink(job.out_path)
        detail = str(res.get("detail", "")).rstrip()
        print(f"[{done}/{total}] {job.name}  FAILED {res['status']}  ({res['sec']:.1f}s)", flush=True)
        if detail:
            print("    " + detail.replace("\n", "\n    "), flush=True)

    if not isolate:
        for job in pending:
            report(job, run_job(job))
    else:
        ctx = multiprocessing.get_context("spawn")
        live: list[tuple[Any, Any, Job, float]] = []
        queue = list(pending)
        while queue or live:
            while queue and len(live) < max(1, jobs):
                job = queue.pop(0)
                q = ctx.Queue(1)
                proc = ctx.Process(target=_worker, args=(job, q), daemon=True)
                proc.start()
                live.append((proc, q, job, time.perf_counter()))
            still = []
            for proc, q, job, t0 in live:
                elapsed = time.perf_counter() - t0
                try:
                    res = q.get(timeout=0.2)
                except Exception:
                    res = None
                if res is not None:
                    proc.join(10)
                    report(job, res)
                elif elapsed > timeout_s:
                    proc.terminate()
                    proc.join(8)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(4)
                    report(job, {"status": "timeout", "sec": elapsed, "detail": f"exceeded {timeout_s:.0f}s"})
                elif not proc.is_alive():
                    code = proc.exitcode
                    reason = "OOM-killed" if code in (-9, 9, -6, 137) else f"worker exit {code}"
                    report(job, {"status": "fail", "sec": elapsed, "detail": reason})
                else:
                    still.append((proc, q, job, t0))
            live = still

    print(f"=== done in {time.perf_counter() - t_all:.1f}s  failed={failed}", flush=True)
    return failed


def acquire_lock() -> None:
    if PIDFILE.exists():
        try:
            old = int(PIDFILE.read_text().strip())
        except ValueError:
            old = None
        if old and old != os.getpid() and _alive(old):
            raise SystemExit(f"Another fractalgen batch is running (pid {old}). Stop it first, or pass --no-lock.")
    PIDFILE.write_text(str(os.getpid()))


def release_lock() -> None:
    try:
        if PIDFILE.exists() and PIDFILE.read_text().strip() == str(os.getpid()):
            PIDFILE.unlink()
    except OSError:
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
