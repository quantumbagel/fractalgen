#!/usr/bin/env python3
"""Browser end-to-end check of the GUI (headless Chromium via Playwright).

    pip install playwright && playwright install chromium-headless-shell
    python tools/gui_check.py [--all] [--port 8821]

Starts `python -m fractalgen gui`, drives the page like a user (edits, undo,
drag / wheel / rotate, palettes, recipe + generator + aspect switching, URL
hash reload, save, export) and checks the results. `--all` also loads every
recipe in the page. Exits non-zero on any failure. Not part of the unit tests
because it needs a browser.
"""
import io
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
ap.add_argument("--all", action="store_true", help="also load every recipe in the page")
ap.add_argument("--port", default="8821")
ap.add_argument("--screenshot", type=Path, help="save a final screenshot here")
args = ap.parse_args()
PORT, ALL = args.port, args.all

failures, logs = [], []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        failures.append(msg)


srv = subprocess.Popen([sys.executable, "-m", "fractalgen", "gui", "--no-browser", "--port", PORT],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(2.5)


def js(page, expr, arg=None):
    return page.evaluate(expr, arg)


def idle(page, timeout=120000):
    page.wait_for_timeout(250)
    page.wait_for_function("!document.getElementById('renderBadge').classList.contains('show')"
                           " && document.getElementById('previewImg').naturalWidth > 0", timeout=timeout)
    page.wait_for_timeout(100)


def frame_shot(page):
    return np.asarray(Image.open(io.BytesIO(page.locator("#previewFrame").screenshot())).convert("RGB"), dtype=float)


def rankdiff(a, b):
    from PIL import ImageFilter

    def ranks(x):
        img = Image.fromarray(x.astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))
        l = (np.asarray(img, dtype=float) @ [0.2126, 0.7152, 0.0722]).ravel()
        return np.argsort(np.argsort(l, kind="stable"), kind="stable") / l.size
    return float(np.abs(ranks(a) - ranks(b)).mean())


def row_input(page, label, kind="number"):
    """The input of the sidebar row whose label text is `label`."""
    return page.locator(f".row:has(> label.lbl:text-is('{label}')) input[type={kind}]").first


def plane_at(page, x, y):
    return js(page, "([x, y]) => { const p = planeFromEvent({clientX: x, clientY: y}); return [p.re, p.im]; }", [x, y])


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1500, "height": 900})
        page.on("console", lambda m: logs.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        page.on("response", lambda r: logs.append(f"HTTP {r.status} {r.url}") if r.status >= 400 and r.status != 409 else None)
        page.goto(f"http://127.0.0.1:{PORT}/")
        idle(page)
        check(js(page, "state.recipe") == "julia_dragon", "boots on julia_dragon")

        print("number box / slider edits stick")
        mi = row_input(page, "max_iter")
        mi.fill("200")
        mi.dispatch_event("change")
        idle(page)
        check(js(page, "state.params.max_iter") == 200, f"number box -> 200 (got {js(page, 'state.params.max_iter')})")
        js(page, """() => { const s = [...document.querySelectorAll('.row')].find(r => r.querySelector('label')?.textContent === 'gamma').querySelector('input[type=range]');
                    s.value = 1.5; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); }""")
        idle(page)
        check(abs(js(page, "state.params.gamma") - 1.5) < 1e-9, f"slider release keeps 1.5 (got {js(page, 'state.params.gamma')})")
        check(js(page, "document.querySelector('.row.modified') !== null"), "edited rows marked modified")

        print("undo / redo")
        page.click("#btnUndo")
        idle(page)
        check(abs(js(page, "state.params.gamma") - 0.85) < 1e-9 and js(page, "state.params.max_iter") == 200, "undo reverts gamma only")
        page.click("#btnUndo")
        idle(page)
        check(js(page, "state.params.max_iter") == 480, "second undo reverts max_iter")
        page.click("#btnRedo")
        page.click("#btnRedo")
        idle(page)
        check(js(page, "state.params.max_iter") == 200 and abs(js(page, "state.params.gamma") - 1.5) < 1e-9, "redo x2 restores both")
        check(js(page, "document.getElementById('btnRedo').disabled"), "redo disabled at end")

        print("view: drag pan / wheel zoom keep the grabbed point under the cursor")
        box = page.locator("#previewOverlay").bounding_box()
        for rot in (0, 35):
            rv = row_input(page, "rotation")
            rv.fill(str(rot))
            rv.dispatch_event("change")
            idle(page)
            x0, y0 = round(box["x"] + box["width"] * 0.4), round(box["y"] + box["height"] * 0.45)
            x1, y1 = x0 + 120, y0 + 70
            p0 = plane_at(page, x0, y0)
            page.mouse.move(x0, y0)
            page.mouse.down()
            page.mouse.move(x1, y1, steps=6)
            page.mouse.up()
            p1 = plane_at(page, x1, y1)
            err = max(abs(p0[0] - p1[0]), abs(p0[1] - p1[1]))
            check(err < 1e-9, f"pan at rotation {rot}: point stays under cursor (err {err:.2e})")
            idle(page)
            xw, yw = round(box["x"] + box["width"] * 0.7), round(box["y"] + box["height"] * 0.3)
            q0 = plane_at(page, xw, yw)
            hw0 = js(page, "state.view.half_width")
            page.mouse.move(xw, yw)
            page.mouse.wheel(0, -200)
            page.wait_for_timeout(50)
            q1 = plane_at(page, xw, yw)
            hw1 = js(page, "state.view.half_width")
            err = max(abs(q0[0] - q1[0]), abs(q0[1] - q1[1]))
            check(hw1 < hw0, f"wheel up zooms in ({hw0:.3f} -> {hw1:.3f})")
            check(err < 1e-9, f"zoom at rotation {rot}: point stays under cursor (err {err:.2e})")
            idle(page)

        print("interim transform matches the fresh render")
        page.click("#btnReset")
        idle(page)
        for desc, change in [("pan", "v.center = [v.center[0] + 0.15, v.center[1] - 0.08]"),
                             ("rotate", "v.rotation = (v.rotation || 0) + 20"),
                             ("zoom", "v.half_width *= 0.75")]:
            a0 = frame_shot(page)
            js(page, f"() => {{ const v = state.view; {change}; updatePreviewTransform(); }}")
            page.wait_for_timeout(100)
            a = frame_shot(page)
            js(page, "() => { seq++; scheduleRender(); }")
            idle(page)
            bb = frame_shot(page)
            h, w = a.shape[:2]
            c = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
            # Histogram coloring renormalizes per frame, so compare rank-normalized luminance.
            d, d0 = rankdiff(a[c], bb[c]), rankdiff(a0[c], bb[c])
            check(d < 0.4 * d0, f"{desc}: transformed preview ~ rendered (rank diff {d:.3f} vs untransformed {d0:.3f})")
        page.click("#btnReset")
        idle(page)

        print("palettes")
        pal0 = js(page, "state.palette")
        post_sel = page.locator(".section:has(h3:text-is('Post')) select.palette-select").first
        check(post_sel.input_value() == "", "post palette remap shows off by default")
        post_sel.select_option("ember")
        idle(page)
        check(js(page, "state.post.palette") == "ember" and js(page, "state.palette") == pal0,
              f"post remap sets post.palette only (post={js(page, 'state.post.palette')}, palette={js(page, 'state.palette')})")
        page.locator(".section:has(h3:text-is('Post')) select.palette-select").first.select_option("")
        idle(page)
        check(js(page, "state.post.palette") is None, "post remap back off")
        page.locator(".section:has(h3:text-is('Parameters')) button:text('⇄ reverse')").first.click()
        idle(page)
        check(js(page, "state.palette") == f"{pal0}_r", f"reverse named palette -> {js(page, 'state.palette')}")
        page.locator(".section:has(h3:text-is('Parameters')) details.stops-edit summary").first.click()
        idle(page)
        check(isinstance(js(page, "state.palette"), list), "edit stops converts to inline stops")
        n = js(page, "state.palette.length")
        page.locator(".section:has(h3:text-is('Parameters')) .stop-row button:text('+')").first.click()
        idle(page)
        check(js(page, "state.palette.length") == n + 1, "add stop")
        tint = page.locator(".row:has(> label.lbl:text-is('tint')) input[type=checkbox]").first
        tint.check()
        idle(page)
        check(js(page, "state.post.tint") is not None, "tint enable")
        tint.uncheck()
        idle(page)
        check(js(page, "state.post.tint") is None, "tint disable")

        print("recipe search / generator picker / aspect / reset")
        page.click("#recipeSearch")
        page.fill("#recipeSearch", "clifford")
        js(page, "() => { filterRecipeList('clifford'); openRecipeList(); }")
        page.locator("#recipeList button.recipe-row", has_text="clifford").first.click(force=True)
        idle(page)
        check("clifford" in (js(page, "state.recipe") or ""), f"picked {js(page, 'state.recipe')}")
        page.select_option("#genPicker", "domain_color")
        idle(page)
        check(js(page, "state.recipe") is None and js(page, "state.generator") == "domain_color", "generator picker")
        page.click("#btnUndo")
        idle(page)
        check("clifford" in (js(page, "state.recipe") or ""), "undo returns to the recipe")
        js(page, "() => selectRecipe('julia_dragon')")
        idle(page)
        page.select_option("#aspectSel", "phone")
        idle(page)
        check(js(page, "state.height > state.width"), "phone aspect is portrait")
        check(js(page, "document.querySelectorAll('.row.modified').length") == 0, "portrait overrides not flagged as modified")
        mi = row_input(page, "max_iter")
        mi.fill("150")
        mi.dispatch_event("change")
        idle(page)
        page.click("#btnReset")
        idle(page)
        check(js(page, "deepEq(state.params, baseline.params)"), "reset restores recipe values")

        print("hash round trip")
        mi = row_input(page, "max_iter")
        mi.fill("321")
        mi.dispatch_event("change")
        idle(page)
        page.reload()
        idle(page)
        check(js(page, "state.params.max_iter") == 321 and js(page, "aspectKey") == "phone", "reload restores edits + aspect")
        page.select_option("#aspectSel", "desktop")
        idle(page)

        print("save / export")
        name = "_e2e_tmp"
        target = ROOT / "custom" / f"{name}.toml"
        target.unlink(missing_ok=True)
        page.click("#btnSave")
        page.fill("#saveName", name)
        page.click("#saveGo")
        page.wait_for_timeout(1500)
        check(target.exists() and "max_iter = 321" in target.read_text(), "save writes custom recipe with the edit")
        check(js(page, f"catalog.recipes.some(r => r.name === '{name}')"), "saved recipe appears in catalog")
        target.unlink(missing_ok=True)
        page.click("#saveCancel")  # the dialog stays open to show where it saved
        page.click("#btnExport")
        page.fill("#exportW", "160")
        page.fill("#exportH", "90")
        page.fill("#exportName", "_e2e")
        page.click("#exportGo")
        page.wait_for_function("document.getElementById('exportStatus').textContent.startsWith('Saved') || "
                               "document.getElementById('exportStatus').textContent.startsWith('Error')", timeout=60000)
        st = page.inner_text("#exportStatus")
        check(st.startswith("Saved"), f"export: {st[:120]}")
        if st.startswith("Saved"):
            out = Path(st.split("Saved: ", 1)[1].strip())
            check(out.exists() and Image.open(out).size == (160, 90), "export file is 160x90")
            out.unlink(missing_ok=True)
        page.click("#exportCancel")

        print("mask")
        n_rows = page.locator(".row[data-path^='mask.']").count()
        before = frame_shot(page)
        text = page.locator(".row:has(> label.lbl:text-is('text')) input[type=text]").first
        text.fill("HI")
        text.press("Enter")
        idle(page)
        check(js(page, "state.mask.text") == "HI", "typing text turns the mask on")
        check(page.locator(".row[data-path^='mask.']").count() > n_rows, "mask controls appear once it is on")
        check(rankdiff(before, frame_shot(page)) > 0.02, "masked preview differs")
        bg = page.locator(".row:has(> label.lbl:text-is('background')) input[type=text]").first
        bg.fill("transparent")
        bg.press("Enter")
        idle(page)
        check(js(page, "document.getElementById('previewFrame').classList.contains('transparent')"),
              "transparent background shows the checkerboard")
        page.click("#btnUndo")
        page.click("#btnUndo")
        idle(page)
        check(not js(page, "maskActive()"), "undo turns the mask off again")

        print("last-click-wins load token")
        delayed = {"n": 0}

        def delay_first_state(route):
            delayed["n"] += 1
            if delayed["n"] == 1:
                resp = route.fetch()
                page.wait_for_timeout(1100)
                try:
                    route.fulfill(response=resp)
                except Exception:
                    pass
            else:
                route.continue_()

        page.route("**/api/state*", delay_first_state)
        try:
            js(page, "() => { void selectRecipe('julia_dragon'); void selectRecipe('mandelbrot_spiral'); }")
            idle(page)
            page.wait_for_timeout(1300)
            check(js(page, "state.recipe") == "mandelbrot_spiral",
                  f"last recipe click wins (got {js(page, 'state.recipe')})")
            js(page, "() => selectRecipe('julia_dragon')")
            idle(page)
            delayed["n"] = 0
            js(page, """() => {
              document.getElementById('aspectSel').value = 'phone';
              void onAspectChange();
              document.getElementById('aspectSel').value = 'desktop';
              void onAspectChange();
            }""")
            idle(page)
            page.wait_for_timeout(1300)
            vw = js(page, "({a: aspectKey, c: state.view.center, hw: state.view.half_width, w: state.width, h: state.height})")
            check(vw["a"] == "desktop" and vw["w"] > vw["h"], f"last aspect wins framing ({vw})")
            check(abs(vw["c"][0]) < 0.05 and abs(vw["c"][1] - 0.05) < 0.05 and abs(vw["hw"] - 1.45) < 0.05,
                  f"desktop view not stale phone ({vw})")
        finally:
            page.unroute("**/api/state*")

        print("nudge is undoable and keeps ints")
        mi0 = js(page, "state.params.max_iter")
        js(page, "() => { undoStack = []; redoStack = []; resetHistoryBase(); updateHistoryButtons(); }")
        page.click("#btnNudge")
        idle(page)
        mi1 = js(page, "state.params.max_iter")
        check(mi1 != mi0, f"nudge changed max_iter ({mi0} -> {mi1})")
        check(js(page, "Number.isInteger(state.params.max_iter)"), f"nudge keeps max_iter as int (got {mi1})")
        check(js(page, "undoStack.length > 0"), "nudge pushed an undo step")
        page.click("#btnUndo")
        idle(page)
        check(js(page, "state.params.max_iter") == mi0, f"undo restores pre-nudge max_iter (got {js(page, 'state.params.max_iter')})")

        print("complex sub-row reset")
        js(page, "() => { state.view.center = [0.5, 0.25]; commitChange('view.center'); }")
        idle(page)
        page.locator(".row[data-path='view.center.0'] .reset-row").click()
        idle(page)
        c = js(page, "state.view.center")
        b0 = js(page, "baseline.view.center[0]")
        check(abs(c[0] - b0) < 1e-12, f"reset real restores baseline ({c[0]} vs {b0})")
        check(abs(c[1] - 0.25) < 1e-12, f"reset real leaves imag ({c[1]})")

        print("export rejects negative size")
        page.click("#btnExport")
        page.fill("#exportW", "-32")
        page.fill("#exportH", "48")
        page.click("#exportGo")
        page.wait_for_timeout(300)
        est = page.inner_text("#exportStatus")
        check("at least 16" in est.lower(), f"export rejects negative size ({est[:120]})")
        page.click("#exportCancel")

        print("half_width rejects non-positive")
        hw0 = js(page, "state.view.half_width")
        js(page, """() => {
          const num = document.querySelector('[data-path="view.half_width"] input[type=number]');
          num.value = '-2';
          num.dispatchEvent(new Event('change'));
        }""")
        page.wait_for_timeout(100)
        check(js(page, "state.view.half_width") == hw0 and hw0 > 0,
              f"negative half_width ignored (still {js(page, 'state.view.half_width')})")

        print("phone-width topbar sizes to content")
        page.set_viewport_size({"width": 390, "height": 700})
        page.wait_for_timeout(250)
        top = js(page, """() => {
          const el = document.getElementById('topbar');
          const cs = getComputedStyle(el);
          return {h: el.getBoundingClientRect().height, sh: el.scrollHeight, basis: cs.flexBasis, overflow: cs.overflow};
        }""")
        check(top["h"] + 1 >= top["sh"], f"phone topbar not clipped (h={top['h']:.0f} scroll={top['sh']})")
        page.set_viewport_size({"width": 1500, "height": 900})
        page.wait_for_timeout(200)

        if ALL:
            print("every recipe renders in the GUI")
            page.evaluate("() => { document.getElementById('previewSize').value = 320; }")
            names = js(page, "catalog.recipes.map(r => r.name)")
            bad = []
            for i, n in enumerate(names):
                js(page, f"() => selectRecipe({n!r})")
                try:
                    idle(page, timeout=180000)
                except Exception as e:
                    bad.append(f"{n}: timeout")
                    continue
                err = js(page, "document.getElementById('errHost').innerText")
                if err.strip() or js(page, "state.recipe") != n:
                    bad.append(f"{n}: {err.strip()[:200]}")
            check(not bad, f"{len(names)} recipes, failures: {bad}")
        if args.screenshot:
            page.screenshot(path=str(args.screenshot))
        b.close()
except Exception:  # a crash mid-run is a failure, not a silent pass (sys.exit below would hide it)
    import traceback

    traceback.print_exc()
    failures.append("crashed: " + traceback.format_exc().strip().splitlines()[-1])
finally:
    srv.terminate()
    out = srv.stdout.read().decode()
    check(not logs, f"no console errors / failed requests: {logs[:10]}")
    if "Traceback" in out:
        print("SERVER:", out[-3000:])
    print(f"\n{len(failures)} failure(s)")
    for f in failures:
        print(" -", f)
    sys.exit(1 if failures else 0)
