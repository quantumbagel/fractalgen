#!/usr/bin/env python3
"""Browser end-to-end check of the GUI (headless Chromium via Playwright).

    pip install playwright && playwright install chromium-headless-shell
    python tools/gui_check.py [--all] [--port 8821]

Starts `python -m fractalgen gui`, drives the page like a user (edits, undo,
drag / wheel / rotate, palettes, browse, screen shapes, layers, effects,
variations, URL reload, save, export, image upload) and checks the results.
`--all` also loads every recipe in the page. Exits non-zero on any failure.
Not part of the unit tests because it needs a browser. The page exposes its
store as `window.fg`.
"""
import argparse
import io
import subprocess
import sys
import tempfile
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
    page.wait_for_function("window.fg && fg.S.state && !fg.S.view.busy && "
                           "document.querySelector('.frame img')?.naturalWidth > 0", timeout=timeout)
    page.wait_for_timeout(120)


def frame_shot(page):
    return np.asarray(Image.open(io.BytesIO(page.locator(".frame").screenshot())).convert("RGB"), dtype=float)


def rankdiff(a, b):
    from PIL import ImageFilter

    def ranks(x):
        img = Image.fromarray(x.astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))
        lum = (np.asarray(img, dtype=float) @ [0.2126, 0.7152, 0.0722]).ravel()
        return np.argsort(np.argsort(lum, kind="stable"), kind="stable") / lum.size
    return float(np.abs(ranks(a) - ranks(b)).mean())


def field(page, label):
    """The field whose label reads `label` (exact)."""
    return page.locator(f".field:has(> .field-head > .field-label:text-is('{label}'))").first


def tab(page, name):
    page.locator(f".tab:has(.tab-label:text-is('{name}'))").click()
    page.wait_for_timeout(150)


def st(page, path):
    return js(page, f"fg.S.state.{path}")


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1500, "height": 900})
        page.on("console", lambda m: logs.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        page.on("response", lambda r: logs.append(f"HTTP {r.status} {r.url}") if r.status >= 400 and r.status != 409 else None)
        page.goto(f"http://127.0.0.1:{PORT}/")
        idle(page)
        check(st(page, "recipe") == "julia_dragon", "boots on julia_dragon")
        check(js(page, "fg.S.aspect") == "screen", "defaults to this screen's shape")

        print("number box / slider edits stick")
        mi = field(page, "Detail (iterations)").locator("input[type=number]")
        mi.fill("200")
        mi.dispatch_event("change")
        idle(page)
        check(st(page, "params.max_iter") == 200, f"number box -> 200 (got {st(page, 'params.max_iter')})")
        js(page, """() => { const s = [...document.querySelectorAll('.field')].find(f => f.querySelector('.field-label')?.textContent === 'Gamma')
                    .querySelector('input[type=range]'); s.value = 1.5; s.dispatchEvent(new Event('input', {bubbles: true})); }""")
        idle(page)
        check(abs(st(page, "params.gamma") - 1.5) < 1e-9, f"slider -> 1.5 (got {st(page, 'params.gamma')})")
        check(js(page, "document.querySelector('.field.mod') !== null"), "edited fields marked modified")

        print("undo / redo")
        page.click("button[title^='Undo (']")
        idle(page)
        check(abs(st(page, "params.gamma") - 0.85) < 1e-9 and st(page, "params.max_iter") == 200, "undo reverts gamma only")
        page.click("button[title^='Undo (']")
        idle(page)
        check(st(page, "params.max_iter") == 480, "second undo reverts max_iter")
        page.click("button[title^='Redo (']")
        page.click("button[title^='Redo (']")
        idle(page)
        check(st(page, "params.max_iter") == 200 and abs(st(page, "params.gamma") - 1.5) < 1e-9, "redo x2 restores both")
        check(js(page, "document.querySelector(\"button[title^='Redo (']\").disabled"), "redo disabled at end")

        print("view: drag pan / wheel zoom keep the grabbed point under the cursor")
        tab(page, "Frame")
        box = page.locator(".frame .overlay").bounding_box()
        for rot in (0, 35):
            js(page, f"() => fg.setPath('view.rotation', {rot})")
            idle(page)
            x0, y0 = round(box["x"] + box["width"] * 0.4), round(box["y"] + box["height"] * 0.45)
            x1, y1 = x0 + 120, y0 + 70
            p0 = js(page, f"fg.planeFromClient({x0}, {y0})")
            page.mouse.move(x0, y0)
            page.mouse.down()
            page.mouse.move(x1, y1, steps=6)
            page.mouse.up()
            p1 = js(page, f"fg.planeFromClient({x1}, {y1})")
            err = max(abs(p0[0] - p1[0]), abs(p0[1] - p1[1]))
            check(err < 1e-9, f"pan at rotation {rot}: point stays under cursor (err {err:.2e})")
            idle(page)
            xw, yw = round(box["x"] + box["width"] * 0.7), round(box["y"] + box["height"] * 0.3)
            q0 = js(page, f"fg.planeFromClient({xw}, {yw})")
            hw0 = st(page, "view.half_width")
            page.mouse.move(xw, yw)
            page.mouse.wheel(0, -200)
            page.wait_for_timeout(60)
            q1 = js(page, f"fg.planeFromClient({xw}, {yw})")
            hw1 = st(page, "view.half_width")
            err = max(abs(q0[0] - q1[0]), abs(q0[1] - q1[1]))
            check(hw1 < hw0, f"wheel up zooms in ({hw0:.3f} -> {hw1:.3f})")
            check(err < 1e-9, f"zoom at rotation {rot}: point stays under cursor (err {err:.2e})")
            idle(page)

        print("interim transform matches the fresh render")
        js(page, "() => fg.resetAll()")
        idle(page)
        for desc, change in [("pan", "v.center = [v.center[0] + 0.15, v.center[1] - 0.08]"),
                             ("rotate", "v.rotation = (v.rotation || 0) + 20"),
                             ("zoom", "v.half_width *= 0.75")]:
            a0 = frame_shot(page)
            js(page, f"() => {{ clearTimeout(window.__t); const v = fg.S.state.view; {change}; fg.emit(); }}")
            page.wait_for_timeout(30)  # before the debounced render lands
            a = frame_shot(page)
            js(page, "() => fg.commit('test')")
            idle(page)
            bb = frame_shot(page)
            h, w = a.shape[:2]
            c = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
            # Histogram coloring renormalizes per frame, so compare rank-normalized luminance.
            d, d0 = rankdiff(a[c], bb[c]), rankdiff(a0[c], bb[c])
            check(d < 0.4 * d0, f"{desc}: transformed preview ~ rendered (rank diff {d:.3f} vs untransformed {d0:.3f})")
        js(page, "() => fg.resetAll()")
        idle(page)

        print("palettes and quick adjustments")
        tab(page, "Color")
        pal0 = st(page, "palette")
        page.locator(".pal:has(.pal-name:text-is('aurora'))").click()
        idle(page)
        check(st(page, "palette") == "aurora", f"palette tile -> {st(page, 'palette')}")
        page.locator("button:has-text('Reverse')").click()
        idle(page)
        check(st(page, "palette") == "aurora_r", f"reverse -> {st(page, 'palette')}")
        page.locator("button:text('Custom colors…')").click()
        page.locator(".stop button[title^='Add a color']").first.click()
        idle(page)
        check(isinstance(st(page, "palette"), list) and len(st(page, "palette")) >= 3, "custom stops edit")
        sat = field(page, "Saturation").locator("input[type=number]")
        sat.fill("1.4")
        sat.dispatch_event("change")
        idle(page)
        check(js(page, "fg.S.state.post.some(e => e.op === 'saturation' && e.value === 1.4)"), "quick saturation adds an effect")
        sat = field(page, "Saturation").locator("input[type=number]")
        sat.fill("1")
        sat.dispatch_event("change")
        idle(page)
        check(js(page, "!fg.S.state.post.some(e => e.op === 'saturation')"), "back to neutral removes it")
        js(page, f"() => fg.setPalette({pal0!r})")
        idle(page)

        print("browse / generator / undo")
        page.locator(".look-btn").click()
        page.locator(".browse-bar input").fill("clifford")
        page.locator(".browse-bar input").press("Enter")
        idle(page)
        check("clifford" in (st(page, "recipe") or ""), f"browse + Enter picked {st(page, 'recipe')}")
        tab(page, "Look")
        page.locator(".section-head:has(h3:text-is('Start from scratch'))").click()
        page.locator(".panel select:has(option:text('Choose a generator…'))").select_option("domain_color")
        idle(page)
        check(st(page, "recipe") is None and st(page, "generator") == "domain_color", "start from a generator")
        page.click("button[title^='Undo (']")
        idle(page)
        check("clifford" in (st(page, "recipe") or ""), "undo returns to the recipe")

        print("screen shapes")
        js(page, "() => fg.loadRecipe('julia_dragon')")
        idle(page)
        page.select_option(".screen-pick select", "phone")
        idle(page)
        check(js(page, "fg.S.state.height > fg.S.state.width"), "phone shape is portrait")
        check(js(page, "fg.S.modified.size") == 0, "portrait overrides not flagged as modified")
        js(page, "() => fg.setPath('params.max_iter', 150)")
        idle(page)
        page.click("button:text-is('Reset')")
        idle(page)
        check(js(page, "JSON.stringify(fg.S.state.params) === JSON.stringify(fg.S.baseline.params)"), "reset restores recipe values")
        page.click("button[title^='Undo (']")
        idle(page)
        check(st(page, "params.max_iter") == 150, "reset is undoable")

        print("hash round trip")
        js(page, "() => fg.setPath('params.max_iter', 321)")
        idle(page)
        page.reload()
        idle(page)
        check(st(page, "params.max_iter") == 321 and js(page, "fg.S.aspect") == "phone", "reload restores edits + shape")
        page.select_option(".screen-pick select", "desktop")
        idle(page)

        print("layers")
        js(page, "() => fg.loadRecipe('julia_dragon')")
        idle(page)
        tab(page, "Layers")
        before = frame_shot(page)
        page.locator(".add-row button:has-text('Text')").click()
        idle(page)
        ta = field(page, "Text").locator("textarea")
        ta.fill("HI")
        ta.dispatch_event("input")
        idle(page)
        check(st(page, "layers.length") == 1 and st(page, "layers[0].text") == "HI", "add a text layer and type")
        check(rankdiff(before, frame_shot(page)) > 0.02, "layered preview differs")
        check(js(page, "fg.S.view.boxes?.length") == 1, "server sends the layer's box")
        x_before = js(page, "fg.S.state.layers[0].x ?? 0.5")
        fb = page.locator(".frame .overlay").bounding_box()
        cx, cy = fb["x"] + fb["width"] / 2, fb["y"] + fb["height"] / 2
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.mouse.move(cx + fb["width"] * 0.2, cy, steps=5)
        page.mouse.up()
        idle(page)
        x_after = st(page, "layers[0].x")
        check(abs(x_after - x_before - 0.2) < 0.02, f"dragging the layer moves it ({x_before} -> {x_after})")
        page.keyboard.press("ArrowUp")
        idle(page)
        check(abs(st(page, "layers[0].y") - 0.49) < 1e-9, f"arrow key nudges ({st(page, 'layers[0].y')})")
        page.locator(".fill-picker button:text-is('Color')").first.click()
        idle(page)
        check(st(page, "layers[0].fill") == "ffffff", "fill -> color")
        page.locator(".fill-picker button:text-is('Fractal')").first.click()
        idle(page)
        page.locator(".fill-picker button:text-is('Transparent')").click()
        idle(page)
        check(js(page, "document.querySelector('.frame').classList.contains('transparent')"),
              "transparent background shows the checkerboard")
        page.locator(".item button[title='Duplicate']").first.click()
        idle(page)
        check(st(page, "layers.length") == 2, "duplicate")
        page.locator(".add-row button:has-text('Shape')").click()
        idle(page)
        page.locator(".shape-grid button[title='heart']").click()
        idle(page)
        check(st(page, "layers[2].shape") == "heart", "shape picker")
        js(page, "() => fg.moveLayer(fg.S.state.layers[2].id, -1)")
        idle(page)
        check(st(page, "layers[1].type") == "shape", "reorder")
        page.locator(".chip-btn:text-is('Neon')").click()
        idle(page)
        check(st(page, "layers[1].glow") == 1.2, "style preset")
        for _ in range(9):
            page.click("button[title^='Undo (']")
        idle(page)
        check(st(page, "layers.length") <= 1, f"undo walks back the layer edits ({st(page, 'layers.length')} left)")

        print("image upload")
        tmp = Path(tempfile.mkdtemp()) / "logo.png"
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        im.paste((255, 255, 255, 255), (16, 16, 48, 48))
        im.save(tmp)
        page.locator(".add-row input[type=file]").set_input_files(str(tmp))
        idle(page)
        img_path = js(page, "fg.S.state.layers.at(-1)?.image")
        check(js(page, "fg.S.state.layers.at(-1)?.type") == "image" and str(img_path).startswith("uploads/"),
              f"upload creates an image layer ({img_path})")
        if img_path:
            (ROOT / "custom" / img_path).unlink(missing_ok=True)

        print("portrait text override survives edits")
        js(page, "() => fg.loadRecipe('dragon_lettering')")
        idle(page)
        page.select_option(".screen-pick select", "phone")
        idle(page)
        check(st(page, "layers[0].text") == "FRAC\nTAL", f"portrait text ({st(page, 'layers[0].text')!r})")
        js(page, "() => fg.setLayer(fg.S.state.layers[0].id, 'glow', 0.2)")
        idle(page)
        page.select_option(".screen-pick select", "desktop")
        idle(page)
        check(st(page, "layers[0].text") == "FRACTAL" and st(page, "layers[0].glow") == 0.2,
              f"landscape keeps its text and the edit ({st(page, 'layers[0]')})")

        print("effects")
        js(page, "() => fg.loadRecipe('julia_dragon')")
        idle(page)
        tab(page, "Effects")
        page.locator("button:text('+ Add effect')").click()
        page.locator(".dd-item:has(span:text-is('Bloom'))").click()
        idle(page)
        check(st(page, "post[0].op") == "bloom", "add bloom")
        page.locator(".chip-btn:text-is('Film')").click()
        idle(page)
        check(js(page, "fg.S.state.post.map(e => e.op).join()") == "contrast,grain,vignette", "Film preset")
        page.locator(".card-head button[title='Later']").first.click()
        idle(page)
        check(js(page, "fg.S.state.post.map(e => e.op).join()") == "grain,contrast,vignette", "reorder effects")
        page.locator(".card-head .eye").first.click()
        idle(page)
        check(st(page, "post[0].enabled") is False, "turn an effect off")

        print("save / export")
        name = "_e2e_tmp"
        target = ROOT / "custom" / f"{name}.toml"
        target.unlink(missing_ok=True)
        js(page, "() => fg.setPath('params.max_iter', 321)")
        idle(page)
        page.locator(".topbar .btn.primary").click()
        page.locator("button:text('Save as recipe…')").click()
        page.locator(".modal input[type=text]").fill(name)
        page.locator(".modal .btn.primary:text-is('Save')").click()
        page.wait_for_timeout(1500)
        text = target.read_text() if target.exists() else ""
        check("max_iter = 321" in text and "grain" in text, "save writes the recipe with the edit and effects")
        check(js(page, f"fg.S.catalog.recipes.some(r => r.name === '{name}')"), "saved recipe appears in the catalog")
        target.unlink(missing_ok=True)
        page.keyboard.press("Escape")
        page.locator(".topbar .btn.primary").click()
        page.locator(".size-card:has(.size-label:text-is('Custom'))").click()
        dims = page.locator(".custom-dims input")
        dims.nth(0).fill("160")
        dims.nth(1).fill("90")
        page.locator(".form-row input[type=text]").fill("_e2e")
        page.locator(".segmented button:text-is('Wallpapers folder')").click()
        page.locator(".modal-foot .btn.primary").click()
        page.wait_for_function("['done', 'error'].includes(fg.S.exportJob?.status)", timeout=60000)
        job = js(page, "fg.S.exportJob")
        check(job["status"] == "done", f"export to folder: {job.get('error') or job['status']}")
        if job["status"] == "done":
            out = Path(job["path"])
            check(out.parent.name == "gui" and Image.open(out).size == (160, 90), "folder export is 160x90 in backgrounds/gui")
            check(page.locator("button:text('Set as desktop wallpaper')").count() == 0, "no set-as-wallpaper button")
            out.unlink(missing_ok=True)
        page.locator(".segmented button:text-is('Download')").click()
        with page.expect_download(timeout=60000) as dl:
            page.locator(".modal-foot .btn.primary").click()
        d = dl.value
        tmp_png = Path(tempfile.mkdtemp()) / d.suggested_filename
        d.save_as(tmp_png)
        check(d.suggested_filename == "_e2e_160x90.png" and Image.open(tmp_png).size == (160, 90),
              f"download export arrives in the browser ({d.suggested_filename})")
        job = js(page, "fg.S.exportJob")
        check(not str(job.get("path", "")).startswith(str(ROOT)), "download export is not kept in the project")
        page.keyboard.press("Escape")

        print("variations")
        page.locator(".tab:has(.tab-label:text-is('Look'))").click()
        page.locator("button:text-is('Variations')").click()
        page.wait_for_function("fg.S.variations.length && fg.S.variations[0].url", timeout=60000)
        p0 = js(page, "JSON.stringify(fg.S.state.params)")
        page.locator(".gallery.variations .card").first.click()
        idle(page)
        check(js(page, "fg.S.ui.overlay") is None, "picking a variation closes the grid")
        page.click("button[title^='Undo (']")
        idle(page)
        check(js(page, "JSON.stringify(fg.S.state.params)") == p0, "undo returns from a variation")

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
            js(page, "() => { void fg.loadRecipe('julia_dragon'); void fg.loadRecipe('mandelbrot_spiral'); }")
            idle(page)
            page.wait_for_timeout(1300)
            check(st(page, "recipe") == "mandelbrot_spiral", f"last recipe click wins (got {st(page, 'recipe')})")
        finally:
            page.unroute("**/api/state*")

        print("phone width")
        page.set_viewport_size({"width": 390, "height": 760})
        page.wait_for_timeout(300)
        over = js(page, "(() => { const t = document.querySelector('.topbar'); return t.scrollWidth - t.clientWidth; })()")
        check(over <= 1, f"phone topbar fits ({over}px over)")
        tab(page, "Layers")
        check(js(page, "document.querySelector('.sidebar').classList.contains('open')"), "tab opens the panel sheet")
        page.set_viewport_size({"width": 1500, "height": 900})
        page.wait_for_timeout(200)

        if ALL:
            print("every recipe renders in the GUI")
            js(page, "() => fg.setPreviewSize(320)")
            names = js(page, "fg.S.catalog.recipes.map(r => r.name)")
            bad = []
            for n in names:
                js(page, f"() => fg.loadRecipe({n!r})")
                try:
                    idle(page, timeout=180000)
                except Exception:
                    bad.append(f"{n}: timeout")
                    continue
                err = js(page, "fg.S.view.error?.message || ''")
                if err or st(page, "recipe") != n:
                    bad.append(f"{n}: {err[:200]}")
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
