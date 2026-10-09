"""Extension API: entry points, reload, LoadError, registry hints, annotations."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Annotated, Literal
from unittest import mock

import numpy as np

import fractalgen as fg

_ep_test_calls: list[int] = []


def _ep_test_hook() -> None:
    _ep_test_calls.append(1)
from fractalgen.loader import LoadError, loaded_sources, reload_user_paths
from fractalgen.registry import Registry


class RegistryExtendTests(unittest.TestCase):
    def test_suggestions_and_source(self):
        r: Registry[str] = Registry("palette")
        r.register("ember", "a", source="/p.toml")
        r.register("ember_r", "b", source="/p.toml")
        self.assertEqual(r.source_of("ember"), "/p.toml")
        with self.assertRaises(KeyError) as ctx:
            r.get("embr")
        msg = str(ctx.exception)
        self.assertIn("Did you mean", msg)
        self.assertIn("known", msg)
        r.unregister("ember")
        self.assertNotIn("ember", r)
        dropped = r.drop_from("/p.toml")
        self.assertIn("ember_r", dropped)

    def test_replace_keeps_source(self):
        r: Registry[int] = Registry("thing")
        r.register("x", 1, source="a")
        r.register("x", 2, replace=True, source="b")
        self.assertEqual(r.get("x"), 2)
        self.assertEqual(r.source_of("x"), "b")


class LoadErrorTests(unittest.TestCase):
    def test_missing_key_loaderror(self):
        from fractalgen.loader import load_toml

        src = textwrap.dedent(
            """
            [formula.bad]
            degree = 3
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(src)
            path = Path(f.name)
        try:
            with self.assertRaises(LoadError) as ctx:
                load_toml(path)
            self.assertIn("missing required key", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, KeyError)
        finally:
            path.unlink(missing_ok=True)

    def test_per_entry_replace(self):
        src = textwrap.dedent(
            """
            [palette.cosmic]
            replace = true
            stops = ["000000", "ffffff"]
            """
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.toml"
            p.write_text(src)
            fg.bootstrap([p])
            pal = fg.get_palette("cosmic")
            self.assertEqual(len(pal.stops), 2)


class EntryPointTests(unittest.TestCase):
    def test_entry_point_load_and_failure(self):
        class EP:
            def __init__(self, name, target):
                self.name = name
                self.value = target

            def load(self):
                mod, attr = self.value.split(":")
                import importlib

                return getattr(importlib.import_module(mod), attr)

        _ep_test_calls.clear()
        fake = [EP("ok", f"{__name__}:_ep_test_hook")]

        def fake_eps(group=None):
            self.assertEqual(group, "fractalgen.plugins")
            return fake

        with mock.patch("importlib.metadata.entry_points", fake_eps):
            from fractalgen.loader import _load_entry_points

            _load_entry_points()
        self.assertEqual(_ep_test_calls, [1])
        self.assertTrue(any(s.startswith("entry-point:ok") for s in loaded_sources()))


class ReloadTests(unittest.TestCase):
    def test_reload_user_paths_picks_up_edit(self):
        with tempfile.TemporaryDirectory() as d:
            plug = Path(d) / "tweak.py"
            plug.write_text(
                textwrap.dedent(
                    """
                    import fractalgen as fg
                    LEVEL = 0.0
                    @fg.generator("reload_me", category="test", replace=True)
                    def reload_me(ctx):
                        '''reload test'''
                        import numpy as np
                        h, w = ctx.height, ctx.width
                        t = np.full((h, w), LEVEL)
                        return fg.sample_palette("ember", t)
                    """
                )
            )
            fg.bootstrap([plug])
            spec = fg.GENERATORS.get("reload_me")
            rgb1 = spec(fg.RenderContext(8, 8))
            plug.write_text(plug.read_text().replace("LEVEL = 0.0", "LEVEL = 1.0"))
            summary = reload_user_paths()
            self.assertIn(str(plug.resolve()), summary["reloaded"])
            spec = fg.GENERATORS.get("reload_me")
            rgb2 = spec(fg.RenderContext(8, 8))
            self.assertGreater(float(rgb2.mean()), float(rgb1.mean()))


class AnnotationHintTests(unittest.TestCase):
    def test_literal_and_range_from_hints(self):
        from fractalgen.generators.base import _hints_from_annotations

        def hint_fn(ctx, *, mode: Literal["a", "b"] = "a", x: Annotated[float, fg.Range(0, 1)] = 0.5):
            """hints"""
            import numpy as np

            h, w = ctx.height, ctx.width
            return np.zeros((h, w, 3))

        rng, chs, docs = _hints_from_annotations(hint_fn)
        self.assertEqual(chs.get("mode"), ["a", "b"])
        self.assertEqual(rng.get("x"), (0.0, 1.0))

        @fg.generator("hint_demo_ext", category="test", replace=True)
        def hint_demo_ext(ctx, *, mode: Literal["a", "b"] = "a", x: Annotated[float, fg.Range(0, 1)] = 0.5):
            """hints"""
            return hint_fn(ctx, mode=mode, x=x)

        spec = fg.GENERATORS.get("hint_demo_ext")
        self.assertEqual(spec.params["mode"].choice_list(), ["a", "b"])
        self.assertEqual(spec.params["x"].range, (0.0, 1.0))
        fg.GENERATORS.unregister("hint_demo_ext")


if __name__ == "__main__":
    unittest.main()
