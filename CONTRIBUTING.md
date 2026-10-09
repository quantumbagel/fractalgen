# Contributing

## Setup

```sh
python -m venv .venv && .venv/bin/pip install -e .
python -m unittest discover tests
```

`python -m fractalgen doctor` prints versions and a tiny smoke render, useful
in bug reports.

## Add a look

The fastest path is a fork of something close:

```sh
python -m fractalgen fork julia_dragon my_dragon   # -> custom/my_dragon.toml
python -m fractalgen gui                           # tweak, then Save recipe
```

Or write TOML by hand in `custom/` (palettes, formulas, maps, flows, recipes).
See `examples/extensions.toml` and the Extending section of the README.

## Add a generator or post op

```sh
python -m fractalgen new generator my_waves --view
python -m fractalgen new post my_grade
python -m fractalgen new palette sunset --stops 000000,ff5500,ffd700,ffffff
```

That scaffolds `custom/<name>.py` (and a starter recipe for generators).
Decorators `@fg.generator` / `@fg.post_op` plus type hints (`Range`, `Doc`,
`Literal`) feed the CLI and the GUI. `examples/annotated_plugin.py` is a
worked example; `examples/pip_plugin/` shows a pip entry point.

A new generator should call `fg.auto_tone(...)` so recipe `tone` adjustments
apply, and `fractalgen.progress.update(fraction)` inside long loops so export
progress bars move.

## Tests

```sh
python -m unittest discover tests        # unit tests + thumbnail smoke of every recipe
python tools/parity.py BASELINE_DIR      # compare against old-code renders
python tools/gui_check.py --all          # headless Chromium (needs playwright)
```

Keep new recipes rendering at thumbnail size (the smoke test will catch a
crash, NaN, or a blank frame). Prefer a focused unit test over a huge render.

## Style

Match the surrounding code: stdlib + numpy, no extra framework, short comments
only for non-obvious constraints. Recipe TOML is the public look format;
Python is for algorithms.
