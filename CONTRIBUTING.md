# Contributing

## Setup

```sh
python -m venv .venv && .venv/bin/pip install -e .
python -m unittest discover tests
```

Run `python -m fractalgen doctor` to print environment details and perform a quick smoke render when troubleshooting or reporting bugs.

## Adding a look

To create a new look, start by forking an existing recipe:

```sh
python -m fractalgen fork julia_dragon my_dragon   # creates custom/my_dragon.toml
python -m fractalgen gui                           # tweak parameters and click Save as recipe
```

You can also write TOML files directly in `custom/` to define palettes, iteration formulas, maps, flows, or complete recipes. Refer to `examples/extensions.toml` and the Extending section in the README.

## Adding a generator or effect

```sh
python -m fractalgen new generator my_waves --view
python -m fractalgen new post my_grade
python -m fractalgen new palette sunset --stops 000000,ff5500,ffd700,ffffff
```

This scaffolds `custom/<name>.py` alongside a starter recipe for new generators. Function decorators such as `@fg.generator` and `@fg.post_op` combine with type hints (`Range`, `Doc`, `Literal`) to populate CLI arguments and GUI controls. See `examples/annotated_plugin.py` for a complete example and `examples/pip_plugin/` for packaging via pip entry points.

Custom generators should call `fg.auto_tone(...)` so recipe `tone` settings take effect properly, and call `fractalgen.progress.update(fraction)` during intensive loops to drive export progress indicators.

## Testing

```sh
python -m unittest discover tests        # runs unit tests and thumbnail smoke renders for all recipes
python tools/parity.py BASELINE_DIR      # compares outputs against legacy reference renders
python tools/gui_check.py --all          # runs headless Chromium checks (requires playwright)
```

Keep test renders fast by using thumbnail dimensions. The test suite verifies that recipes generate without errors, NaNs, or empty canvases. Prefer targeted unit tests over slow high-resolution renders.

## Code style

Keep code style consistent with existing modules by relying on the standard library and numpy rather than extra dependencies. Use comments sparingly, focusing on mathematical constraints or performance notes. Recipe TOML files define public looks, while Python handles underlying algorithms.
