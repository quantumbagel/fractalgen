# custom/

Your own recipes, palettes, formulas and plugins. Everything here (`*.toml`,
`*.py`) is loaded automatically. `python -m fractalgen fork NAME` copies a
built-in recipe here with every parameter spelled out, ready to edit.

Scaffold new extensions without hand-copying boilerplate:

```sh
python -m fractalgen new generator my_waves --view    # -> custom/my_waves.py + .toml
python -m fractalgen new post my_grade               # -> custom/my_grade.py
python -m fractalgen new palette sunset --stops 000000,ff5500,ffd700,ffffff
```

Use `--dir /path/to/dir` to write elsewhere (tests or a separate package).
`python -m fractalgen doctor` prints versions and a tiny smoke render for bug reports.
