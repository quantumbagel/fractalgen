# custom/

Place custom recipes, palettes, formulas, and plugins in this directory. Any `*.toml` and `*.py` files placed here load automatically when fractalgen starts.

To modify a built-in look, run `python -m fractalgen fork NAME`. This writes an editable recipe file here with every default parameter explicitly populated.

You can scaffold starter files from the command line:

```sh
python -m fractalgen new generator my_waves --view    # creates custom/my_waves.py and custom/my_waves.toml
python -m fractalgen new post my_grade               # creates custom/my_grade.py
python -m fractalgen new palette sunset --stops 000000,ff5500,ffd700,ffffff
```

Pass `--dir /path/to/dir` to write scaffolded files to an alternate folder. Run `python -m fractalgen doctor` to verify environment dependencies and active search paths.
