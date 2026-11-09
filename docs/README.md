# docs/

Release assets for the project README.

| File | What |
|---|---|
| `assets/header.jpg` | The word *fractalgen* cut from `julia_dragon` by the mask engine |
| `assets/*.jpg` | Gallery thumbs (960px) of built-in looks and a few GUI sessions |
| `assets/looks/*.jpg` | One look per family (480×270) for the README's "What it draws" grid |
| `assets/gui.jpg` | Screenshot of the GUI (1440×900 window, Color tab) for the README |
| `render_assets.py` | Regenerates the header (and gallery thumbs if `backgrounds/` is present) |

```sh
python docs/render_assets.py            # header
python docs/render_assets.py --gallery  # header + thumbs
python docs/render_assets.py --looks    # only the per-family grid, rendered fresh
```
