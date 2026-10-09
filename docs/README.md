# docs/

Release assets for the project README.

| File | What |
|---|---|
| `assets/header.jpg` | The word *fractalgen* cut from `julia_dragon` by the mask engine |
| `assets/*.jpg` | Gallery thumbs (960px) of built-in looks and a few GUI sessions |
| `render_assets.py` | Regenerates the header (and gallery thumbs if `backgrounds/` is present) |

```sh
python docs/render_assets.py            # header
python docs/render_assets.py --gallery  # header + thumbs
```
