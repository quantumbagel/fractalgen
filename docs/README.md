# docs/

Assets used by the project documentation and README.

| File                 | Description                                                          |
| -------------------- | -------------------------------------------------------------------- |
| `assets/header.jpg`  | Text mask rendering of the project name using `julia_dragon`         |
| `assets/*.jpg`       | Gallery previews (960px) of built-in looks and GUI sessions          |
| `assets/looks/*.jpg` | Reference thumbnails (480x270) representing each mathematical family |
| `assets/gui.jpg`     | Interface capture for the README                                     |
| `render_assets.py`   | Regeneration script for project assets                               |

```sh
python docs/render_assets.py            # regenerates the header image
python docs/render_assets.py --gallery  # regenerates the header and gallery thumbnails
python docs/render_assets.py --looks    # re-renders only the mathematical family thumbnails
```
