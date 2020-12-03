import numpy as np

import multigrid
import tile
from drawing.pil_draw_simple import Draw

palette = [
    "#222831",
    "#393e46",
    "#ffd369",
    "#eeeeee",
    # "#000000"
]

draw = Draw(scale=200, width=3*1280, height=3*1280, bg_color=palette[-2])
draw.line_color = None
index_range = (-6, 6)
grid = multigrid.Pentagrid()
tiling_builder = tile.TilingBuilder(grid)
tiling_builder.prepare_grid(index_range)
tiling_builder.generate_rhomb_list()

radius = (1 * np.sin(np.pi/5) / 2.1, 1 * np.sin(np.pi/5 * 2) / 2.1)

for rhomb in tiling_builder._rhombs.values():
    c = rhomb.type() in (2, 3)
    if not c:
        draw.polygon(rhomb.xy(), color=palette[-1])
    else:
        draw.polygon(rhomb.xy(), color=palette[c])
        for a, b in rhomb.get_edges():
            draw.edge(a.get_xy(homogenous=False), b.get_xy(homogenous=False), color=palette[-1], width=8)

for rhomb in tiling_builder._rhombs.values():
    c = rhomb.type() in (2, 3)
    if not c:
        a, _, b, _ = rhomb.get_vertices()
        center = (a.get_xy() + b.get_xy())/2
        draw.circle(*center[:2], radius[c], color=palette[c])

draw.show()
