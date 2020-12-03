import colorsys
import numpy as np

import multigrid
import tile
from drawing.pil_draw_simple import Draw

draw = Draw(scale=90, width=3*1280, height=3*1280, bg_color=None, color_mode="RGBA")
draw.line_color = None
index_range = (-6, 6)
grid = multigrid.Pentagrid()
tiling_builder = tile.TilingBuilder(grid)
tiling_builder.prepare_grid(index_range)
tiling_builder.generate_rhomb_list()

# assign a color to each ribbon
base_colors = [colorsys.hsv_to_rgb(1/15*i, 1, 1) for i in range(15, 20)]

# compute colors for each ribbon intersection type
palette = {}
for i, j in multigrid.triangle_iterator(5):
    c = [("0" + hex(int(a))[2:])[-2:] for a in (((np.sqrt(base_colors[i]) + np.sqrt(base_colors[j]))/2) ** 2) * 256]
    palette[(i, j)] = "#" + "".join(c)

for rhomb in tiling_builder._rhombs.values():
    c = tuple(rhomb.node[:2])
    draw.polygon(rhomb.xy(), color=palette[c], outline=None)
    # if rhomb.type() in (1, 4):
    #     # paint all thin rhombs white
    #     draw.polygon(rhomb.xy(), color="#fff", outline=None)
    # else:
    #     draw.polygon(rhomb.xy(), color=palette[c], outline=None)

draw.show()
