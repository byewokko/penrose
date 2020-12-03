import multigrid
import tile
from drawing.pil_draw_simple import Draw

draw = Draw(scale=140, width=3*1280, height=3*1280, bg_color="#fff591")
draw.line_color = None
index_range = (-7, 7)
grid = multigrid.Pentagrid()
tiling_builder = tile.TilingBuilder(grid)
tiling_builder.prepare_grid(index_range)
tiling_builder.generate_rhomb_list()

palette = [
    "#fa26a0",
    "#05dfd7",
    "#fff591",
    "#a3f7bf",
    "#3b2e5a"
]

for rhomb in tiling_builder._rhombs.values():
    c = rhomb.node[0]
    draw.polygon(rhomb.xy(), color=palette[c])
    for a, b in rhomb.get_edges():
        draw.edge(a.get_xy(homogenous=False), b.get_xy(homogenous=False), color="#3b2e5a", width=8)

draw.show()
