import multigrid
import ntiling
from drawing.pil_draw_simple import Draw

draw = Draw(scale=140, width=3*1280, height=3*1280, bg_color="#fff591")
draw.line_color = None
index_range = (-7, 7)
grid = multigrid.Pentagrid()
tiling_builder = ntiling.TilingBuilder(grid)
tiling_builder.prepare_grid(index_range)
tiling_builder.generate_rhomb_list()

palette = [
    "#fa26a0",
    "#fff591",
    "#3b2e5a"
]

for rhomb in tiling_builder._rhombs.values():
    c = rhomb.type() in (2, 3)
    draw.polygon(rhomb.xy(form="xy1"), color=palette[c])
    for a, b in rhomb.get_edges():
        draw.edge(a.get_xy(form="xy"), b.get_xy(form="xy"), color=palette[-1], width=8)

draw.show()
