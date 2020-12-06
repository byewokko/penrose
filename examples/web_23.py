import multigrid
from ntiling import TilingBuilder
from drawing.pil_draw_simple import Draw

palette = [
    "#293462",
    "#216583",
    "#00818a",
    "#f7be16",
]
draw = Draw(scale=90, width=3*1280, height=3*1280, bg_color=palette[-2])
draw.line_color = None
index_range = (-1, 1)
grid = multigrid.Multigrid(23)
tiling_builder = TilingBuilder(grid)
tiling_builder.prepare_grid(index_range)
tiling_builder.generate_rhomb_list()

for rhomb in tiling_builder._rhombs.values():
    for a, b in rhomb.get_edges():
        draw.edge(a.get_xy(homogenous=False), b.get_xy(homogenous=False), color=palette[-1], width=4)

draw.show()
