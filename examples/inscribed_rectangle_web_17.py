import multigrid
from ntiling import TilingBuilder
from drawing.pil_draw_simple import Draw

palette = [
    "#ffee7d",
    "#b767ff",
    "#44fadd",
    "#ffe0f7",
    "#fe91ca",
]
draw = Draw(scale=110, width=3 * 1280, height=3 * 1280, bg_color=palette[-2])
draw.line_color = None
index_range = (-3, 3)
grid = multigrid.Multigrid(17)
tiling_builder = TilingBuilder(grid)
tiling_builder.prepare_grid(index_range)
tiling_builder.generate_rhomb_list()

for rhomb in tiling_builder._rhombs.values():
    new = []
    for a, b in rhomb.get_edges():
        new.append((a.get_xy() + b.get_xy()) / 2)
    p1 = new[-1]
    for p2 in new:
        draw.edge(p2[:2], p1[:2], color=palette[-1], width=8)
        p1 = p2

draw.show()
