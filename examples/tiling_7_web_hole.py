import multigrid
from ntiling import TilingBuilder
from drawing.pil_draw_simple import Draw

palette = [
    "#ffee7d",
    "#b767ff",
    "#44fadd",
    "#fe91ca",
    "#ffe0f7",
]
palette = [
    "#7fb414",
    "#df0e62",
    "#127681",
    "#fac70b",
    "#092a35",
]
draw = Draw(scale=90, width=4*1280, height=4*1280, bg_color=palette[-1])
draw.line_color = None
index_range = (-4, 5)
# offsets = list(map(float, "0.04885131 0.38705046 0.15540683 0.37524718 0.09360688 0.04554864 0.0424169".split(" ")))
offsets = None
grid = multigrid.Multigrid(7, offsets)
# grid = multigrid.Multigrid(7, [0.11071195, 0.40178219, 0.38167641, 0.05840904, 0.26593674, 0.30876262, 0.40169052])
tiling_builder = TilingBuilder(grid)
tiling_builder.prepare_grid(index_range)
tiling_builder.generate_rhomb_list()

for rhomb in tiling_builder._rhombs.values():
    c = (rhomb.type() in (1,6,2,5)) + (rhomb.type() in (1,6))
    if c != 2 and abs(rhomb.node[2]) < 3 and abs(rhomb.node[3]) < 3:
        continue
    # c = (rhomb.type() in (1, 2, 5, 6)) + (rhomb.type() in (1, 6))
    draw.polygon(rhomb.xy(form="xy1"), color=palette[-1])
    for a, b in rhomb.get_edges():
        draw.edge(a.get_xy(form="xy"), b.get_xy(form="xy"), color=palette[-3], width=4)

draw.show()
