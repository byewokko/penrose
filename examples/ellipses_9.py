import multigrid
from ntiling import TilingBuilder
from drawing.pil_draw_simple import Draw

palette = [
    "#7fb414",
    "#fac70b",
    "#127681",
    "#df0e62",
    "#21174a",
]
draw = Draw(scale=150, width=3*1280, height=3*1280, bg_color=palette[-1])
draw.line_color = None
index_range = (-2, 2)
grid = multigrid.Multigrid(9)
tiling_builder = TilingBuilder(grid)
tiling_builder.prepare_grid(index_range)
tiling_builder.generate_rhomb_list()

for rhomb in tiling_builder._rhombs.values():
    col = (rhomb.type() in (4, 5, 3, 6, 2, 7)) + (rhomb.type() in (3, 6, 2, 7)) + (rhomb.type() in (2, 7))
    if col != 0:
        new = []
        for a, b in rhomb.get_edges():
            new.append((a.get_xy(form="xy1") + b.get_xy(form="xy1")) / 2)
        # coords = rhomb.center(form="xy")
        draw.polygon(new, color=palette[col])
    else:
        draw.polygon(rhomb.xy(form="xy1"), color=palette[col])

draw.show()
