import pentagrid
import tile
from drawing.pil_draw_simple import Draw

palette = [
    "#fa26a0",
    "#05dfd7",
    "#fff591",
    "#a3f7bf",
    "#3b2e5a"
]

draw = Draw(scale=90, width=3*1280, height=3*1280, bg_color=palette[2])
draw.line_color = None
index_range = (-6, 6)
grid = pentagrid.Pentagrid()
tiling_builder = tile.TilingBuilder(grid)
tiling_builder.prepare_grid(index_range)
tiling_builder.generate_rhomb_list()

for rhomb in tiling_builder._rhombs.values():
    c = rhomb.type() in (2, 3)
    # draw.polygon(rhomb.xy(), color=color[c+1], outline=color[0])
    new = []
    for a, b in rhomb.get_edges():
        new.append((a.get_xy() + b.get_xy())/2)
    p1 = new[-1]
    for p2 in new:
        draw.edge(p2[:2], p1[:2], color=palette[-1], width=7)
        p1 = p2
    # draw.polygon(new, color=color[c])

draw.show()
