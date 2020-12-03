import multigrid
from drawing.pil_draw_simple import Draw

draw = Draw(scale=80)
grid = multigrid.Multigrid(7)
index_range = (-2, 2)
multigrid.plot_grid(grid, draw, *index_range)
intersections = grid.calculate_intersections(index_range)
# pentagrid.plot_intersections(grid, draw, *index_range)
edges = multigrid.intersections_to_edges_dict(intersections, index_range)
draw.show()
