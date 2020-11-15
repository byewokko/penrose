from p5 import *

from pentagrid import Pentagrid
from tile import TilingBuilder


draw_once = True
generator = None
scale_ = 40

colors = [
    Color("#fa26a0"),
    Color("#05dfd7"),
    Color("#fff591"),
    Color("#a3f7bf"),
    Color("#3b2e5a")
]


def setup():
    size(1280, 1024)
    stroke_weight(4)
    no_stroke()


def draw():
    if frame_count < 50:
        return

    global draw_once
    if draw_once:
        stroke(colors[2])
        background(colors[4])
        draw_once = False

    push_matrix()
    translate(width / 2, height / 2)
    scale(scale_, scale_)


    rhomb = next(generator)
    verts = rhomb.xy()
    begin_shape()
    # x = rhomb.node[2] % 5
    # x = int(rhomb.node[2] == rhomb.node[3]) + 2 * int(rhomb.node[2] == -rhomb.node[3])
    # x = int(2 in rhomb.node[:2])
    # x = rhomb.type()-1
    # a = np.asarray([list(v) for v in rhomb.get_vertices()])
    # x = int(np.sum(np.abs(a.sum(axis=0))) % 5)
    x = rhomb.type() in (1, 4)
    # x = rhomb.node[0]
    fill(colors[x])

    for x, y, z in verts:
        vertex(x, y)
    end_shape(CLOSE)

    # stroke(colors[rhomb.type() in (1, 4)])
    # a, b, c, d = verts
    # n_lines = 4
    # lines_a = np.linspace(a, b, n_lines)
    # lines_b = np.linspace(d, c, n_lines)
    # for a, b in zip(lines_a, lines_b):
    #     line(*a, *b)

    pop_matrix()


if __name__ == "__main__":
    steps = 100
    edge_length = 50
    draw_once = True

    # grid = Pentagrid([0.27603443, 0.25264065, 0.3475783,  0.21077636, 0.25933642])
    # grid = Pentagrid([0.33328288, 0.18981448, 0.02987785, 0.27362873, 0.29810506])
    grid = Pentagrid()
    tiling = TilingBuilder(grid)
    tiling.prepare_grid((-50, 50))
    generator = tiling.generate_rhombs()
    run()
