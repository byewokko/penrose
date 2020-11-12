from p5 import *

from pentagrid import Pentagrid
from tile import TilingBuilder


draw_once = True
generator = None


def setup():
    size(1280, 1024)
    stroke_weight(3)
    # no_stroke()


def draw():
    if frame_count < 100:
        return

    global draw_once
    if draw_once:
        background("#fff591")
        draw_once = False

    push_matrix()
    translate(width / 2, height / 2)
    scale(40, 40)
    stroke(Color("#a3f7bf"))

    try:
        c_small = Color("#fa26a0")
        c_big = Color("#05dfd7")
        rhomb = next(generator)
        verts = rhomb.xy()
        begin_shape()
        if rhomb.type() in (1, 4):
            fill(c_small)
        else:
            fill(c_big)
        for x, y, z in verts:
            vertex(x, y)
        end_shape(CLOSE)
    except StopIteration:
        pass

    pop_matrix()


if __name__ == "__main__":
    steps = 100
    edge_length = 50
    draw_once = True

    grid = Pentagrid([0.27603443, 0.25264065, 0.3475783,  0.21077636, 0.25933642])
    tiling = TilingBuilder(grid)
    tiling.prepare_grid((-50, 50))
    generator = tiling.generate_rhombs()
    run()
