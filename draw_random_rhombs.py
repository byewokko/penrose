from p5 import *
import tiling

steps = 100
edge_length = 50

walk = None


def setup():
    size(1024, 768)
    background(255)
    stroke_weight(1)
    # no_stroke()


def draw():
    # global img
    if frame_count % 2:
        return

    push_matrix()
    translate(width / 2, height / 2)

    try:
        a, b, c, d, rhomb_type = next(walk)
        a, b, c, d = [x.get_xy(40) for x in (a, b, c, d)]
        if rhomb_type:
            fill(255, 130, 0, 255)
        else:
            fill(50, 50, 255, 255)
        begin_shape()
        vertex(*a)
        vertex(*b)
        vertex(*c)
        vertex(*d)
        end_shape(CLOSE)
    except StopIteration:
        pass
    pop_matrix()


if __name__ == "__main__":
    graph = tiling.RhombNet()
    walk = tiling.random_tiling(graph, "rhombs")
    run()
