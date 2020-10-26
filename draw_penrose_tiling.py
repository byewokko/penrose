from p5 import *
import tiling_new

steps = 100
edge_length = 50
draw_once = True

walk = None


def setup():
    size(1024, 768)
    stroke_weight(3)


def draw():
    if frame_count % 2:
        return

    global draw_once
    if draw_once:
        background(0)
        draw_once = False

    push_matrix()
    translate(width / 2, height / 2)

    try:
        edge = next(walk)
        if edge.value() in (2, -2):
            stroke(255, 127, 191, 255)
        else:
            stroke(191, 255, 63, 255)
        line(*edge.get_xy_coordinates(100))
    except StopIteration:
        pass

    pop_matrix()


if __name__ == "__main__":
    graph = tiling_new.RhombNet()
    walk = tiling_new.random_tiling(graph)
    run()
