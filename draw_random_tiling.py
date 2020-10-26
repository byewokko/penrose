from p5 import *
import tiling

steps = 100
edge_length = 50

walk = None


def setup():
    size(1024, 768)
    background(255)
    stroke_weight(2)


def draw():
    # global img
    if frame_count % 2:
        return

    # if img:
    #     image(img, 0, 0)
    push_matrix()
    translate(width / 2, height / 2)

    try:
        n1, n2, t = next(walk)
        line(*n1.get_xy(30), *n2.get_xy(30))
    except StopIteration:
        pass
    # clear()
    # img = get()
    # circle(*n2, 5)
    pop_matrix()


if __name__ == "__main__":
    graph = tiling.RhombNet()
    walk = tiling.random_tiling(graph)
    run()
