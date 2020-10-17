from p5 import *
import tiling

steps = 100
edge_length = 50

walk = None


def setup():
    size(1024, 768)
    background(255)
    stroke_weight(3)


def draw():
    # global img
    if frame_count % 5:
        return

    # if img:
    #     image(img, 0, 0)
    push_matrix()
    translate(width / 2, height / 2)

    try:
        n1, n2 = [x.get_xy(50) for x in next(walk)]
    except StopIteration:
        pass
    clear()
    line(*n1, *n2)
    # img = get()
    circle(*n2, 5)
    pop_matrix()


if __name__ == "__main__":
    graph = tiling.PenroseRhombNet()
    walk = tiling.random_walk(graph)
    run()
