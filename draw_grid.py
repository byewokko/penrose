from p5 import *
import compose
import pentagrid_2

counter = 0
edge_length = 50
draw_once = True

walk = None


def setup():
    size(1024, 768)
    stroke_weight(3)


def draw():
    clear()
    background(0)

    g1, g2, g3 = pentagrid_2.test(frame_count/10 + 15)

    push_matrix()
    translate(width / 2, height / 2)
    scale(10, 10)

    RADIUS = 1
    for x, y, z in g1:
        fill(63, 63, 63, 255)
        circle(x, -y, RADIUS)
    for x, y, z in g2:
        fill(63, 63, 255, 255)
        circle(x, -y, RADIUS)
    for x, y, z in g3:
        fill(255, 63, 63, 255)
        circle(x, -y, RADIUS)

    pop_matrix()


if __name__ == "__main__":
    run()
