from p5 import *

from pentagrid import Pentagrid

scaling = 50
edges = []


def setup():
    size(1024, 768)
    background(255)
    stroke_weight(1)


def draw():
    global edges
    push_matrix()
    translate(width / 2, height / 2)
    scale(30, 30, 1)

    for e in edges:
        line(*e)
    edges = []

    pop_matrix()


if __name__ == "__main__":
    grid = Pentagrid()
    for f in Pentagrid.LINE_FAMILIES:
        for i in range(-10, 11):
            edges.append((*grid.get_node_given_p(-15, i, f), *grid.get_node_given_p(15, i, f)))
    run()
