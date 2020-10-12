from p5 import *
from tiling import generate

vertices = []
edges = []


def setup():
    size(1024, 768)
    background(255)
    stroke_weight(3)


def draw():
    clear()
    push_matrix()
    translate(width / 2, height / 2)
    for a, b in edges[:(frame_count // 2)]:
        # print(*vertices[a], *vertices[b])
        line(*vertices[a], *vertices[b])
    circle(*vertices[(frame_count // 2)], 5)
    pop_matrix()


if __name__ == "__main__":
    vertices, edges = generate(100, edge_length=50)
    run()
