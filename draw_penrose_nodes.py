from p5 import *
import compose

counter = 0
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

    global counter
    counter += 1
    if counter >= len(graph.nodes) - 1:
        graph.expand_edge(graph.edge_frontier_pop())

    push_matrix()
    translate(width / 2, height / 2)

    RADIUS_BIG = 10
    RADIUS_SMALL = 6
    for node in graph.nodes:
        x, y = list(node.get_xy(edge_length))
        fill(255, 0, 0, 255)
        circle(x, y, RADIUS_BIG)
        for i in range(10):
            x2 = x + math.cos(2 * math.pi / 10 * (i + 0.5)) * edge_length / 3
            y2 = y - math.sin(2 * math.pi / 10 * (i + 0.5)) * edge_length / 3
            c2 = fill(0, 0, 255, 255) if node.is_free(i) else fill(255, 0, 0, 255)
            circle(x2, y2, RADIUS_SMALL)

    pop_matrix()


if __name__ == "__main__":
    graph = compose.RhombNet()
    a = graph.add_node(compose.RhombNode())
    b = graph.add_node(a.step(0))
    e = graph.add_edge(compose.RhombEdge(a, b, 1))
    graph.edge_frontier_add(e)
    graph.expand_edge(graph.edge_frontier_pop())
    run()
