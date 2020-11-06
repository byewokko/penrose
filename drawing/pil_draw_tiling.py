import math

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 800, 600
EDGE_LENGTH = 50

img = Image.new("RGB", (WIDTH, HEIGHT))
draw = ImageDraw.Draw(img)


def draw_edge(edge, color="white"):
    x1, y1, x2, y2 = list(edge.get_xy_coordinates(EDGE_LENGTH))
    x1 += WIDTH / 2
    y1 += HEIGHT / 2
    x2 += WIDTH / 2
    y2 += HEIGHT / 2
    draw.line([x1, y1, x2, y2], width=3, fill=color)


def draw_edge_v(edge):
    OFFSET = 0.5
    x1, y1, x2, y2 = list(edge.get_xy_coordinates(EDGE_LENGTH))
    x1 += WIDTH/2
    y1 += HEIGHT/2
    x2 += WIDTH/2
    y2 += HEIGHT/2
    off_x = math.sin(2*math.pi/10 * edge.direction()) * OFFSET
    off_y = -math.cos(2*math.pi/10 * edge.direction()) * OFFSET
    cl = "blue" if edge.is_free("left") else "red"
    cr = "blue" if edge.is_free("right") else "red"
    draw.line([x1 - off_x, y1 + off_y, x2 - off_x, y2 + off_y], width=2, fill=cl)
    draw.line([x1 + off_x, y1 - off_y, x2 + off_x, y2 - off_y], width=2, fill=cr)


def draw_edge_arrow(edge):
    x1, y1, x2, y2 = list(edge.get_xy_coordinates(EDGE_LENGTH))
    x1 += WIDTH/2
    y1 += HEIGHT/2
    x2 += WIDTH/2
    y2 += HEIGHT/2
    d = edge.direction()
    if edge.value() in (-1, 1):
        mid = (x1 + x2) / 2, (y1 + y2) / 2
        sgn = edge.value()
        tip = (sgn * math.cos(2 * math.pi/10 * d) * EDGE_LENGTH / 6 + mid[0],
               -sgn * math.sin(2 * math.pi/10 * d) * EDGE_LENGTH / 6 + mid[1])
        left = (sgn * math.cos(2 * math.pi/10 * (d + 4)) * EDGE_LENGTH / 6 + mid[0],
                -sgn * math.sin(2 * math.pi/10 * (d + 4)) * EDGE_LENGTH / 6 + mid[1])
        right = (sgn * math.cos(2 * math.pi/10 * (d - 4)) * EDGE_LENGTH / 6 + mid[0],
                 -sgn * math.sin(2 * math.pi/10 * (d - 4)) * EDGE_LENGTH / 6 + mid[1])
        draw.polygon([tip, left, right], fill="white")
    elif edge.value() in (-2, 2):
        mid = (x1 + x2) / 2, (y1 + y2) / 2
        sgn = edge.value() / 2
        mid = (mid[0] + math.cos(2 * math.pi / 10 * d) * EDGE_LENGTH / 8,
               mid[1] - math.sin(2 * math.pi / 10 * d) * EDGE_LENGTH / 8)
        tip = (sgn * math.cos(2 * math.pi/10 * d) * EDGE_LENGTH / 8 + mid[0],
               -sgn * math.sin(2 * math.pi/10 * d) * EDGE_LENGTH / 8 + mid[1])
        left = (sgn * math.cos(2 * math.pi/10 * (d + 4)) * EDGE_LENGTH / 8 + mid[0],
                -sgn * math.sin(2 * math.pi/10 * (d + 4)) * EDGE_LENGTH / 8 + mid[1])
        right = (sgn * math.cos(2 * math.pi/10 * (d - 4)) * EDGE_LENGTH / 8 + mid[0],
                 -sgn * math.sin(2 * math.pi/10 * (d - 4)) * EDGE_LENGTH / 8 + mid[1])
        draw.polygon([tip, left, right], fill="white")

        mid = (mid[0] - math.cos(2 * math.pi / 10 * d) * EDGE_LENGTH / 4,
               mid[1] + math.sin(2 * math.pi / 10 * d) * EDGE_LENGTH / 4)
        tip = (sgn * math.cos(2 * math.pi/10 * d) * EDGE_LENGTH / 8 + mid[0],
               -sgn * math.sin(2 * math.pi/10 * d) * EDGE_LENGTH / 8 + mid[1])
        left = (sgn * math.cos(2 * math.pi/10 * (d + 4)) * EDGE_LENGTH / 8 + mid[0],
                -sgn * math.sin(2 * math.pi/10 * (d + 4)) * EDGE_LENGTH / 8 + mid[1])
        right = (sgn * math.cos(2 * math.pi/10 * (d - 4)) * EDGE_LENGTH / 8 + mid[0],
                 -sgn * math.sin(2 * math.pi/10 * (d - 4)) * EDGE_LENGTH / 8 + mid[1])
        draw.polygon([tip, left, right], fill="white")


def draw_edges(graph):
    if isinstance(graph, list):
        for edge in graph:
            draw_edge_v(edge)
    else:
        for edge in graph.edges:
            draw_edge_v(edge)


def draw_node(node, color="red"):
    RADIUS_BIG = 3
    RADIUS_SMALL = 1
    x, y = list(node.get_xy(EDGE_LENGTH))
    x += WIDTH / 2
    y += HEIGHT / 2
    draw.ellipse([x - RADIUS_BIG, y - RADIUS_BIG, x + RADIUS_BIG, y + RADIUS_BIG], fill=color)
    for i in range(10):
        x2 = x + math.cos(2*math.pi/10 * (i + 0.5)) * EDGE_LENGTH / 3
        y2 = y - math.sin(2*math.pi/10 * (i + 0.5)) * EDGE_LENGTH / 3
        c2 = "blue" if node.is_free(i) else "red"
        draw.ellipse([x2 - RADIUS_SMALL, y2 - RADIUS_SMALL, x2 + RADIUS_SMALL, y2 + RADIUS_SMALL], fill=c2)


def draw_nodes(graph):
    if isinstance(graph, list):
        for node in graph:
            draw_node(node)
    else:
        for node in graph.nodes:
            draw_node(node)


def show():
    img.show()


def clear():
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill="black")
