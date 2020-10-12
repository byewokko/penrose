import numpy as np

step = np.asarray([
    (4, 0, 0, 0),
    (1, 1, 1, 0),
    (-1, 1, 0, 1),
    (1, -1, 0, 1),
    (-1, -1, 1, 0),
    (-4, 0, 0, 0),
    (-1, -1, -1, 0),
    (1, -1, 0, -1),
    (-1, 1, 0, -1),
    (1, 1, -1, 0),
])


def coordinates(p1, p2, p3, p4, edge_length):
    x = edge_length * (p1 + np.sqrt(5) * p2)/4
    y = edge_length * (np.sin(np.pi / 5) * p3 + np.sin(np.pi * 2 / 5) * p4)
    return x, y


def random_walk(length: int, start: tuple = (0, 0, 0, 0)):
    node = np.asarray(start)
    yield node
    for _ in range(length):
        ind = np.random.choice(len(step))
        node += step[ind]
        yield node


def random_walk_2(length: int, start: tuple = (0, 0, 0, 0)):
    node = np.asarray(start)
    ind = 0
    yield node
    for _ in range(length):
        ind = (ind + np.random.choice([-1, 1])) % len(step)
        node += step[ind]
        yield node


def loop_walk(start: tuple = (0, 0, 0, 0)):
    node = np.asarray(start)
    yield node
    for i in range(len(step)):
        ind = i % len(step)
        node += step[ind]
        yield node


def all_directions(start: tuple = (0, 0, 0, 0)):
    for i in range(len(step)):
        node = np.asarray(start)
        yield node
        node += step[i]
        yield node


def row_walk(start: tuple = (0, 0, 0, 0)):
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for l in range(-4, 5):
                    yield i, j, k, l


def generate(n, edge_length: float):
    vertices = []
    edges = []
    for node in row_walk():
        vertices.append(coordinates(*node, edge_length))
    for i in range(1, len(vertices)):
        edges.append((i - 1, i))
    return vertices, edges
