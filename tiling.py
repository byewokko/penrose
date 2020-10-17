from __future__ import annotations

import random
from typing import Optional, Union, Tuple, List, Sequence
from collections import OrderedDict
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


class Node4D:
    def __init__(self, params: Optional[Sequence[int]] = None):
        if not params:
            self._p = (0, 0, 0, 0)
        else:
            assert len(params) == 4
            self._p = tuple(params)

    def as_tuple(self):
        return tuple(self._p)

    def __repr__(self):
        return f"Node4D{str(self._p)}"

    def __hash__(self):
        return hash(self.as_tuple())

    def __neg__(self):
        return type(self)([-x for x in self._p])

    def __add__(self, other: Node4D):
        return type(self)([self._p[i] + other._p[i] for i in range(4)])

    def __sub__(self, other: Node4D):
        return self + (-other)

    def __rmul__(self, other: int):
        return type(self)([other * x for x in self._p])

    def __mul__(self, other: int):
        return type(self)([other * x for x in self._p])

    def __eq__(self, other: Node4D):
        return self._p == other._p

    def __ne__(self, other: Node4D):
        return not (self._p == other._p)


class PenroseNode4D(Node4D):
    _step = tuple(Node4D(x) for x in [
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
    
    def __init__(self,
                 params: Optional[Sequence[int, int, int, int]] = None,
                 flag: Union[-1, 0, 1, None] = None):
        super().__init__(params)
        if not flag:
            self.flag = 0
        else:
            self.flag = flag
        self.free_edges = [True for _ in range(10)]

    def step(self, direction):
        direction = direction % 10
        return self + self._step[direction]

    def alt_coords(self):
        return (
            (self._p[0] + self._p[1] + 2 * self._p[2]) // 4,
            (self._p[1] + self._p[2] + self._p[3]) // 2,
            self._p[2],
            self._p[3]
        )

    def block_edge(self, direction):
        self.free_edges[direction % 10] = False

    def get_xy(self, edge_length: float):
        x = edge_length * (self._p[0] + np.sqrt(5) * self._p[1]) / 4
        y = edge_length * (np.sin(np.pi / 5) * self._p[2] + np.sin(np.pi * 2 / 5) * self._p[3])
        return x, y
    
    
class PenroseRhombNet:
    def __init__(self):
        self.nodes: List[PenroseNode4D] = []
        self.edges: List[Tuple[int, int]] = []
        self.rhombs: List[Tuple[int, int, int, int]] = []
        self.frontier: List[int] = []

    def add_node(self, node: PenroseNode4D, add_to_frontier: bool = True):
        index = self.find_node(node)
        if index > -1:
            return index
        self.nodes.append(node)
        index = len(self.nodes) - 1
        self.frontier.append(index)
        return index

    def find_node(self, node: PenroseNode4D) -> int:
        if node not in self.nodes:
            return -1
        return self.nodes.index(node)

    def get_node(self, index: int) -> PenroseNode4D:
        return self.nodes[index]

    def add_edge(self, i1: int, i2: int):
        self.edges.append((i1, i2))

    def expand_node(self, index: int):
        node = self.nodes[index]
        adjacent = [node.step(d) for d in range(10)]
        raise NotImplementedError

    def get_node_xy(self, index: int, edge_length: float) -> Tuple[int, int]:
        return self.nodes[index].get_xy(edge_length)

    def generate_edges(self, edge_length: float):
        for n1, n2 in self.edges:
            yield *self.get_node_xy(n1, edge_length), *self.get_node_xy(n2, edge_length)


def random_walk(graph: PenroseRhombNet):
    d = random.randint(0, 10)
    d_last = None
    last = graph.add_node(PenroseNode4D())
    while True:
        while d_last in (d, (d + 5) % 10):
            d = random.randint(0, 10)
        this = graph.add_node(graph.nodes[last].step(d))
        yield graph.get_node(last), graph.get_node(this)
        last = this
        d_last = d
