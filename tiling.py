from __future__ import annotations

import random
import sys
from typing import Optional, Union, Tuple, List, Sequence, Deque
from collections import OrderedDict, deque
import numpy as np

INVERT_Y = True

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
        self.free_slots = [True for _ in range(10)]  # slots IN BETWEEN edges

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

    def take_slot(self, slot: Union[int, Sequence[int]]):
        if isinstance(slot, int):
            self.free_slots[slot % 10] = False
        else:
            for i in slot:
                self.free_slots[i % 10] = False

    def is_free(self, slot: Union[int, Sequence[int]]):
        if isinstance(slot, int):
            return self.free_slots[slot % 10]
        else:
            for i in slot:
                if self.free_slots[i % 10] is False:
                    return False
            return True

    def take_if_free(self, slot: Union[int, Sequence[int]]):
        # TODO: inefficient?
        if not self.is_free(slot):
            print("node is not free", file=sys.stderr)
            raise ValueError("node is not free")
            # self.take_slot(slot)
        else:
            self.take_slot(slot)

    def get_xy(self, edge_length: float):
        x = edge_length * (self._p[0] + np.sqrt(5) * self._p[1]) / 4
        y = edge_length * (np.sin(np.pi / 5) * self._p[2] + np.sin(np.pi * 2 / 5) * self._p[3])
        if INVERT_Y:
            y = -y
        return x, y
    
    
class PenroseRhombNet:
    def __init__(self):
        self.nodes: List[PenroseNode4D] = []
        self.edges: List[Tuple[PenroseNode4D, PenroseNode4D]] = []
        self.rhombs: List[Tuple[PenroseNode4D, PenroseNode4D, PenroseNode4D, PenroseNode4D]] = []
        self.frontier: Deque[int] = deque()
        self.debug = []
        self.stopflag = False

    def add_node(self, node: PenroseNode4D, add_to_frontier: bool = True):
        index = self.find_node(node)
        if index > -1:
            return index, self.nodes[index]
        self.nodes.append(node)
        index = len(self.nodes) - 1
        self.frontier.append(index)
        return index, node

    def find_node(self, node: PenroseNode4D) -> int:
        if node not in self.nodes:
            return -1
        return self.nodes.index(node)

    def get_node(self, index: int) -> PenroseNode4D:
        return self.nodes[index]

    def add_edge(self,
                 a: Union[int, PenroseNode4D],
                 b: Union[int, PenroseNode4D]):
        if isinstance(a, int):
            a = self.get_node(a)
        if isinstance(b, int):
            b = self.get_node(b)
        self.edges.append((a, b))

    def add_rhomb(self, index: int, direction: int, angle: int):
        assert angle in (1, 2, 3, 4)
        start = direction % 10
        iA = index
        A = self.nodes[iA]
        try:
            A.take_if_free(range(start, start + angle))
        except ValueError:
            self.stopflag = True
            A.take_slot(range(start, start + angle))

        B = A.step(start)
        iB, B = self.add_node(B)
        try:
            B.take_if_free(range(start + angle, start + 5))
        except ValueError:
            self.stopflag = True
            B.take_slot(range(start + angle, start + 5))

        C = B.step(start + angle)
        iC, C = self.add_node(C)
        try:
            C.take_if_free(range(start + 5, start + 5 + angle))
        except ValueError:
            self.stopflag = True
            C.take_slot(range(start + 5, start + 5 + angle))

        D = C.step(start + 5)
        iD, D = self.add_node(D)
        try:
            D.take_if_free(range(start + 5 + angle, start + 10))
        except ValueError:
            self.stopflag = True
            D.take_slot(range(start + 5 + angle, start + 10))

        self.add_edge(iA, iB)
        self.add_edge(iB, iC)
        self.add_edge(iC, iD)
        self.add_edge(iD, iA)

    def expand_node(self, index: Optional[int] = None):
        if index is None:
            index = self.frontier.popleft()
        A = self.nodes[index]
        print(f"Expanding node {index}: {A}")
        free_slots = A.free_slots
        print(f"free_slots {free_slots}")
        while True in free_slots:
            if False in free_slots:
                # rotate the list so that free_slots[-1] == False and free_slots[0] == True
                start = free_slots.index(False)
                while free_slots[start] is False:
                    start = (start + 1) % 10
                end = (start + 1) % 10
                while free_slots[end % 10] is True:
                    end = (end + 1) % 10
                stretch = (end - start) % 10
            else:
                start = random.randint(0, 9)
                end = start + 10
                stretch = 10
            print(f"Start direction: {start}, end: {end}")

            # split the stretch
            angles = []
            while stretch > 0:
                print(f"stretch = {stretch}")
                angle = random.randint(1, min(4, stretch))
                stretch -= angle
                angles.append(angle)
            print(f"Angles: {angles}")

            # build the rhombs
            iA = index
            for angle in angles:
                print(f"angle: {angle}")
                self.add_rhomb(iA, start, angle)
                if self.stopflag:
                    return
                start += angle
        print(f"Expanding node {index} completed")







    def get_node_xy(self, index: int, edge_length: float) -> Tuple[int, int]:
        return self.nodes[index].get_xy(edge_length)

    def generate_edges(self, edge_length: float):
        for n1, n2 in self.edges:
            yield *n1.get_xy(edge_length), *n2.get_xy(edge_length)

    def consume_edges(self, edge_length: float):
        for n1, n2 in self.edges:
            yield *n1.get_xy(edge_length), *n2.get_xy(edge_length)
        self.edges = []


def random_walk(graph: PenroseRhombNet):
    d = random.randint(0, 9)
    d_last = None
    last, last_node = graph.add_node(PenroseNode4D())
    while True:
        while d_last in (d, (d + 5) % 10):
            d = random.randint(0, 10)
        this, this_node = graph.add_node(graph.nodes[last].step(d))
        yield graph.get_node(last), graph.get_node(this)
        last = this
        d_last = d


def random_tiling(graph: PenroseRhombNet):
    last, last_node = graph.add_node(PenroseNode4D())
    while True:
        graph.expand_node()
        yield from graph.edges
        if graph.stopflag:
            break
        graph.edges = []

    # for e, m in zip(graph.edges, graph.debug):
    #     print(m)
    #     yield e
    # yield from graph.edges


def test_circle(graph: PenroseRhombNet):
    last, last_node = graph.add_node(PenroseNode4D())
    for d in range(10):
        this_node = last_node.step(d)
        print(d, this_node._p)
        yield last_node, this_node
        last_node = this_node


def test_star(graph: PenroseRhombNet):
    last, last_node = graph.add_node(PenroseNode4D())
    for d in range(10):
        this_node = last_node.step(d)
        print(d, this_node._p)
        yield last_node, this_node
