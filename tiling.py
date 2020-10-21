from __future__ import annotations

import heapq
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


def round_up(val: int, base: int = 2):
    return (val + base - 1) // base * base


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
                 flag: Union[-1, 0, 1] = 0):
        super().__init__(params)
        self.flag = flag
        self.free_slots = [True for _ in range(10)]  # slots IN BETWEEN edges
        self.free_slots_saved = self.free_slots.copy()

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

    def book_slots(self, slot: Union[int, Sequence[int]]):
        if isinstance(slot, int):
            self.free_slots[slot % 10] = False
        else:
            for i in slot:
                self.free_slots[i % 10] = False

    def unbook_slots(self):
        self.free_slots = self.free_slots_saved.copy()

    def save_slots(self):
        self.free_slots_saved = self.free_slots.copy()

    def is_free(self, slot: Union[int, Sequence[int]]):
        if isinstance(slot, int):
            return self.free_slots[slot % 10]
        else:
            for i in slot:
                if self.free_slots[i % 10] is False:
                    return False
            return True

    def get_max_angle(self, direction: int, orientation: bool = True):
        """
        Get max size of angle with base in <direction>, oriented CCW (True) or CW (False)
        """
        angle = 0
        if orientation:
            # Counter-clockwise
            while angle < 10:
                if not self.free_slots[(direction + angle) % 10]:
                    break
                else:
                    angle += 1
        else:
            # Clockwise
            while angle < 10:
                if not self.free_slots[(direction - angle - 1) % 10]:
                    break
                else:
                    angle += 1
        return angle

    def take_if_free(self, slot: Union[int, Sequence[int]]):
        # TODO: inefficient?
        if not self.is_free(slot):
            print("node is not free", file=sys.stderr)
            raise ValueError("node is not free")
            # self.take_slot(slot)
        else:
            self.book_slots(slot)

    def get_free_stretch(self):
        if True not in self.free_slots:
            return 0, 0
        if False in self.free_slots:
            # "rotate" the list so that free_slots[-1] == False and free_slots[0] == True
            start = self.free_slots.index(False)
            while self.free_slots[start] is False:
                start = (start + 1) % 10
            stretch = self.get_max_angle(start, True)
        else:
            start = random.randint(0, 9)
            stretch = 10
        return start, stretch

    def get_xy(self, edge_length: float):
        x = edge_length * (self._p[0] + np.sqrt(5) * self._p[1]) / 4
        y = edge_length * (np.sin(np.pi / 5) * self._p[2] + np.sin(np.pi * 2 / 5) * self._p[3])
        if INVERT_Y:
            y = -y
        return x, y

    def __lt__(self, other: PenroseNode4D):
        # TODO: need to find better solution for frontier sorting, this is ugly
        return sum(self.free_slots) < sum(other.free_slots)
    
    
class RhombNet:
    def __init__(self):
        self.nodes: List[PenroseNode4D] = []
        self.edges: List[Tuple[PenroseNode4D, PenroseNode4D]] = []
        self.rhombs: List[Tuple[PenroseNode4D, PenroseNode4D, PenroseNode4D, PenroseNode4D, bool]] = []
        self.frontier: List[Tuple] = []
        self.debug = []
        self.stopflag = False

    def add_node(self, node: PenroseNode4D, add_to_frontier: bool = False) -> (int, PenroseNode4D):
        index = self.find_node(node)
        if index > -1:
            return index, self.nodes[index]
        self.nodes.append(node)
        index = len(self.nodes) - 1
        if add_to_frontier:
            self.add_to_frontier(index)
        return index, node

    def add_to_frontier(self, index: int):
        AGE_FACTOR = 6
        heapq.heappush(self.frontier, (index // AGE_FACTOR, self.get_node(index)))

    def pop_from_frontier(self):
        # heapq.heapify(self.frontier)
        return heapq.heappop(self.frontier)[1]

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

    def construct_rhomb(self, index: int, direction: int, angle: int):
        """
        :param index: Node index
        :param direction: Direction to expand towards, must be in [0, 10)
        :param angle: Angle to expand with, must be in {1, 2, 3, 4}
        :return: True if successful, False if fail
        """
        assert angle in (1, 2, 3, 4)
        start = direction % 10
        iA = index
        A = self.nodes[iA]
        if not A.is_free(range(start, start + angle)):
            print("A", A.free_slots, file=sys.stdout)
            return False

        B = A.step(start)
        iB, B = self.add_node(B, True)
        if not B.is_free(range(start + angle, start + 5)):
            print("B", B.free_slots, file=sys.stdout)
            return False

        C = B.step(start + angle)
        iC, C = self.add_node(C, True)
        if not C.is_free(range(start + 5, start + 5 + angle)):
            print("C", C.free_slots, file=sys.stdout)
            return False

        D = C.step(start + 5)
        iD, D = self.add_node(D, True)
        if not D.is_free(range(start + 5 + angle, start + 10)):
            print("D", D.free_slots, file=sys.stdout)
            return False

        A.book_slots(range(start, start + angle))
        B.book_slots(range(start + angle, start + 5))
        C.book_slots(range(start + 5, start + 5 + angle))
        D.book_slots(range(start + 5 + angle, start + 10))
        self.add_edge(iA, iB)
        self.add_edge(iB, iC)
        self.add_edge(iC, iD)
        self.add_edge(iD, iA)
        self.add_rhomb(A, B, C, D, angle in (1, 4))
        return True

    def add_rhomb(self, a, b, c, d, rhomb_type):
        self.rhombs.append((a, b, c, d, rhomb_type))

    def expand_node(self, node: Optional[PenroseNode4D] = None):
        if node is None:
            node = self.pop_from_frontier()
        print(f"Expanding node {node}")
        free_slots = node.free_slots
        # print(f"free_slots {free_slots}")
        while True in free_slots:
            alpha, stretch = node.get_free_stretch()
            beta = alpha + stretch
            # print(f"Start direction: {alpha}, end: {beta}")

            # check the edge angles
            alpha_node = node.step(alpha)
            i_alpha_node, alpha_node = self.add_node(alpha_node)
            beta_node = node.step(beta)
            i_beta_node, beta_node = self.add_node(beta_node)
            min_alpha_angle = max(5 - alpha_node.get_max_angle(direction=alpha + 5, orientation=False), 1)
            min_beta_angle = max(5 - beta_node.get_max_angle(direction=beta + 5, orientation=True), 1)
            # print(f"a {min_alpha_angle}, b {min_beta_angle}")

            # split the stretch
            # print(f"a>={min_alpha_angle}, b>={min_beta_angle}, {stretch}")
            if stretch < min_alpha_angle + min_beta_angle:
                if stretch > 4:
                    raise RuntimeError(f"Unexpandable node {node}")
                else:
                    angles = [stretch]
            else:
                alpha_angle = random.randint(min_alpha_angle, min(4, stretch - min_beta_angle))
                beta_angle = random.randint(min_beta_angle, min(4, stretch - alpha_angle))
                angles = [alpha_angle]
                while sum(angles) < stretch - beta_angle:
                    # angle = min(4, stretch - sum(angles))
                    # while angle > 1:
                    #     if random.random() > 0.618:
                    #         angle -= 1
                    #     else:
                    #         break
                    # print(f"stretch = {remaining_stretch}")
                    angle = sum(
                        [random.randint(1, min(4, stretch - beta_angle - sum(angles))) for _ in range(10)]) // 10
                    # angle = random.randint(1, min(4, stretch - sum(angles)))
                    angles.append(angle)
                    print(angles)
                angles.append(beta_angle)
            # print(f"Angles: {angles}, a>={min_alpha_angle}, b>={min_beta_angle}, {stretch}")

            # build the rhombs
            iA, A = self.add_node(node)
            for angle in angles:
                # print(f"angle: {angle}")
                if not self.construct_rhomb(iA, alpha, angle):
                    break
                alpha += angle
        # TODO: save node (and edge) status before expanding, keep new changes separate, submit them when sure
        # print(f"Expanding node {node} completed")

    def expand_penrose_node(self, node: Optional[PenroseNode4D] = None):
        if node is None:
            node = self.pop_from_frontier()
        print(f"Expanding node {node}, flag {node.flag}")
        free_slots = node.free_slots
        # print(f"free_slots {free_slots}")
        while True in free_slots:
            alpha, stretch = node.get_free_stretch()
            beta = alpha + stretch
            print(f"Start direction: {alpha}, end: {beta}")

            if node.flag == 1 and stretch % 2:
                raise RuntimeError(f"Unexpandable node {node}: marked nodes must have an even number of slots.")

            # check the edge angles
            alpha_node = node.step(alpha)
            i_alpha_node, alpha_node = self.add_node(alpha_node)
            beta_node = node.step(beta)
            i_beta_node, beta_node = self.add_node(beta_node)
            min_alpha_angle = max(5 - alpha_node.get_max_angle(direction=alpha + 5, orientation=False), 1)
            min_beta_angle = max(5 - beta_node.get_max_angle(direction=beta + 5, orientation=True), 1)
            if node.flag == 1:
                min_alpha_angle = round_up(min_alpha_angle)
                min_beta_angle = round_up(min_beta_angle)
            # print(f"a {min_alpha_angle}, b {min_beta_angle}")

            # TODO: continue from here
            # split the stretch
            # print(f"a>={min_alpha_angle}, b>={min_beta_angle}, {stretch}")
            if stretch < min_alpha_angle + min_beta_angle:
                if stretch > 4:
                    raise RuntimeError(f"Unexpandable node {node}")
                else:
                    angles = [stretch]
            else:
                alpha_angle = random.randint(min_alpha_angle, min(4, stretch - min_beta_angle))
                beta_angle = random.randint(min_beta_angle, min(4, stretch - alpha_angle))
                angles = [alpha_angle]
                while sum(angles) < stretch - beta_angle:
                    # angle = min(4, stretch - sum(angles))
                    # while angle > 1:
                    #     if random.random() > 0.618:
                    #         angle -= 1
                    #     else:
                    #         break
                    # print(f"stretch = {remaining_stretch}")
                    angle = sum(
                        [random.randint(1, min(4, stretch - beta_angle - sum(angles))) for _ in range(10)]) // 10
                    # angle = random.randint(1, min(4, stretch - sum(angles)))
                    angles.append(angle)
                    print(angles)
                angles.append(beta_angle)
            # print(f"Angles: {angles}, a>={min_alpha_angle}, b>={min_beta_angle}, {stretch}")

            # build the rhombs
            iA, A = self.add_node(node)
            for angle in angles:
                # print(f"angle: {angle}")
                if not self.construct_rhomb(iA, alpha, angle):
                    break
                alpha += angle
        # TODO: save node (and edge) status before expanding, keep new changes separate, submit them when sure
        # print(f"Expanding node {node} completed")

    def get_node_xy(self, index: int, edge_length: float) -> Tuple[int, int]:
        return self.nodes[index].get_xy(edge_length)

    def generate_edges(self, edge_length: float):
        for n1, n2 in self.edges:
            yield *n1.get_xy(edge_length), *n2.get_xy(edge_length)

    def consume_edges(self, edge_length: float):
        for n1, n2 in self.edges:
            yield *n1.get_xy(edge_length), *n2.get_xy(edge_length)
        self.edges = []


def random_walk(graph: RhombNet):
    d = random.randint(0, 9)
    d_last = None
    _, last_node = graph.add_node(PenroseNode4D())
    while True:
        while d_last in (d, (d + 5) % 10):
            d = random.randint(0, 10)
        _, this_node = graph.add_node(last_node.step(d))
        yield last_node, this_node
        last_node = this_node
        d_last = d


def random_tiling(graph: RhombNet, mode: str = "edges"):
    last, last_node = graph.add_node(PenroseNode4D(), True)
    while True:
        graph.expand_node()
        if mode == "edges":
            yield from graph.edges
            if graph.stopflag:
                break
            graph.edges = []
        elif mode == "rhombs":
            yield from graph.rhombs
            if graph.stopflag:
                break
            graph.rhombs = []


def random_penrose_tiling(graph: RhombNet, mode: str = "edges"):
    graph.add_node(PenroseNode4D(flag=1), True)
    while True:
        graph.expand_node()
        if mode == "edges":
            yield from graph.edges
            if graph.stopflag:
                break
            graph.edges = []
        elif mode == "rhombs":
            yield from graph.rhombs
            if graph.stopflag:
                break
            graph.rhombs = []


def test_circle(graph: RhombNet):
    last, last_node = graph.add_node(PenroseNode4D())
    for d in range(10):
        this_node = last_node.step(d)
        print(d, this_node._p)
        yield last_node, this_node
        last_node = this_node


def test_star(graph: RhombNet):
    last, last_node = graph.add_node(PenroseNode4D())
    for d in range(10):
        this_node = last_node.step(d)
        print(d, this_node._p)
        yield last_node, this_node
