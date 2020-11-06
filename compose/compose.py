from __future__ import annotations

import heapq
import random
import sys
from typing import Optional, Union, Tuple, List, Sequence, Any, Literal
import numpy as np

from naive import tiling

"""
An attempt to create Penrose tiling with bottom up approach. The rhomb tiles are generated randomly, 
obeying only local rules. It doesn't work, because local approach cannot work, as shown in the linked articles.

Plus, my implementation is buggy.

But I might use it with the pentagrid oracle later.
"""


#debug

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


class RhombNode(tiling.Node4D):
    _step = tuple(tiling.Node4D(x) for x in [
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

    @classmethod
    def get_direction_from_step(cls, step: tiling.Node4D):
        return cls._step.index(step)
    
    def __init__(self,
                 params: Optional[Sequence[int, int, int, int]] = None,
                 flag: Any = None):
        super().__init__(params)
        self.flag = flag
        self.free_slots = [True for _ in range(10)]  # slots IN BETWEEN edges

    def step(self, direction):
        direction = direction % 10
        return self + self._step[direction]

    def book_slots(self, slot: Union[int, Sequence[int]]):
        if isinstance(slot, int):
            self.free_slots[slot % 10] = False
        else:
            for i in slot:
                self.free_slots[i % 10] = False

    def is_free(self, slot: int):
        return self.free_slots[slot % 10]

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

    def is_angle_free(self, direction: int, angle: int, orientation: int):
        if orientation == 1:
            for x in range(direction, direction + angle):
                if not self.is_free(x):
                    return False
        elif orientation == -1:
            for x in range(direction - 1, direction - angle - 1, -1):
                if not self.is_free(x):
                    return False
        return True

    def occupy_angle(self, direction: int, angle: int, orientation: int = 1):
        if orientation == 1:
            for x in range(direction, direction + angle):
                if not self.free_slots[x % 10]:
                    raise ValueError(f"{self}: slot {x} is already occupied")
                self.free_slots[x % 10] = False
        elif orientation == -1:
            for x in range(direction - 1, direction - angle - 1, -1):
                if not self.free_slots[x % 10]:
                    raise ValueError(f"{self}: slot {x} is already occupied")
                self.free_slots[x % 10] = False


class RhombEdge:
    def __init__(self,
                 a: RhombNode,
                 b: RhombNode,
                 value: Union[-2, -1, 1, 2],
                 free=None):
        if free is None:
            free = [True, True]
        self._nodes = (a, b)
        self._flipped_view = False
        self._free = free  # is there NOT a rhomb (on the left, on the right)
        self._value = value
        self._direction = RhombNode.get_direction_from_step(b - a)

    def __repr__(self):
        val_dict = {-2: "<<", -1: "<", 1: ">", 2: ">>"}
        return f"Edge({str(self._nodes[0])} {val_dict[self._value]} {str(self._nodes[1])})"

    def __eq__(self, other: RhombEdge):
        """
        Doesn't concern direction nor value, only the nodes. May be misleading.
        """
        if (self.a() == other.a() and self.b() == other.b()) or (self.a() == other.b() and self.b() == other.a()):
            return True
        return False

    def __lt__(self, other):
        """
        Required for the heap queue.
        """
        return False

    def a(self):
        if self._flipped_view:
            return self._nodes[1]
        return self._nodes[0]

    def b(self):
        if self._flipped_view:
            return self._nodes[0]
        return self._nodes[1]

    def value(self):
        if self._flipped_view:
            return -self._value
        return self._value

    def direction(self):
        if self._flipped_view:
            return (self._direction + 5) % 10
        return self._direction

    def is_free(self, side: Union[Literal["left", "right", "any"], 1, -1, 0] = 0):
        if side in (1, "left"):
            return self._free[0 - self._flipped_view]
        elif side in (-1, "right"):
            return self._free[1 - self._flipped_view]
        elif side in (0, "any"):
            return True in self._free
        else:
            raise ValueError(f"Invalid argument 'side': '{side}'")

    def occupy_side(self, side: Union[Literal["left", "right"], 1, -1]):
        if side in (1, "left"):
            self._free[0 - self._flipped_view] = False
        elif side in (-1, "right"):
            self._free[1 - self._flipped_view] = False
        else:
            raise ValueError(f"Invalid argument 'side': {side}")

    def has_node(self, node: RhombNode):
        return node in self._nodes

    def flipped_copy(self):
        return RhombEdge(self.b(), self.a(), -self.value(), [self._free[1], self._free[0]])

    def flip_view(self):
        self._flipped_view = not self._flipped_view

    def can_build_rhomb(self, a_angle: int, side: Union[Literal["left", "right"], 0, 1] = "left"):
        if side in (1, "right"):
            raise NotImplementedError()
        return self.a().is_angle_free(self.direction(), a_angle, 1) \
            and self.b().is_angle_free(self.direction() + a_angle, 5 - a_angle, 1)

    def get_xy_coordinates(self, edge_length: float):
        return *self.a().get_xy(edge_length), *self.b().get_xy(edge_length)
    
    
class RhombNet:
    def __init__(self):
        self.nodes: List[RhombNode] = []
        self.edges: List[RhombEdge] = []
        self.rhombs: List[Tuple[RhombNode, RhombNode, RhombNode, RhombNode, bool]] = []
        self.node_frontier: List[Tuple] = []
        self.edge_frontier: List[Tuple] = []
        self.debug = []
        self.stopflag = False

    def add_node(self, node: RhombNode) -> RhombNode:
        """
        Every used node should be added here before performing any operations on it.
        If it is already in the list, it is replaced by the older one.
        """
        index = self.find_node(node)
        if index > -1:
            if self.nodes[index].flag == 0:
                self.nodes[index].flag = node.flag
            return self.nodes[index]
        self.nodes.append(node)
        return node

    def add_edge(self, edge: RhombEdge) -> RhombEdge:
        """
        Every used edge should be added here before performing any operations on it.
        If it is already in the list, it is replaced by the older one.
        """
        # FIXME: doesn't always work
        index = self.find_edge(edge)
        if index > -1:
            if edge.value() == self.edges[index].value():
                if not edge.is_free(1):
                    self.edges[index].occupy_side(1)
                if not edge.is_free(-1):
                    self.edges[index].occupy_side(-1)
                return self.edges[index]
            elif edge.value() == -self.edges[index].value():
                self.edges[index].flip_view()
                if not edge.is_free(1):
                    self.edges[index].occupy_side(1)
                if not edge.is_free(-1):
                    self.edges[index].occupy_side(-1)
                return self.edges[index]
            else:
                raise ValueError(f"Edge values do not match: old {self.edges[index].value()}, new {edge.value()}")
        self.edges.append(edge)
        return edge

    def edge_frontier_add(self, edge: RhombEdge):
        # Heap is probably not necessary
        AGE_FACTOR = 4
        index = self.find_edge(edge)
        if index == -1:
            raise RuntimeError(f"{edge} is not in edge list!")
        heapq.heappush(self.edge_frontier, (index // AGE_FACTOR, edge))

    def edge_frontier_pop(self):
        # heapq.heapify(self.frontier)
        return heapq.heappop(self.edge_frontier)[1]

    def find_node(self, node: RhombNode) -> int:
        if node not in self.nodes:
            return -1
        return self.nodes.index(node)

    def find_edge(self, edge: RhombEdge) -> int:
        if edge not in self.edges:
            return -1
        return self.edges.index(edge)

    def get_node(self, index: int) -> RhombNode:
        return self.nodes[index]

    def add_rhomb(self, a, b, c, d, rhomb_type):
        self.rhombs.append((a, b, c, d, rhomb_type))

    def expand_edge(self, edge: RhombEdge):
        edge = self.add_edge(edge)
        print(f"Expanding {edge} ...")
        if not edge.is_free(1):
            if edge.is_free(-1):
                edge.flip_view()
            else:
                print(f"Nowhere to expand at {edge}", file=sys.stderr)
                return

        if edge.value() == -2:
            possible_angles = [4, 2]
        elif edge.value() == -1:
            possible_angles = [4, 3]
        elif edge.value() == 1:
            possible_angles = [1, 2]
        elif edge.value() == 2:
            possible_angles = [1, 3]
        else:
            raise ValueError(f"Invalid edge value: {edge.value()}")
        random.shuffle(possible_angles)

        a_angle = None
        while possible_angles:
            a_angle = possible_angles.pop()
            if edge.can_build_rhomb(a_angle):
                break
            a_angle = None

        if not a_angle:
            print(f"Cannot expand {edge}", file=sys.stderr)
            return
        self.build_rhomb_from_edge(edge, a_angle)
        print(f"Graph contains {len(self.edges)} edges. {len(self.edge_frontier)} edges in frontier.")

    def build_rhomb_from_edge(self, edge: RhombEdge, a_angle: int):
        angle_to_values = {
            1: [1, -1, 2, -2],
            2: [1, 2, -2, -1],
            3: [-1, 1, 2, -2],
            4: [-1, 2, -2, 1]
        }
        edge_values = angle_to_values[a_angle]
        if edge_values[0] != edge.value():
            edge_values = edge_values[2:4] + edge_values[0:2]
        a = self.add_node(edge.a())
        b = self.add_node(edge.b())
        c = self.add_node(b.step(edge.direction() + a_angle))
        d = self.add_node(a.step(edge.direction() + a_angle))
        a.occupy_angle(edge.direction(), a_angle, 1)
        b.occupy_angle(edge.direction() + a_angle, 5 - a_angle, 1)
        c.occupy_angle(edge.direction() + 5, a_angle, 1)
        d.occupy_angle(edge.direction() + a_angle + 5, 5 - a_angle, 1)
        ab = edge
        ab.occupy_side("left")
        bc = self.add_edge(RhombEdge(b, c, edge_values[1], [False, True]))
        cd = self.add_edge(RhombEdge(c, d, edge_values[2], [False, True]))
        da = self.add_edge(RhombEdge(d, a, edge_values[3], [False, True]))
        # clear()
        # draw_edges(self)
        # draw_node(a, "yellow")
        # draw_nodes([b, c, d])
        # for e in (ab, bc, cd, da):
        #     draw_edge_arrow(e)
        # show()
        for e in ab, bc, cd, da:
            if e.is_free("any"):
                self.edge_frontier_add(e)
        self.add_rhomb(a, b, c, d, a_angle in (1, 4))

    def generate_edges(self, edge_length: float):
        for n1, n2, t in self.edges:
            yield *n1.get_xy(edge_length), *n2.get_xy(edge_length)

    def consume_edges(self, edge_length: float):
        for n1, n2, t in self.edges:
            yield *n1.get_xy(edge_length), *n2.get_xy(edge_length)
        self.edges = []


def random_tiling(graph: RhombNet, mode: str = "edges"):
    a = graph.add_node(RhombNode())
    b = graph.add_node(a.step(0))
    e = graph.add_edge(RhombEdge(a, b, 1))
    graph.edge_frontier_add(e)
    while True:
        graph.expand_edge(graph.edge_frontier_pop())
        if mode == "edges":
            yield from graph.edges
            if graph.stopflag:
                break
        elif mode == "rhombs":
            yield from graph.rhombs
            if graph.stopflag:
                break


def main():
    graph = RhombNet()
    a = graph.add_node(RhombNode())
    b = graph.add_node(a.step(0))
    e = graph.add_edge(RhombEdge(a, b, 1))
    graph.edge_frontier_add(e)
    while True:
        graph.expand_edge(graph.edge_frontier_pop())


if __name__ == "__main__":
    main()
