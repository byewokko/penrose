from __future__ import annotations

import heapq
import random
import sys
import logging
from typing import Optional, Union, Tuple, List, Sequence, Any, Literal, Dict
import numpy as np

from naive import tiling
import pentagrid
from drawing.pil_draw_simple import Draw

L = logging.getLogger(__name__)
L.setLevel(logging.DEBUG)


def normalize_rhomb_type(rhomb_type):
    e1 = (rhomb_type[0] * 2) % 5
    e2 = (rhomb_type[1] * 2) % 5
    if e1 > e2:
        e1, e2 = e2, e1
    return e1, e2


def get_ribbon_and_direction(adjacent_node, this_node):
    if this_node[0] == adjacent_node[0]:
        ribbon = this_node[0]
        direction = adjacent_node[2] - this_node[2]
    elif this_node[0] == adjacent_node[1]:
        ribbon = this_node[0]
        direction = adjacent_node[3] - this_node[2]
    elif this_node[1] == adjacent_node[0]:
        ribbon = this_node[1]
        direction = adjacent_node[2] - this_node[3]
    elif this_node[1] == adjacent_node[1]:
        ribbon = this_node[1]
        direction = adjacent_node[3] - this_node[3]
    else:
        L.error(f"No matching ribbon for {this_node} and {adjacent_node}")
        ribbon, direction = None, None
    assert direction in (-1, 1)
    return ribbon, direction


class Vertex(tiling.Node4D):
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

    _all_vertices = {}

    @classmethod
    def get_vertex(cls, coords):
        if tuple(coords) in cls._all_vertices:
            return cls._all_vertices[tuple(coords)]
        vertex = cls(coords)
        cls._all_vertices[tuple(coords)] = vertex
        return vertex

    def __init__(self, coords: Optional[Sequence[int]] = None):
        super().__init__(coords)

    def step(self, direction):
        direction = direction % 10
        return self.get_vertex(self + self._step[direction])

    def get_xy(self, edge_length: float, homogenous: bool = True):
        x = edge_length * (self._p[0] + np.sqrt(5) * self._p[1]) / 4
        y = edge_length * (np.sin(np.pi / 5) * self._p[2] + np.sin(np.pi * 2 / 5) * self._p[3])
        if homogenous:
            return np.asarray([x, y, 1])
        return x, y


class Edge:
    _all_edges = set()

    @classmethod
    def exists(cls, a: Vertex, b: Vertex):
        if a > b:
            return (b, a) in cls._all_edges
        else:
            return (a, b) in cls._all_edges

    @classmethod
    def add(cls, a: Vertex, b: Vertex):
        if a > b:
            cls._all_edges.add((b, a))
        else:
            cls._all_edges.add((a, b))

    @classmethod
    def remove(cls, a: Vertex, b: Vertex):
        if a > b:
            cls._all_edges.remove((b, a))
        else:
            cls._all_edges.remove((a, b))


class Rhomb:

    # TODO: find good data structure for rhomb

    def __init__(self, vertices: Sequence[Edge], type: Tuple[int, int]):
        raise NotImplementedError
        # self._edges: Dict[Tuple[int, int, int, int], Rhomb] = {}
        self._edges = {}

    def xy(self, edge_length: float = 1.):
        raise NotImplementedError


class Tiling:
    def __init__(self):
        self._vertices: Dict[Tuple[int, int, int, int], Vertex] = {}
        self._rhombs: Dict[Tuple[int, int, int, int], Rhomb] = {}
        self._edges = {}

    def get_vertex(self, coords):
        if tuple(coords) in self._vertices:
            return self._vertices[tuple(coords)]
        vertex = Vertex(coords)
        self._vertices[tuple(coords)] = vertex
        return vertex

    def n_rhombs(self):
        return len(self._rhombs)


class TilingBuilder:
    """
    Generates Penrose tiling from a given Pentagrid, using rhomb tiles.
    """
    def __init__(self, grid: pentagrid.Pentagrid):
        self._grid = grid
        self._grid_edges = None
        self._vertices: Dict[Tuple[int, int, int, int], Vertex] = {}
        self._rhombs: Dict[Tuple[int, int, int, int], Rhomb] = {}
        self._edges = {}
        self._frontier = []
        self._frontier_counter = 0
        self._expanded_nodes = set()

    def prepare_grid(self, index_range: Tuple[int, int]):
        L.debug("Calculating pentagrid edges")
        grid_nodes = self._grid.calculate_intersections(index_range)
        self._grid_edges = pentagrid.intersections_to_edges(grid_nodes)
        # TODO: remove nodes with 3 or less neighbors
        L.info("Pentagrid edges ready")

    def generate_rhombs(self,
                        start_node: Tuple[int, int, int, int] = (0, 1, 0, 0),
                        n_rhombs: int = None):
        assert self._grid_edges
        self.frontier_push(start_node)
        self.add_rhomb(start_node, self.build_start_rhomb(start_node[:2]))
        yield self._rhombs[start_node]
        while True:
            # Stopping conditions
            if len(self._rhombs) >= n_rhombs:
                break
            grid_node = self.frontier_pop()
            if not grid_node:
                break
            # Add adjacents to frontier and build them
            for adjacent in self._grid_edges[grid_node]:
                if adjacent not in self._expanded_nodes:
                    self.frontier_push(adjacent)
                    self.build_adjacent_rhomb(grid_node, adjacent)
                    yield self._rhombs[adjacent]

    def build_adjacent_rhomb(self, this_node, adjacent_node):
        # find out rhomb types
        n0_type = this_node[:2]
        n1_type = adjacent_node[:2]
        ribbon, direction = get_ribbon_and_direction(adjacent_node, this_node)
        this_rhomb = self._rhombs[this_node]
        shared_edge = this_rhomb.get_edge(ribbon, direction)  # TODO: add rhomb method
        if ribbon == adjacent_node[0]:
            cross_ribbon = adjacent_node[1]
        else:
            cross_ribbon = adjacent_node[0]
        if cross_ribbon in ((ribbon + 1) % 5, (ribbon + 2) % 5):
            cross_direction = direction
        else:
            cross_direction = -direction
        a = shared_edge[0]  # TODO: add edge __get__ method
        b = shared_edge[1]
        d = a.step(cross_ribbon, cross_direction)  # TODO: update vertex.step() method
        c = b.step(cross_ribbon, cross_direction)
        ab = shared_edge
        bc = Edge.get_edge(b, c)
        cd = Edge.get_edge(c, d)
        da = Edge.get_edge(d, a)
        rhomb_edges = {
            (ribbon, -direction): ab,
            (cross_ribbon, cross_direction): bc,
            (ribbon, direction): cd,
            (cross_ribbon, cross_direction): da
        }
        self.add_rhomb(adjacent_node, Rhomb(rhomb_edges))

    def frontier_push(self, grid_node: tuple):
        item = (self._frontier_counter, grid_node)
        heapq.heappush(self._frontier, item)
        self._frontier_counter += 1

    def frontier_pop(self):
        _, grid_node = heapq.heappop(self._frontier)
        return grid_node

    def build_start_rhomb(self,
                          rhomb_type: Tuple[int, int],
                          start_vertex: Optional[Vertex] = None):
        if not start_vertex:
            # This is the lower left vertex of the rhomb
            start_vertex = Vertex.get_vertex((0, 0, 0, 0))
        # TODO: rewrite this like build_adjacent_rhomb
        e1, e2 = normalize_rhomb_type(rhomb_type)
        a = start_vertex
        b = a.step(e1)
        d = a.step(e2)
        c = b.step(e2)
        return Rhomb((a, b, c, d), type=(e1, e2))

    def add_rhomb(self, grid_node: tuple, rhomb: Rhomb):
        self._rhombs[grid_node] = rhomb


def main():
    draw = Draw(scale=30)
    index_range = (-4, 4)
    grid = pentagrid.Pentagrid()
    tiling_builder = TilingBuilder(grid)
    tiling_builder.prepare_grid(index_range)
    for rhomb in tiling_builder.generate_rhombs(n_rhombs=5):
        draw.polygon(rhomb.xy())


if __name__ == "__main__":
    main()
