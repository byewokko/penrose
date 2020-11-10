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
    """
    Only holds mapping of directions to edges.
    """

    def __init__(self, edges: Dict[int, tuple]):
        self.edges = edges

    def type(self):
        return (max(self.edges.keys()) - min(self.edges.keys())) % 5

    def get_vertices(self):
        direction = next(iter(self.edges.keys()))
        a, b = self.edges[direction]
        c, d = self.edges[(direction + 5) % 10]
        return a, b, c, d

    def get_edges(self):
        return self.edges.values()

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
        self._grid_edges = pentagrid.intersections_to_edges_dict(grid_nodes)
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
            for direction, adjacent in self._grid_edges[grid_node].items():
                if adjacent not in self._expanded_nodes:
                    self.frontier_push(adjacent)
                    self._rhombs[adjacent] = self.build_adjacent_rhomb(grid_node, direction, adjacent)
                    yield self._rhombs[adjacent]

    def build_adjacent_rhomb(self, this_node: tuple, this_direction: int, adjacent_node: tuple):
        this_rhomb = self._rhombs[this_node]
        shared_edge = this_rhomb.edges[this_direction]
        cd_direction = this_direction
        ab_direction = (cd_direction + 5) % 10
        if adjacent_node[0] == ab_direction % 5:
            cross_direction = adjacent_node[1]
        elif adjacent_node[1] == ab_direction % 5:
            cross_direction = adjacent_node[0]
        else:
            raise ValueError(f"Direction {ab_direction} not found in {adjacent_node}")

        # The edges in a rhomb must follow in ascending order, with steps no larger than 4
        if (ab_direction < cross_direction < ab_direction + 5
                or not (ab_direction - 5 < cross_direction < ab_direction)):
            bc_direction = cross_direction
            da_direction = (cross_direction + 5) % 10
        else:
            da_direction = cross_direction
            bc_direction = (cross_direction + 5) % 10

        # invert the edge
        b, a = shared_edge
        c = b.step(bc_direction)
        d = a.step(bc_direction)
        ab = shared_edge
        bc = (b, c)
        cd = (c, d)
        da = (d, a)
        rhomb_edges = {
            ab_direction: ab,
            bc_direction: bc,
            cd_direction: cd,
            da_direction: da
        }
        return Rhomb(rhomb_edges)

    def frontier_push(self, grid_node: tuple):
        # check if the node has enough neighbors
        if len(self._grid_edges[grid_node]) < 4:
            L.debug(f"Node {grid_node} has only {len(self._grid_edges[grid_node])} neighbors. Skipping.")
            return
        item = (self._frontier_counter, grid_node)
        heapq.heappush(self._frontier, item)
        self._frontier_counter += 1

    def frontier_pop(self):
        _, grid_node = heapq.heappop(self._frontier)
        return grid_node

    def build_start_rhomb(self,
                          node: tuple,
                          start_vertex: Optional[Vertex] = None):
        if not start_vertex:
            # This is the lower left vertex of the rhomb
            start_vertex = Vertex.get_vertex((0, 0, 0, 0))

        ab_direction, cross_direction = node
        # The edges in a rhomb must follow in ascending order, with steps no larger than 4
        if (ab_direction < cross_direction < ab_direction + 5
                or not (ab_direction - 5 < cross_direction < ab_direction)):
            bc_direction = cross_direction
        else:
            bc_direction = (cross_direction + 5) % 10
        a = start_vertex
        b = a.step(ab_direction)
        c = b.step(bc_direction)
        d = a.step(bc_direction)
        ab = (a, b)
        bc = (b, c)
        cd = (c, d)
        da = (d, a)
        rhomb_edges = {
            ab_direction: ab,
            bc_direction: bc,
            (ab_direction + 5) % 10: cd,
            (bc_direction + 5) % 10: da
        }
        return Rhomb(rhomb_edges)

    def add_rhomb(self, grid_node: tuple, rhomb: Rhomb):
        self._rhombs[grid_node] = rhomb


def main():
    draw = Draw(scale=30)
    index_range = (-5, 5)
    grid = pentagrid.Pentagrid()
    tiling_builder = TilingBuilder(grid)
    tiling_builder.prepare_grid(index_range)
    for rhomb in tiling_builder.generate_rhombs(n_rhombs=5):
        draw.polygon(rhomb.xy())


if __name__ == "__main__":
    main()
