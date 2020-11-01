from __future__ import annotations

import heapq
import random
import sys
from typing import Optional, Union, Tuple, List, Sequence, Deque, Any, Iterable, Dict, FrozenSet, Set
from collections import OrderedDict, deque
import numpy as np
import itertools

import pil_draw_simple as draw
from structures import FrozenSetDict


def _set_base_lines():
    lines = []
    for d in range(5):
        alpha = d * 2 * np.pi / 5
        lines.append([np.sin(alpha), -np.cos(alpha), 0])
    return np.asarray(lines)


def _set_intersections(lines: np.ndarray):
    intersections = {}
    for i in range(5 - 1):
        for j in range(i + 1, 5):
            cross = intersection(lines[i], lines[j])
            if cross is None:
                raise ZeroDivisionError("No intersection between parallel lines.")
            intersections[frozenset([i, j])] = cross
    return intersections


def intersection(l: Union[np.ndarray, list], m: Union[np.ndarray, list]):
    cross = np.cross(l, m)
    if cross[2] == 0:
        return None
    return cross / cross[2]


class Pentagrid:
    """
    Five families of equidistant parallel lines.
    The angle between each two families is an integer multiple of 2*PI/5.
    No more than two lines may intersect at any given point.
    Each area delimited by the lines is equivalent to a node in penrose tiling.

    Thi class uses homogeneous normal representation of lines and points.
    """
    GROUPS = 5

    def __init__(self):
        self._base_offset = np.asarray([[0, 0, np.sqrt(np.random.random() + 1) - 1] for _ in range(self.GROUPS)])
        self._base_lines = _set_base_lines()
        self._base_intersections = _set_intersections(self._base_lines + self._base_offset)
        self.grid_nodes: Dict[FrozenSet[Tuple[int, int]], np.ndarray] = {}
        self.grid_edges: Set[FrozenSet[FrozenSet[Tuple[int, int]]]] = set()

    def get_line(self, group: int, index: float):
        return self._base_lines[group] + self._base_offset[group] + [0, 0, index]

    def get_line_x(self, group: int, index: float, y: float):
        line = self.get_line(group, index)
        y = [0, -1, y]
        cross = intersection(line, y)
        if cross is not None:
            cross = cross[0]
        return cross

    def get_line_y(self, group: int, index: float, x: float):
        line = self.get_line(group, index)
        x = [-1, 0, x]
        cross = intersection(line, x)
        if cross is not None:
            cross = cross[1]
        return cross

    def calculate_grid_edges(self, index_range: Tuple[int, int]):
        """
        For each line:
        - get instersections with all the other lines
        - put them in heapq cast as tuples (ndarray cannot __lt__)
        - generate edges
        """
        for g1 in range(self.GROUPS-1):
            for i1 in range(*index_range):
                # print(g1, i1)
                a = self.get_line(g1, i1)
                heap = []
                for g2 in range(g1 + 1, self.GROUPS):
                    for i2 in range(*index_range):
                        b = self.get_line(g2, i2)
                        p = frozenset([(g1, i1), (g2, i2)])
                        pxy = intersection(a, b)
                        self.grid_nodes[p] = pxy
                        heapq.heappush(heap, (tuple(pxy), p))
                _, p = heapq.heappop(heap)
                while heap:
                    p0 = p
                    _, p = heapq.heappop(heap)
                    self.grid_edges.add(frozenset((p0, p)))

    def calculate_faces(self, index_range: Tuple[int, int]):
        """
        - get all instersections (in one array)
        - for each group:
            - dot product of all the points with each line
            - count >0 for each point --> index
        """
        for g1 in range(self.GROUPS - 1):
            lines = np.stack([self.get_line(g1, i) for i in index_range]).T


def save_vertex_dict(vertex_dict, filename):
    with open(filename, "w") as f:
        for node, xyz in vertex_dict.items():
            print(",".join(map(str, node)), ",".join(map(str, xyz)), sep=";", file=f)


def load_vertex_dict(filename):
    vertex_dict = {}
    with open(filename, "r") as f:
        for line in f:
            node, xyz = line.strip().split(";")
            node = tuple(map(int, node.split(",")))
            xyz = np.asarray([float(x) for x in xyz.split(",")])
            vertex_dict[node] = xyz
    return vertex_dict


def main():
    grid = Pentagrid()
    grid.calculate_grid_edges((-10, 10))
    edges = [(grid.grid_nodes[n], grid.grid_nodes[m]) for n, m in grid.grid_edges]
    for e in edges:
        draw.draw_edge(*e)
    draw.draw_points([(x, y) for x, y, c in grid.grid_nodes.values()], color="red")
    draw.show()


if __name__ == "__main__":
    main()
    # draw_pentagrid()
