from __future__ import annotations

import heapq
import random
import sys
from typing import Optional, Union, Tuple, List, Sequence, Deque, Any, Iterable, Dict, FrozenSet, Set
from collections import OrderedDict, deque
import numpy as np
import itertools

import transformations as trans
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


def triangle_iterator(n: int):
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield i, j


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

    def get_line_group(self, group: int, index_range: Tuple[int, int]):
        array = np.zeros([index_range[1] - index_range[0], 3])
        array[:, -1] = np.arange(index_range[0], index_range[1])
        return array + self._base_lines[group] + self._base_offset[group]

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

    def calculate_intersections_as_dict(self, index_range: Tuple[int, int]):
        intersections = FrozenSetDict()
        base_grid = np.array(
            np.meshgrid(np.arange(*index_range), np.arange(*index_range), [1])).T.reshape(-1, 3).T
        for g1, g2 in triangle_iterator(self.GROUPS):
            # g1 corresponds to x
            # g2 corresponds to y
            iota = 2 * np.pi / 5 * (g2 - g1)
            theta = 2 * np.pi / 5 * g1
            trans_matrix = np.matmul(trans.skew_rot_y(iota),
                                     trans.translate(self._base_offset[g1][2], self._base_offset[g2][2]))
            trans_matrix = np.matmul(trans.rotate(theta),
                                     trans_matrix)
            grid = np.matmul(trans_matrix,
                             base_grid)
            intersections[(g1, g2)] = grid.T

        return base_grid, intersections

    def calculate_intersections(self, index_range: Tuple[int, int]):
        """
        Computes all the intersections in a given section of the pentagrid.
        Returns np.ndarray with shape [5, 5, index_range_size, index_range_size, 3].
        The first two dimensions form a triangular matrix without diagonal.
        :param index_range:
        :return:
        """
        points = np.zeros([self.GROUPS,
                           self.GROUPS,
                           index_range[1] - index_range[0],
                           index_range[1] - index_range[0],
                           3])
        base = np.array(np.meshgrid(np.arange(*index_range), np.arange(*index_range), [1])).T
        for g1, g2 in triangle_iterator(self.GROUPS):
            iota = 2 * np.pi / 5 * (g2 - g1)
            theta = 2 * np.pi / 5 * g1
            trans_matrix = np.matmul(trans.skew_rot_y(iota),
                                     trans.translate(self._base_offset[g1][2], self._base_offset[g2][2]))
            trans_matrix = np.matmul(trans.rotate(theta),
                                     trans_matrix)
            grid = np.matmul(trans_matrix,
                             base.transpose([0, 1, 3, 2]))
            points[g1, g2, :, :, :] = grid.transpose([0, 1, 3, 2])
        return points

    def annotate_intersections(self, base_grid, intersections):
        for groups, points in intersections.items():
            for g in range(self.GROUPS):
                if g in groups:
                    # different indexing
                    continue
                # TODO need index_range here !
                lines = self.get_line_group(g, index_range)
                positions = np.matmul(lines, points.T) > 0
                indices = np.sum(positions, axis=1)  # ?which axis
                # store indices

    def calculate_grid_edges(self, index_range: Tuple[int, int]):
        """
        For each line:
        - get instersections with all the other lines
        - put them in heapq cast as tuples (ndarray cannot __lt__)
        - generate edges
        """
        for g1 in range(self.GROUPS - 1):
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
        grid = np.meshgrid(np.arange(*index_range), np.arange(*index_range), [1])

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


def test(theta=0):
    grid = np.array(np.meshgrid(np.arange(10), np.arange(10), [1])).T.reshape(-1, 3).T * 3
    for x, y, z in grid.T:
        draw.draw_point(x, y, "gray", 10)
    theta = 2 * np.pi / 5 * 2  # - np.pi / 2
    grid_rot = np.matmul(trans.rotate(theta - np.pi / 2), grid)
    for x, y, z in grid_rot.T:
        draw.draw_point(x, y, "blue", 10)
    grid_skewed = np.matmul(trans.skew_rot_y(theta), grid)
    for x, y, z in grid_skewed.T:
        draw.draw_point(x, y, "red", 10)
    draw.show()
    return grid.T, grid_rot.T, grid_skewed.T


def plot_grid():
    grid = Pentagrid()
    points = grid.calculate_intersections((-20, 20))
    for g1, g2 in triangle_iterator(grid.GROUPS):
        if 0 in (g1, g2) or 1 in (g1, g2):
            if 0 in (g1, g2) and 1 in (g1, g2):
                # continue
                color = "white"
            else:
                # continue
                color = "blue"
        else:
            # continue
            color = "red"
        for index in np.ndindex(points.shape[2:-1]):
            x, y, z = points[(g1, g2, *index)] * 3
            draw.draw_point(x, y, color=color, size=6)
    draw.show()


def main():
    grid = Pentagrid()
    base_grid, ints = grid.calculate_intersections((-100, 100))
    for key, val in ints.items():
        if 0 in key or 1 in key:
            if 0 in key and 1 in key:
                color = "white"
            else:
                color = "black"
        else:
            color = "red"
        for x, y, z in val * 3:
            draw.draw_point(x, y, color=color, size=6)
    draw.show()


if __name__ == "__main__":
    # main()
    # test()
    plot_grid()
    # draw_pentagrid()
