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
        theta = d * 2 * np.pi / 5
        lines.append([np.cos(theta), np.sin(theta), 0])
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
        self.grid_nodes: Dict[FrozenSet[Tuple[int, int]], np.ndarray] = {}
        self.grid_edges: Set[FrozenSet[FrozenSet[Tuple[int, int]]]] = set()

    def get_group_offset(self, group: int):
        return self._base_offset[group, -1]

    def get_line(self, group: int, index: float = 0):
        theta = 2*np.pi/5*group
        distance = index + self.get_group_offset(group)
        return np.matmul(trans.angular_translate(theta, distance), self._base_lines[group])

    def get_line_group(self, group: int, index_range: Tuple[int, int]):
        array = np.zeros([index_range[1] - index_range[0], 3])
        array[:, -1] = np.arange(index_range[0], index_range[1])
        return array + self._base_lines[group] + self._base_offset[group]

    def _get_line_group(self, group: int, index_range: Tuple[int, int]):
        # FIXME: BASE OFFSET IS BROKEN
        t = np.stack([trans.angular_translate(2*np.pi/5*group, i) for i in range(*index_range)])
        # line =
        a = np.matmul(self.get_line(group), t)
        return a

    def get_line_x(self, group: int, index: float, y: float, scale: float = 1):
        line = self.get_line(group, index)
        y = [0, -1, y]
        cross = intersection(line, y)
        if cross is not None:
            cross = cross[0] * scale
        return cross

    def get_line_y(self, group: int, index: float, x: float, scale: float = 1):
        line = self.get_line(group, index)
        x = [-1, 0, x]
        cross = intersection(line, x)
        if cross is not None:
            cross = cross[1] * scale
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
            trans_matrix = np.matmul(trans.angular_skew_y(iota),
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
            trans_matrix = np.matmul(trans.angular_skew_y(iota),
                                     trans.translate(self._base_offset[g1][2], self._base_offset[g2][2]))
            trans_matrix = np.matmul(trans.rotate(theta),
                                     trans_matrix)
            grid = np.matmul(trans_matrix,
                             base.transpose([0, 1, 3, 2]))
            points[g1, g2, :, :, :] = grid.transpose([0, 1, 3, 2])
        return points

    def annotate_intersections(self, points: np.ndarray, index_range: Tuple[int, int]):
        newshape = list(points.shape)
        newshape[-1] = 5
        coordinates = np.zeros(newshape)
        for g in range(self.GROUPS):
            lines = self.get_line_group(g, index_range)
            coordinates[..., g] = np.sum(np.matmul(points, lines.T) > 0, axis=-1, dtype=int)

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
    raise NotImplementedError
    with open(filename, "w") as f:
        for node, xyz in vertex_dict.items():
            print(",".join(map(str, node)), ",".join(map(str, xyz)), sep=";", file=f)


def load_vertex_dict(filename):
    raise NotImplementedError
    vertex_dict = {}
    with open(filename, "r") as f:
        for line in f:
            node, xyz = line.strip().split(";")
            node = tuple(map(int, node.split(",")))
            xyz = np.asarray([float(x) for x in xyz.split(",")])
            vertex_dict[node] = xyz
    return vertex_dict


def test():
    grid = Pentagrid()
    plot_grid(-3, 3, -300, 200, 300, -200, scale=6, grid=grid)
    plot_intersections(-3, 3, scale=6, grid=grid)


def plot_intersections(i1: int, i2: int,
                       scale: float = 1,
                       grid: Optional[Pentagrid] = None):
    if not grid:
        grid = Pentagrid()
    points = grid.calculate_intersections((i1, i2))
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
            x, y, z = points[(g1, g2, *index)] * scale
            draw.draw_point(x, y, color=color, size=6)
    draw.show()


def plot_grid(i1: int, i2: int,
              x1: float, y1: float,
              x2: float, y2: float,
              scale: float = 1,
              grid: Optional[Pentagrid] = None):
    # TODO: use matmul instead
    if not grid:
        grid = Pentagrid()
    top = np.asarray([0, -1, y1])
    bottom = np.asarray([0, -1, y2])
    left = np.asarray([1, 0, x1])
    right = np.asarray([1, 0, x2])
    for g in range(grid.GROUPS):
        lines = grid._get_line_group(g, (i1, i2))
        pts_top = np.cross(lines, top)
        pts_bottom = np.cross(lines, bottom)
        pts_left = np.cross(lines, left)
        pts_right = np.cross(lines, right)
        pts_top /= pts_top[:, [-1]]
        pts_bottom /= pts_bottom[:, [-1]]
        pts_left /= pts_left[:, [-1]]
        pts_right /= pts_right[:, [-1]]
        for i in range(i1, i2):
            lx1, ly1, lz1 = pts_top[i] * scale
            if np.isnan(lz1) or lx1 < x1:
                lx1, ly1, lz1 = pts_left[i] * scale
            elif lx1 > x2:
                lx1, ly1, lz1 = pts_right[i] * scale

            lx2, ly2, lz2 = pts_bottom[i] * scale
            if np.isnan(lz2) or lx2 > x2:
                lx2, ly2, lz2 = pts_right[i] * scale
            elif lx2 < x1:
                lx2, ly2, lz2 = pts_left[i] * scale

            draw.draw_line(lx1, -ly1, lx2, -ly2, color="white", width=3)
        # break
    draw.show()


def main():
    index_range = (-10, 10)
    grid = Pentagrid()
    points = grid.calculate_intersections(index_range)
    grid.annotate_intersections(points, index_range)



if __name__ == "__main__":
    # main()
    test()
    # plot_intersections(-100, 100, scale=3)
    # plot_grid(-20, 20, -300, 200, 300, -200, scale=3)
    # draw_pentagrid()
