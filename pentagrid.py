from __future__ import annotations

import heapq
import random
import sys
from typing import Optional, Union, Tuple, List, Sequence, Deque, Any, Iterable
from collections import OrderedDict, deque
import numpy as np
import itertools

import pil_draw_simple as draw


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


def intersection(l: np.ndarray, m: np.ndarray):
    cross = np.cross(l, m)
    if cross[2] == 0:
        return None
    return cross[:2] / cross[2]


class Pentagrid:
    """
    Five families of equidistant parallel lines.
    The angle between each two families is an integer multiple of 2*PI/5.
    No more than two lines may intersect at any given point.
    Each area delimited by the lines is equivalent to a node in penrose tiling.
    """
    DIMENSIONS = (0, 1, 2, 3, 4)

    def __init__(self):
        self._base_offset = np.asarray([[0, 0, np.sqrt(np.random.random() + 1)] for _ in range(5)])
        self._base_lines = _set_base_lines()
        self._base_intersections = _set_intersections(self._base_lines + self._base_offset)

    def get_line_norm(self, dimension: int, index: float):
        """
        Returns the normal vector of the specified line.
        :param dimension:
        :param index:
        :return:
        """
        return self._base_lines[dimension] + self._base_offset[dimension] - [0, 0, index]

    def get_line_x(self, dimension: int, index: float, y: float):
        line = self.get_line_norm(dimension, index)
        y = [0, -1, y]
        cross = intersection(line, y)
        if cross is not None:
            cross = cross[0]
        return cross

    def get_line_y(self, dimension: int, index: float, x: float):
        line = self.get_line_norm(dimension, index)
        x = [-1, 0, x]
        cross = intersection(line, x)
        if cross is not None:
            cross = cross[1]
        return cross

    def get_nodes(self, index_range: Tuple[int, int]):
        """
        5 dimensions
        5 x N parallel lines
        Each area delimited by the lines is equivalent to a node in penrose tiling.
        Equivalent to finding the overlap of two irregular equiangular pentagons.
        Well, sort of. We'd have to consider line orientation.
        """
        nodes = []
        coords = itertools.product(*[range(*index_range) for _ in range(len(self.DIMENSIONS))])
        for C in coords:
            x, y = 0, 0
            xmin, xmax = -np.infty, np.infty
            ymin, ymax = -np.infty, np.infty
            for dim, c in enumerate(C):
                x1 = np.cos(dim * 2 * np.pi / 5)
                x0 = np.sin(dim * 2 * np.pi / 5) * (c + self._base_offset[dim])
                y1 = np.sin(dim * 2 * np.pi / 5)
                y0 = -np.cos(dim * 2 * np.pi / 5) * (c + self._base_offset[dim])


def main():
    top = 20
    bottom = -20
    left = -30
    right = 30
    scale = 5
    grid = Pentagrid()
    lines = []
    for d in range(5):
        for i in range(2):
            x1 = grid.get_line_x(d, i, bottom)
            if x1 is None:
                x1 = left
            x2 = grid.get_line_x(d, i, top)
            if x2 is None:
                x2 = right
            # if x1 > x2:
            #     x1, x2 = x2, x1
            x1, x2 = max([x1, left]), min([x2, right])
            y1 = grid.get_line_y(d, i, x1)
            y2 = grid.get_line_y(d, i, x2)
            lines.append(np.asarray([x1, y1, x2, y2]) * scale)
    print(lines)
    points = []
    for x in grid._base_intersections.values():
        points.append(x * scale)
    print(points)
    draw.draw_lines(lines)
    draw.draw_points(points)
    draw.show()


if __name__ == "__main__":
    main()
