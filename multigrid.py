from __future__ import annotations

import heapq
from typing import Union, Tuple, Dict, Sequence
import numpy as np

from utils import transformations as trans
from drawing.pil_draw_simple import Draw

import logging


L = logging.getLogger(__name__)
L.setLevel(logging.DEBUG)
L.addHandler(logging.StreamHandler())


def _set_base_lines(n: int, offset: Union[np.ndarray, Sequence, None] = None):
    lines = []
    for d in range(n):
        theta = d * np.pi / n
        # works, but I'm not sure about the minuses
        lines.append([-np.cos(theta), -np.sin(theta), offset[d]])
    return np.asarray(lines)


def intersection(l: Union[np.ndarray, list], m: Union[np.ndarray, list]):
    cross = np.cross(l, m)
    if cross[2] == 0:
        return None
    return cross / cross[2]


def triangle_iterator(n: int):
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield i, j


def inverse_triangle_iterator(n: int):
    for i in range(n - 1):
        for j in range(i, n):
            yield j, i


class Multigrid:
    """
    Abstraction of the pentagrid, for any reasonable integer.
    """

    def __init__(self, n: int, offset: Union[np.ndarray, Sequence, None] = None):
        self.N = n
        if offset:
            self._base_offset = offset
        else:
            self._base_offset = np.asarray([np.sqrt(np.random.random() + 1) - 1 for _ in range(self.N)])
        print(self._base_offset)
        self._base_lines = _set_base_lines(self.N, self._base_offset)

    def get_line(self, group: int, index: float = 0):
        theta = np.pi / self.N * group
        distance = index
        return np.matmul(trans.angular_translate(theta, distance), self._base_lines[group])

    def get_line_group(self, group: int, index_range: Tuple[int, int]):
        a = np.repeat(self._base_lines[group][np.newaxis, :],
                      index_range[1] - index_range[0],
                      axis=0)
        a[:, -1] += np.arange(*index_range)
        return a

    def calculate_intersections(self, index_range: Tuple[int, int]):
        """
        Computes all the intersections in a given section of the pentagrid.
        Returns np.ndarray with shape [5, 5, index_range_size, index_range_size, 3].
        The first two dimensions form a triangular matrix without diagonal.
        """
        points = np.zeros([self.N,
                           self.N,
                           index_range[1] - index_range[0],
                           index_range[1] - index_range[0],
                           3])
        points.fill(np.nan)
        base = np.array(np.meshgrid(np.arange(*index_range), np.arange(*index_range), [1])).T
        for g1, g2 in triangle_iterator(self.N):
            iota = np.pi / self.N * (g2 - g1)
            theta = np.pi / self.N * g1
            trans_matrix = np.matmul(trans.angular_skew_y(iota),
                                     trans.translate(self._base_offset[g1], self._base_offset[g2]))
            trans_matrix = np.matmul(trans.rotate(theta),
                                     trans_matrix)
            grid = np.matmul(trans_matrix,
                             base.transpose([0, 1, 3, 2]))
            points[g1, g2, :, :, :] = grid.transpose([0, 1, 3, 2])
        return points


class Pentagrid(Multigrid):
    """
    Five families of equidistant parallel lines.
    The angle between each two families is an integer multiple of 2*PI/5.
    No more than two lines may intersect at any given point.
    Each area delimited by the lines is equivalent to a node in penrose tiling.

    Thi class uses homogeneous normal representation of lines and points.
    """

    def __init__(self, offset: Union[np.ndarray, Sequence, None] = None):
        super().__init__(5, offset)


def intersections_to_edges_dict(intersections: np.ndarray, index_range: tuple):
    edges: Dict[tuple, dict] = {}
    yx_intersections = np.zeros_like(intersections)
    yx_intersections[..., [0, 1]] = intersections[..., [1, 0]]
    shape = yx_intersections.shape
    for g1 in range(shape[0]):
        if g1 > shape[0] / 2:
            order = -1
        else:
            order = 1
        for i1 in range(shape[2]):
            h = []
            for g2 in range(shape[1]):
                if g1 == g2:
                    continue
                for i2 in range(shape[3]):
                    if g1 < g2:
                        heapq.heappush(
                            h, (tuple(order * yx_intersections[g1, g2, i1, i2]),
                                (g1, g2, i1 + index_range[0], i2 + index_range[0])))
                    else:
                        heapq.heappush(
                            h, (tuple(order * yx_intersections[g2, g1, i2, i1]),
                                (g2, g1, i2 + index_range[0], i1 + index_range[0])))
            _, n2 = heapq.heappop(h)
            if n2 not in edges.keys():
                edges[n2] = {}
            while h:
                _, n1 = heapq.heappop(h)
                if n1 not in edges.keys():
                    edges[n1] = {}
                edges[n1][g1] = n2
                edges[n2][(g1 + shape[0]) % (shape[0] * 2)] = n1
                n2 = n1
    return edges


def plot_intersections(grid: Multigrid,
                       draw: Draw,
                       i1: int, i2: int):
    points = grid.calculate_intersections((i1, i2))
    for g1, g2 in triangle_iterator(grid.N):
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
            x, y, z = points[(g1, g2, *index)]
            draw.point(x, y, color=color)


def plot_grid(grid: Multigrid,
              draw: Draw,
              i1: int, i2: int):
    for g in range(grid.N):
        lines = grid.get_line_group(g, (i1, i2))
        for line in lines:
            draw.norm_line(line)


def example():
    draw = Draw(scale=80)
    grid = Multigrid(7)
    index_range = (-2, 2)
    plot_grid(grid, draw, *index_range)
    plot_intersections(grid, draw, *index_range)
    draw.show()


if __name__ == "__main__":
    example()
