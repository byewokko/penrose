from __future__ import annotations

from typing import Union, Tuple
import numpy as np

from utils import transformations as trans
from drawing.pil_draw_simple import Draw


def _set_base_lines():
    lines = []
    for d in range(5):
        theta = d * 2 * np.pi / 5
        # works, but I'm not sure about the minuses
        lines.append([-np.cos(theta), -np.sin(theta), np.sqrt(np.random.random() + 1) - 1])
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
        self._base_lines = _set_base_lines()
        self._base_offset = self._base_lines[:, -1]

    def get_line(self, group: int, index: float = 0):
        theta = 2*np.pi/5*group
        distance = index
        return np.matmul(trans.angular_translate(theta, distance), self._base_lines[group])

    def get_line_group(self, group: int, index_range: Tuple[int, int]):
        a = np.repeat(self.get_line(group)[np.newaxis, :],
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
                                     trans.translate(self._base_offset[g1], self._base_offset[g2]))
            trans_matrix = np.matmul(trans.rotate(theta),
                                     trans_matrix)
            grid = np.matmul(trans_matrix,
                             base.transpose([0, 1, 3, 2]))
            points[g1, g2, :, :, :] = grid.transpose([0, 1, 3, 2])
        return points

    def annotate_intersections(self, points: np.ndarray, index_range: Tuple[int, int]):
        # TODO: this next!
        raise NotImplementedError
        newshape = list(points.shape)
        newshape[-1] = 5
        coordinates = np.zeros(newshape)
        for g in range(self.GROUPS):
            lines = self.get_line_group(g, index_range)
            coordinates[..., g] = np.sum(np.matmul(points, lines.T) > 0, axis=-1, dtype=int)


def test():
    grid = Pentagrid()
    draw = Draw(scale=80)
    draw.draw_line(-1, 0, 1, 0)
    draw.draw_line(0, -1, 0, 1)
    plot_grid(grid, draw, -3, 3, -300, 200, 300, -200)
    plot_intersections(grid, draw, -3, 3)


def plot_intersections(grid: Pentagrid,
                       draw: Draw,
                       i1: int, i2: int):
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
            x, y, z = points[(g1, g2, *index)]
            draw.draw_point(x, y, color=color)
    draw.show()


def plot_grid(grid: Pentagrid,
              draw: Draw,
              i1: int, i2: int):
    for g in range(grid.GROUPS):
        lines = grid.get_line_group(g, (i1, i2))
        for line in lines:
            draw.draw_norm_line(line)
    draw.show()


def main():
    index_range = (-10, 10)
    grid = Pentagrid()
    points = grid.calculate_intersections(index_range)
    grid.annotate_intersections(points, index_range)


if __name__ == "__main__":
    # main()
    test()
    # plot_intersections(Pentagrid(), Draw(scale=30), -100, 100)
    # plot_grid(-20, 20, -300, 200, 300, -200, scale=3)
