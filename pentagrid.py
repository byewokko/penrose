from __future__ import annotations

import heapq
from typing import Union, Tuple, Dict, Sequence
import numpy as np

from utils import transformations as trans
from drawing.pil_draw_simple import Draw


def _set_base_lines(offset: Union[np.ndarray, Sequence, None] = None):
    lines = []
    for d in range(5):
        theta = d * np.pi / 5
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


class Pentagrid:
    """
    Five families of equidistant parallel lines.
    The angle between each two families is an integer multiple of 2*PI/5.
    No more than two lines may intersect at any given point.
    Each area delimited by the lines is equivalent to a node in penrose tiling.

    Thi class uses homogeneous normal representation of lines and points.
    """
    GROUPS = 5

    def __init__(self, offset: Union[np.ndarray, Sequence, None] = None):
        if offset:
            self._base_offset = offset
        else:
            self._base_offset = np.asarray([np.sqrt(np.random.random() + 1) - 1 for _ in range(self.GROUPS)])
        self._base_lines = _set_base_lines(self._base_offset)

    def get_line(self, group: int, index: float = 0):
        theta = np.pi/5*group
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
        points = np.zeros([self.GROUPS,
                           self.GROUPS,
                           index_range[1] - index_range[0],
                           index_range[1] - index_range[0],
                           3])
        points.fill(np.nan)
        base = np.array(np.meshgrid(np.arange(*index_range), np.arange(*index_range), [1])).T
        for g1, g2 in triangle_iterator(self.GROUPS):
            iota = np.pi / 5 * (g2 - g1)
            theta = np.pi / 5 * g1
            trans_matrix = np.matmul(trans.angular_skew_y(iota),
                                     trans.translate(self._base_offset[g1], self._base_offset[g2]))
            trans_matrix = np.matmul(trans.rotate(theta),
                                     trans_matrix)
            grid = np.matmul(trans_matrix,
                             base.transpose([0, 1, 3, 2]))
            points[g1, g2, :, :, :] = grid.transpose([0, 1, 3, 2])
        return points

    def annotate_intersections(self, intersections: np.ndarray, index_range: Tuple[int, int]):
        """
        Returns ndarray of penta-coordinates of all the intersections.
        UPDATE!! the coordinates are not unique. some neighboring points share coords
        """
        newshape = list(intersections.shape)
        newshape[-1] = 5
        coordinates = np.zeros(newshape)
        for g in range(self.GROUPS):
            lines = self.get_line_group(g, index_range)
            coordinates[..., g] = np.sum(np.matmul(intersections, lines.T) < 0, axis=-1, dtype=int)
        coordinates += index_range[0]
        # For all g: Replace the g coordinate for all g-group points
        mesh = np.asarray(np.meshgrid(np.arange(*index_range), np.arange(*index_range)))
        for g1, g2 in triangle_iterator(self.GROUPS):
            coordinates[g1, g2, :, :, (g2, g1)] = mesh
        for g1, g2 in inverse_triangle_iterator(self.GROUPS):
            coordinates[g1, g2].fill(np.nan)
        indices = np.where(np.any(coordinates[:, :, :, :] >= np.nanmax(coordinates), axis=-1) |
            np.any(coordinates[:, :, :, :] <= np.nanmin(coordinates), axis=-1))
        coordinates[indices] = np.nan
        return coordinates


def intersections_to_edges(intersections: np.ndarray):
    edges: Dict[tuple, set] = {}
    shape = intersections.shape
    for g1 in range(shape[0]):
        for i1 in range(shape[2]):
            h = []
            for g2 in range(shape[1]):
                if g1 == g2:
                    continue
                for i2 in range(shape[3]):
                    if g1 < g2:
                        heapq.heappush(h, (tuple(intersections[g1, g2, i1, i2]), (g1, g2, i1, i2)))
                    else:
                        heapq.heappush(h, (tuple(intersections[g2, g1, i2, i1]), (g2, g1, i2, i1)))
            _, n2 = heapq.heappop(h)
            if n2 not in edges.keys():
                edges[n2] = set()
            while h:
                _, n1 = heapq.heappop(h)
                if n1 not in edges.keys():
                    edges[n1] = set()
                edges[n1].add(n2)
                edges[n2].add(n1)
                n2 = n1
    return edges


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
            draw.point(x, y, color=color)


def plot_grid(grid: Pentagrid,
              draw: Draw,
              i1: int, i2: int):
    for g in range(grid.GROUPS):
        lines = grid.get_line_group(g, (i1, i2))
        for line in lines:
            draw.norm_line(line)


def _find_duplicates(penta_points):
    first = {}
    second = {}
    third = {}
    fourth = {}
    fifth = {}
    shape = penta_points.shape
    for g1, g2 in triangle_iterator(5):
        for i1, i2 in np.ndindex(shape[2:-1]):
            k = tuple(penta_points[g1, g2, i1, i2])
            v = (g1, g2, i1, i2)
            if k in fifth:
                print(k)
            elif k in fourth:
                fifth[k] = v
            elif k in third:
                fourth[k] = v
            elif k in second:
                third[k] = v
            elif k in first:
                second[k] = v
            else:
                first[k] = v
    return first, second, third, fourth, fifth


def _show_duplicates():
    index_range = (-4, 4)
    grid = Pentagrid()
    draw = Draw(scale=80)
    xy_points = grid.calculate_intersections(index_range)
    penta_points = grid.annotate_intersections(xy_points, index_range)
    plot_grid(grid, draw, -4, 4)
    first, second, third, fourth, fifth = _find_duplicates(penta_points)
    keys = []
    for k in fifth.keys():
        print(k)
        keys.append(k)
        if len(keys) > 2:
            break
    first_array = xy_points[tuple(zip(*[first[k] for k in keys]))]
    second_array = xy_points[tuple(zip(*[second[k] for k in keys]))]
    third_array = xy_points[tuple(zip(*[third[k] for k in keys]))]
    fourth_array = xy_points[tuple(zip(*[fourth[k] for k in keys]))]
    fifth_array = xy_points[tuple(zip(*[fifth[k] for k in keys]))]
    # first_array = xy_points[tuple(zip(*first.keys()))]
    # second_array = xy_points[tuple(zip(*second.keys()))]
    draw.point_array(first_array, color="blue")
    draw.point_array(second_array, color="green")
    draw.point_array(third_array, color="yellow")
    draw.point_array(fourth_array, color="red")
    draw.point_array(fifth_array, color="orange")
    # xy_points = xy_points.reshape([-1, 3])
    # penta_points = penta_points.reshape([-1, 5])
    # indices = np.where(penta_points[:, 2] == 2)
    # strip = np.squeeze(xy_points[indices, :])
    # draw.point_array(strip)
    draw.show()


def main():
    index_range = (-2, 2)
    grid = Pentagrid()
    draw = Draw(scale=80)
    xy_points = grid.calculate_intersections(index_range)
    edges = intersections_to_edges(xy_points)
    ln = {i: 0 for i in range(12)}
    for e in edges.values():
        ln[len(e)] += 1
    print(ln)
    plot_grid(grid, draw, *index_range)
    for k, v in edges.items():
        if len(v) == 3:
            draw.norm_point(xy_points[k])
    for k, v in edges.items():
        if len(v) == 4:
            draw.norm_point(xy_points[k], color="green")
    # plot_intersections(grid, draw, *index_range)
    draw.show()
    return


def generate_edges(index_range: Tuple[int, int], offset: Union[np.ndarray, Sequence, None] = None):
    grid = Pentagrid(offset)
    intersections = grid.calculate_intersections(index_range)
    return intersections_to_edges(intersections)


if __name__ == "__main__":
    # show_duplicates()
    main()
    # test()
    # plot_intersections(Pentagrid(), Draw(scale=30), -100, 100)
    # plot_grid(-20, 20, -300, 200, 300, -200, scale=3)
