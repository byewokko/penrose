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


def intersection(l: Union[np.ndarray, list], m: Union[np.ndarray, list], mode: str = "2d"):
    cross = np.cross(l, m)
    if cross[2] == 0:
        return None
    if mode == "3d":
        return cross / cross[2]
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
        return self._base_lines[dimension] + self._base_offset[dimension] + [0, 0, index]

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

    def get_intersection(self, a_dim: int, a_ind: float, b_dim: int, b_ind: float):
        return (self._base_intersections[frozenset([a_dim, b_dim])]
                + a_ind * self._base_lines[a_dim][:2]
                + b_ind * self._base_lines[b_dim][:2])

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


def draw_pentagrid():
    grid_range = range(-10, 10)
    box = (-30, 20, 30, -20)
    scale = 3
    grid = Pentagrid()
    lines = generate_grid_lines(grid, grid_range, box) * scale
    points = calculate_intersection_points(grid, grid_range) * scale

    # print(lines)
    # print(points)
    draw.draw_lines(lines)
    draw.draw_points(points)
    draw.show()


def calculate_intersection_points(grid, grid_range):
    # TODO: use matrices and meshgrid?
    intersections = []
    for i in grid_range:
        print(f"\rCalculating pentagrid nodes: {i}/{grid_range}", end="", file=sys.stderr)
        for j in grid_range:
            for a_dim, b_dim in grid._base_intersections.keys():
                a = grid.get_line_norm(a_dim, i)
                b = grid.get_line_norm(b_dim, j)
                point = intersection(a, b)
                intersections.append(point)
    return np.asarray(intersections)


def calculate_intersection_dict(grid, grid_range):
    intersections = {}
    for i in grid_range:
        print(f"\rCalculating pentagrid nodes: {i}/{grid_range}", end="", file=sys.stderr)
        for j in grid_range:
            for a_dim, b_dim in grid._base_intersections.keys():
                a = grid.get_line_norm(a_dim, i)
                b = grid.get_line_norm(b_dim, j)
                point = intersection(a, b, "3d")
                intersections[frozenset([(a_dim, i), (b_dim, j)])] = point
    return intersections


def generate_grid_lines(grid, grid_range, box):
    # TODO: use matrices?
    left, top, right, bottom = box
    lines = []
    for i in grid_range:
        for d in range(5):
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
            lines.append([x1, y1, x2, y2])
    return np.asarray(lines)


def calculate_midpoints(grid, intersections, grid_range):
    midpoints = {}
    coordinates = itertools.product(*[grid_range for _ in range(5)])
    for n, field in enumerate(coordinates):
        print(f"\rCalculating pentagrid midpoints: {n+1}/{len(grid_range)**5}", end="", file=sys.stderr)
        points = set(
            itertools.combinations([(d, i) for d, i in enumerate(field)] + [(d, i+1) for d, i in enumerate(field)], r=2)
        )
        for d in range(5):
            active = set()
            for point in points:
                if point[0][0] == point[1][0]:
                    continue
                if point[0][0] == d or point[1][0] == d:
                    active.add(point)
                    continue  # skip points ON the line
                xy = intersections[frozenset(point)]
                a = grid.get_line_norm(d, field[d])
                b = grid.get_line_norm(d, field[d] + 1)
                if (np.dot(a, xy) > 0) == (np.dot(b, xy) < 0):
                    active.add(point)
            points = active
        if points:
            midpoints[field] = np.mean([intersections[frozenset(point)] for point in points], axis=0)
    print("", file=sys.stderr)
    return midpoints


def connect_midpoints(vertex_dict):
    edges = set()
    for vertex in vertex_dict.keys():
        for i in range(len(vertex)):
            plus = vertex[:i] + (vertex[i] + 1,) + vertex[i+1:]
            minus = vertex[:i] + (vertex[i] - 1,) + vertex[i+1:]
            if plus in vertex_dict:
                edges.add(frozenset((vertex, plus)))
            if minus in vertex_dict:
                edges.add(frozenset((vertex, minus)))
    return edges


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
    scale = 15
    box = (-30, 20, 30, -20)
    size = 3
    grid = Pentagrid()
    lines = generate_grid_lines(grid, range(-size, size), box) * scale
    intersections = calculate_intersection_dict(grid, range(-size, size + 1))
    vertex_dict = calculate_midpoints(grid, intersections, range(-size, size))
    save_vertex_dict(vertex_dict, f"vertexdict_{size}.txt")
    vertices = np.asarray(list(vertex_dict.values()))[:, :2] * scale
    edges = connect_midpoints(vertex_dict)
    edges_xy = np.asarray([(vertex_dict[v1][:2], vertex_dict[v2][:2]) for v1, v2 in edges]) * scale

    # print(lines)
    # print(points)
    # draw.draw_lines(lines, color="blue")
    draw.draw_points(vertices)
    draw.draw_edges(edges_xy, color="white")
    draw.show()


if __name__ == "__main__":
    main()
    # draw_pentagrid()
