from __future__ import annotations
from typing import Optional, Union, Tuple, List, Sequence, Deque

import numpy as np


class Point:
    def __init__(self, coords: Sequence[int]):
        self._p = tuple(coords)
        self.length = len(self._p)

    def as_tuple(self):
        return tuple(self._p)

    def __repr__(self):
        return f"Node{str(self._p)}"

    def __iter__(self):
        for p in self._p:
            yield p

    def __hash__(self):
        return hash(self.as_tuple())

    def __neg__(self):
        return type(self)([-x for x in self._p])

    def __add__(self, other: Point):
        assert self.length == other.length
        return type(self)([self._p[i] + other._p[i] for i in range(self.length)])

    def __sub__(self, other: Point):
        return self + (-other)

    def __rmul__(self, other: int):
        return type(self)([other * x for x in self._p])

    def __mul__(self, other: int):
        return type(self)([other * x for x in self._p])

    def __eq__(self, other: Point):
        assert self.length == other.length
        return self._p == other._p

    def __gt__(self, other: Point):
        assert self.length == other.length
        return self._p > other._p

    def __ge__(self, other: Point):
        assert self.length == other.length
        return self._p >= other._p

    def __lt__(self, other: Point):
        assert self.length == other.length
        return self._p < other._p

    def __le__(self, other: Point):
        assert self.length == other.length
        return self._p <= other._p

    def __ne__(self, other: Point):
        assert self.length == other.length
        return not (self._p == other._p)


class Vertex(Point):
    def __init__(self, bag: VertexBag, coords: Sequence[int]):
        super().__init__(coords)
        self._bag = bag

    def step(self, direction: int):
        polarity, direction = divmod(direction, self._bag.n_dims)
        coords = list(self)
        if polarity % 2:
            coords[direction] -= 1
        else:
            coords[direction] += 1
        return self._bag[coords]

    def get_xy(self, edge_length: float = 1.0, homogenous: bool = True):
        x = edge_length * sum([p * np.cos(np.pi / self.length * i) for i, p in enumerate(self._p)])
        y = edge_length * sum([p * np.sin(np.pi / self.length * i) for i, p in enumerate(self._p)])
        if homogenous:
            return np.asarray([x, y, 1])
        return np.asarray([x, y])


class VertexBag:
    def __init__(self, n_dims: int):
        self._dict = dict()
        self.n_dims = n_dims

    def __getitem__(self, item):
        assert len(item) == self.n_dims
        if isinstance(item, Vertex):
            item = item.as_tuple()
        if tuple(item) in self.keys():
            return self[tuple(item)]
        vertex = Vertex(self, item)
        self._dict[tuple(item)] = vertex
        return vertex

    def keys(self):
        return self._dict.keys()

    def as_nparray(self):
        return np.stack(self.keys())

    def get_xy(self, edge_length):
        transmat = np.asarray(
            [(np.cos(np.pi / self.n_dims * i), np.sin(np.pi / self.n_dims * i)) for i in range(self.n_dims)]
        ) * edge_length
        return np.matmul(self.as_nparray(), transmat)


class EdgeBag:
    def __init__(self):
        self._set = set()

    def exists(self, a: Vertex, b: Vertex):
        if a > b:
            return (b, a) in self._set
        else:
            return (a, b) in self._set

    def add(self, a: Vertex, b: Vertex):
        if a > b:
            self._set.add((b, a))
        else:
            self._set.add((a, b))

    def remove(self, a: Vertex, b: Vertex):
        if a > b:
            self._set.remove((b, a))
        else:
            self._set.remove((a, b))
