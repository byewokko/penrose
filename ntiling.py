from __future__ import annotations

import heapq
import logging
from typing import Optional, Union, Tuple, List, Sequence, Any, Literal, Dict
import numpy as np

from ndimgraph import Vertex, VertexBag, EdgeBag
import multigrid
from drawing.pil_draw_simple import Draw

L = logging.getLogger(__name__)
L.setLevel(logging.DEBUG)
L.addHandler(logging.StreamHandler())


class Rhomb:
    def __init__(self, n: int, edges: Dict[int, tuple], grid_node: Optional[tuple] = None):
        self.n = n
        self.edges = edges
        self.node = grid_node

    def type(self):
        return (max(self.edges.keys()) - min(self.edges.keys())) % self.n

    def get_vertices(self) -> Tuple[Vertex, Vertex, Vertex, Vertex]:
        direction = next(iter(self.edges.keys()))
        a, b = self.edges[direction]
        c, d = self.edges[(direction + self.n) % (2 * self.n)]
        return a, b, c, d

    def get_edges(self):
        return self.edges.values()

    def xy(self, edge_length: float = 1., form: Union[Literal["xy"], Literal["xy1"]] = "xy"):
        return [n.get_xy(edge_length, form=form) for n in self.get_vertices()]

    def center(self, edge_length: float = 1., form: Union[Literal["xy"], Literal["xy1"]] = "xy"):
        a, b, c, d = self.xy(edge_length=edge_length, form=form)
        return (a + c) / 2



class TilingBuilder:
    def __init__(self, grid: multigrid.Multigrid):
        self._grid = grid
        self._n = grid.N
        self._grid_edges = None
        # self._vertices: Dict[Tuple[int, int, int, int], Vertex] = {}
        self._rhombs: Dict[Tuple[int, int, int, int], Rhomb] = {}
        self._vertices = VertexBag(grid.N)
        self._edges = {}
        self._frontier = []
        self._frontier_counter = 0
        self._expanded_nodes = set()

    def prepare_grid(self, index_range: Tuple[int, int]):
        L.debug("Calculating pentagrid edges")
        grid_nodes = self._grid.calculate_intersections(index_range)
        self._grid_edges = multigrid.intersections_to_edges_dict(grid_nodes, index_range)
        L.info("Pentagrid edges ready")

    def generate_rhombs(self,
                        start_node: Tuple[int, int, int, int] = (0, 1, 0, 0),
                        n_rhombs: int = None):
        assert self._grid_edges
        next_alert = 0
        self.frontier_push(start_node)
        self.add_rhomb(start_node, self.build_start_rhomb(start_node[:2]))
        yield self._rhombs[start_node]
        while True:
            if self._frontier_counter > next_alert:
                L.debug(f"{len(self._frontier)} nodes in frontier")
                L.debug(f"{len(self._rhombs)} rhombs built")
                next_alert += 100
            # Stopping conditions
            if n_rhombs and len(self._rhombs) >= n_rhombs:
                break
            grid_node = self.frontier_pop()
            if not grid_node:
                break
            # Add adjacents to frontier and build them
            for direction, adjacent in self._grid_edges[grid_node].items():
                if adjacent not in self._expanded_nodes:
                    self._expanded_nodes.add(adjacent)
                    self.frontier_push(adjacent)
                    self._rhombs[adjacent] = self.build_adjacent_rhomb(grid_node, direction, adjacent)
                    yield self._rhombs[adjacent]

    def generate_rhomb_list(self,
                            start_node: Tuple[int, int, int, int] = (0, 1, 0, 0),
                            n_rhombs: int = None):
        assert self._grid_edges
        next_alert = 0
        self.frontier_push(start_node)
        self.add_rhomb(start_node, self.build_start_rhomb(start_node[:2]))
        # yield self._rhombs[start_node]
        while True:
            if self._frontier_counter > next_alert:
                L.debug(f"{len(self._frontier)} nodes in frontier")
                L.debug(f"{len(self._rhombs)} rhombs built")
                next_alert += 200
            # Stopping conditions
            if n_rhombs and len(self._rhombs) >= n_rhombs:
                break
            grid_node = self.frontier_pop()
            if not grid_node:
                break
            # Add adjacents to frontier and build them
            for direction, adjacent in self._grid_edges[grid_node].items():
                if adjacent not in self._expanded_nodes:
                    self._expanded_nodes.add(adjacent)
                    self.frontier_push(adjacent)
                    self._rhombs[adjacent] = self.build_adjacent_rhomb(grid_node, direction, adjacent)
                    # yield self._rhombs[adjacent]

    def build_adjacent_rhomb(self, this_node: tuple, this_direction: int, adjacent_node: tuple):
        this_rhomb = self._rhombs[this_node]
        shared_edge = this_rhomb.edges[this_direction]
        cd_direction = this_direction
        ab_direction = (cd_direction + self._n) % (2 * self._n)
        if adjacent_node[0] == ab_direction % self._n:
            cross_direction = adjacent_node[1]
        elif adjacent_node[1] == ab_direction % self._n:
            cross_direction = adjacent_node[0]
        else:
            raise ValueError(f"Direction {ab_direction} not found in {adjacent_node}")

        # The edges in a rhomb must follow in ascending order, with steps no larger than 4
        if (ab_direction < cross_direction < ab_direction + self._n
                or not (ab_direction - self._n < cross_direction < ab_direction)):
            bc_direction = cross_direction
            da_direction = (cross_direction + self._n) % (2 * self._n)
        else:
            da_direction = cross_direction
            bc_direction = (cross_direction + self._n) % (2 * self._n)

        # invert the edge
        b, a = shared_edge
        c = b.step(bc_direction)
        d = a.step(bc_direction)
        ab = (a, b)
        bc = (b, c)
        cd = (c, d)
        da = (d, a)
        rhomb_edges = {
            ab_direction: ab,
            bc_direction: bc,
            cd_direction: cd,
            da_direction: da
        }
        return Rhomb(self._n, rhomb_edges, adjacent_node)

    def frontier_push(self, grid_node: tuple):
        # check if the node has enough neighbors
        if len(self._grid_edges[grid_node]) < 4:
            L.debug(f"Node {grid_node} has only {len(self._grid_edges[grid_node])} neighbors. Skipping.")
            return
        item = (self._frontier_counter, grid_node)
        heapq.heappush(self._frontier, item)
        self._frontier_counter += 1

    def frontier_pop(self):
        if not self._frontier:
            return None
        _, grid_node = heapq.heappop(self._frontier)
        return grid_node

    def build_start_rhomb(self,
                          node: tuple,
                          start_vertex: Optional[Vertex] = None):
        if not start_vertex:
            # This is the lower left vertex of the rhomb
            start_vertex = self._vertices[[0 for _ in range(self._n)]]

        ab_direction, cross_direction = node
        # The edges in a rhomb must follow in ascending order, with steps no larger than 4
        if (ab_direction < cross_direction < ab_direction + self._n
                or not (ab_direction - self._n < cross_direction < ab_direction)):
            bc_direction = cross_direction
        else:
            bc_direction = (cross_direction + self._n) % (2 * self._n)
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
            (ab_direction + self._n) % (2 * self._n): cd,
            (bc_direction + self._n) % (2 * self._n): da
        }
        return Rhomb(self._n, rhomb_edges, node)

    def add_rhomb(self, grid_node: tuple, rhomb: Rhomb):
        self._rhombs[grid_node] = rhomb


def main():
    palette = [
        "#293462",
        "#216583",
        "#00818a",
        "#f7be16",
    ]
    draw = Draw(scale=70, width=4*1280, height=4*1280, bg_color=palette[-2])
    draw.line_color = None
    index_range = (-2, 2)
    grid = multigrid.Multigrid(25)
    tiling_builder = TilingBuilder(grid)
    tiling_builder.prepare_grid(index_range)
    tiling_builder.generate_rhomb_list()

    for rhomb in tiling_builder._rhombs.values():
        for a, b in rhomb.get_edges():
            draw.edge(a.get_xy(homogenous=False), b.get_xy(homogenous=False), color=palette[-1], width=3)

    draw.show()


if __name__ == "__main__":
    main()
