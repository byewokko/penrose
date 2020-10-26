from __future__ import annotations

import heapq
import random
import sys
from typing import Optional, Union, Tuple, List, Sequence, Deque, Any
from collections import OrderedDict, deque
import numpy as np


class Pentagrid:
    """
    Five families of equidistant parallel lines.
    The angle between each two families is an integer multiple of 2*PI/5.
    No more than two lines may intersect at any given point.
    """
    LINE_SPACING = 1.0
    LINE_FAMILIES = (0, 1, 2, 3, 4)

    def __init__(self):
        self.base_offset = [np.sqrt(np.random.random() + 1) * self.LINE_SPACING for _ in range(5)]
        # self.base_offset = [0 for _ in range(len(self.LINE_FAMILIES))]

    def get_node_given_p(self, p: float, line_index: int, family: Union[0, 1, 2, 3, 4]):
        x = np.cos(family * 2 * np.pi / 5) * p + \
            np.sin(family * 2 * np.pi / 5) * (line_index * self.LINE_SPACING + self.base_offset[family])
        y = np.sin(family * 2 * np.pi / 5) * p + \
            -np.cos(family * 2 * np.pi / 5) * (line_index * self.LINE_SPACING + self.base_offset[family])
        return x, y
