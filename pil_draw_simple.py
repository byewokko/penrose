import math
from typing import Iterable, Optional, Tuple, Union, Sequence
import numpy as np

from PIL import Image, ImageDraw

import transformations as trans

WIDTH, HEIGHT = 1280, 1024
SCALE = 50
POINT_SIZE = 5
LINE_WEIGHT = 2
POINT_COLOR = "red"
LINE_COLOR = "grey"
FILL_COLOR = "pink"
BG_COLOR = "black"

img = Image.new("RGB", (WIDTH, HEIGHT))
draw = ImageDraw.Draw(img)


class Draw:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT, scale: int = SCALE):
        self._img = Image.new("RGB", (width, height))
        self._draw = ImageDraw.Draw(self._img)
        self.width = width
        self.height = height
        self.scale = scale
        self.point_size = POINT_SIZE
        self.line_weight = LINE_WEIGHT
        self.point_color = POINT_COLOR
        self.line_color = LINE_COLOR
        self.fill_color = FILL_COLOR
        self.bg_color = BG_COLOR

    def normalize(self, *args: float):
        assert not len(args) % 2, f"The number of args must be even. Received: {len(args)}"

        def norm(p):
            i, a = p
            if not i % 2:
                return self.width / 2 + a * self.scale
            else:
                return self.height / 2 - a * self.scale

        return map(norm, enumerate(args))

    def normalize_matrix(self, array: np.ndarray):
        t = np.matmul(trans.translate(self.width / 2, self.height / 2), trans.scale(self.scale, -self.scale))
        array = np.matmul(array, t.T)
        print(array)
        return array / array[..., [-1]]

    def draw_line(self,
                  *args: Union[float, Tuple[float, float]],
                  color: Optional[str] = None,
                  width: Optional[int] = None):
        if len(args) == 4:
            x1, y1, x2, y2 = self.normalize(*args)
        elif len(args) == 2:
            x1, y1, x2, y2 = self.normalize(*args[0], *args[1])
        else:
            raise TypeError("The input must be 4 floats or 2 tuples of 2 floats.")
        self._draw.line([x1, y1, x2, y2],
                        width=width or self.line_weight,
                        fill=color or self.line_color)

    def draw_norm_line(self,
                       coords: Union[Sequence[float], np.ndarray],
                       color: Optional[str] = None,
                       width: Optional[int] = None):
        assert len(coords) == 3
        raise NotImplementedError
        self._draw.line([x1, y1, x2, y2],
                        width=width or self.line_weight,
                        fill=color or self.line_color)

    def draw_lines(self,
                   lines: Iterable,
                   color: Optional[str] = None,
                   width: Optional[int] = None):
        for line in lines:
            self.draw_line(*line, color=color, width=width)

    def draw_edge(self,
                  v1: tuple, v2: tuple,
                  color: Optional[str] = None,
                  width: Optional[int] = None):
        print("draw_edge is deprecated, use draw_line")
        x1, y1, x2, y2 = self.normalize(v1[0], v1[1], v2[0], v2[1])
        self._draw.line([x1, y1, x2, y2],
                        width=width or self.line_weight,
                        fill=color or self.line_color)

    def draw_edges(self,
                   lines: Iterable,
                   color: Optional[str] = None,
                   width: Optional[int] = None):
        print("draw_edges is deprecated, use draw_lines")
        for line in lines:
            self.draw_edge(*line, color=color, width=width)

    def draw_norm_point(self,
                        coords: Union[Sequence[float], np.ndarray],
                        *args, **kwargs):
        x, y, *_ = coords / coords[-1]
        self.draw_point(x, y, *args, **kwargs)

    def draw_point_array(self,
                         points: np.ndarray,
                         *args, **kwargs):
        points = self.normalize_matrix(points)
        for x, y, z in points:
            self.draw_point(x, y, *args, normalize=False, **kwargs)

    def draw_point(self,
                   x: float, y: float,
                   normalize: bool = True,
                   color: Optional[str] = None,
                   size: Optional[int] = None):
        size = size or self.point_size
        if normalize:
            x, y = self.normalize(x, y)
        self._draw.ellipse([x - size, y - size, x + size, y + size],
                           fill=color or self.point_color,
                           outline=None)

    def draw_points(self, points: Iterable, color: str = "red"):
        for point in points:
            self.draw_point(*point, color=color)

    def show(self):
        self._img.show()

    def clear(self):
        self._draw.rectangle([0, 0, self.width, self.height], fill="black")


def main():
    d = Draw()
    d.scale = 50
    d.draw_point(0, 0)
    d.draw_line(0, 5, 0, -2)
    arr = np.asarray(
        [[1,1,1],
         [-2,-4,2],
         [0,5,-1],
         [1,1,-1]]
    )
    d.draw_point_array(arr, color="blue")
    d.show()


if __name__ == "__main__":
    main()
