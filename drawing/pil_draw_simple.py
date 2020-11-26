from typing import Iterable, Optional, Tuple, Union, Sequence
import numpy as np

from PIL import Image, ImageDraw

from utils import transformations as trans

WIDTH, HEIGHT = 1280, 1024
SCALE = 20
POINT_SIZE = 3
LINE_WEIGHT = 1
POINT_COLOR = "red"
LINE_COLOR = "grey"
FILL_COLOR = "pink"
BG_COLOR = "black"


class Draw:
    def __init__(self,
                 width: int = WIDTH,
                 height: int = HEIGHT,
                 scale: int = SCALE,
                 point_size: int = POINT_SIZE,
                 line_weight: int = LINE_WEIGHT,
                 bg_color: Optional[str] = BG_COLOR,
                 color_mode: str = "RGB"):
        self.width = width
        self.height = height
        self.scale = scale
        self.point_size = point_size
        self.line_weight = line_weight
        self.point_color = POINT_COLOR
        self.line_color = LINE_COLOR
        self.fill_color = FILL_COLOR
        if not bg_color:
            self.bg_color = "#00000000"
        else:
            self.bg_color = bg_color
        self._img = Image.new(color_mode, (width, height), self.bg_color)
        self._draw = ImageDraw.Draw(self._img)

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
        return array / array[:, [-1]]

    def point(self,
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

    def circle(self,
              x: float, y: float, radius: float,
              normalize: bool = True,
              color: Optional[str] = None,
              outline: Optional[str] = None):
        coords = x - radius, y + radius, x + radius, y - radius
        if normalize:
            coords = list(self.normalize(*coords))
        self._draw.ellipse(coords,
                           fill=color or self.point_color,
                           outline=None)

    def points(self, points: Iterable, color: str = "red"):
        for point in points:
            self.point(*point, color=color)

    def point_array(self,
                    points: np.ndarray,
                    *args, **kwargs):
        points = self.normalize_matrix(points)
        for x, y, z in points:
            self.point(x, y, *args, normalize=False, **kwargs)

    def edge(self,
             *args: Union[float, Tuple[float, float]],
             normalize: bool = True,
             color: Optional[str] = None,
             width: Optional[int] = None):

        if len(args) == 4:
            if normalize:
                x1, y1, x2, y2 = self.normalize(*args)
            else:
                x1, y1, x2, y2 = args
        elif len(args) == 2:
            if normalize:
                x1, y1, x2, y2 = self.normalize(*args[0], *args[1])
            else:
                (x1, y1), (x2, y2) = args
        else:
            raise TypeError("The input must be 4 floats or 2 tuples of 2 floats.")
        self._draw.line([x1, y1, x2, y2],
                        width=width or self.line_weight,
                        fill=color or self.line_color)

    def edges(self,
              lines: Iterable,
              color: Optional[str] = None,
              width: Optional[int] = None):
        for line in lines:
            self.edge(*line, color=color, width=width)

    def norm_point(self,
                   coords: Union[Sequence[float], np.ndarray],
                   *args, **kwargs):
        x, y, *_ = coords / coords[-1]
        self.point(x, y, *args, **kwargs)

    def border(self, side: str):
        if side == "left":
            return np.asarray([1, 0, self.width / 2])
        elif side == "right":
            return np.asarray([1, 0, -self.width / 2])
        elif side == "top":
            return np.asarray([0, 1, -self.height / 2])
        elif side == "bottom":
            return np.asarray([0, 1, self.height / 2])

    def norm_line(self,
                  line: np.ndarray,
                  *args, **kwargs
                  ):
        """
        Calculates the two intersections with canvas borders and plots the line segment they define.
        """
        assert len(line) == 3
        a = np.cross(line, self.border("top"))
        b = np.cross(line, self.border("bottom"))
        if a[-1] == 0 or b[-1] == 0:
            a = np.cross(line, self.border("left"))
            b = np.cross(line, self.border("right"))
        a /= a[-1]
        b /= b[-1]
        a, b = self.normalize_matrix(np.asarray([a, b]))
        self.edge(a[:2], b[:2], normalize=False, **kwargs)

    def norm_lines(self,
                   lines: np.ndarray,
                   *args, **kwargs
                   ):
        assert len(lines[0]) == 3
        raise NotImplementedError
        a = np.cross(lines, self.border("top"))
        b = np.cross(lines, self.border("bottom"))
        if a[-1] == 0 or b[-1] == 0:
            a = np.cross(lines, self.border("left"))
            b = np.cross(lines, self.border("right"))
        a /= a[-1]
        b /= b[-1]
        self.edge(a[:2], b[:2], normalize=True, **kwargs)

    def show(self):
        self._img.show()

    def clear(self):
        self._draw.rectangle([0, 0, self.width, self.height], fill=self.bg_color)

    def polygon(self,
                vertices: Union[Sequence, np.ndarray],
                color: Optional[str] = None,
                outline: Optional[str] = None):
        vertices = self.normalize_matrix(vertices)[:, :2]
        vertices = tuple(map(tuple, vertices))
        self._draw.polygon(vertices,
                           outline=outline or self.line_color,
                           fill=color or self.fill_color)
