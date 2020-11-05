import math
from typing import Iterable

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 4*1280, 4*1024
SCALE = 50

img = Image.new("RGB", (WIDTH, HEIGHT))
draw = ImageDraw.Draw(img)


def draw_line(x1: float, y1: float, x2: float, y2: float, color: str = "white", width: int = 1):
    x1 = WIDTH / 2 + x1 * SCALE
    y1 = HEIGHT / 2 - y1 * SCALE
    x2 = WIDTH / 2 + x2 * SCALE
    y2 = HEIGHT / 2 - y2 * SCALE
    draw.line([x1, y1, x2, y2], width=width, fill=color)


def draw_lines(lines: Iterable, color: str = "white", width: int = 1):
    for line in lines:
        draw_line(*line, color=color, width=width)


def draw_edge(v1: tuple, v2: tuple, color: str = "white", width: int = 5):
    x1 = WIDTH / 2 + v1[0] * SCALE
    y1 = HEIGHT / 2 - v1[1] * SCALE
    x2 = WIDTH / 2 + v2[0] * SCALE
    y2 = HEIGHT / 2 - v2[1] * SCALE
    draw.line([x1, y1, x2, y2], width=width, fill=color)


def draw_edges(lines: Iterable, color: str = "white", width: int = 5):
    for line in lines:
        draw_edge(*line, color=color, width=width)


def draw_point(x: float, y: float, color: str = "red", size: int = 3):
    x = WIDTH / 2 + x * SCALE
    y = HEIGHT / 2 - y * SCALE
    draw.ellipse([x - size, y - size, x + size, y + size], fill=color)


def draw_points(points: Iterable, color: str = "red"):
    for point in points:
        draw_point(*point, color=color)


def show():
    img.show()


def clear():
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill="black")
