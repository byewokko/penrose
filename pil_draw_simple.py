import math
from typing import Iterable

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 800, 600
SCALE = 10

img = Image.new("RGB", (WIDTH, HEIGHT))
draw = ImageDraw.Draw(img)


def draw_line(x1: float, y1: float, x2: float, y2: float, color: str = "white"):
    x1 = WIDTH / 2 + x1 * SCALE
    y1 = HEIGHT / 2 - y1 * SCALE
    x2 = WIDTH / 2 + x2 * SCALE
    y2 = HEIGHT / 2 - y2 * SCALE
    draw.line([x1, y1, x2, y2], width=1, fill=color)


def draw_lines(lines: Iterable, color: str = "white"):
    for line in lines:
        draw_line(*line, color=color)


def draw_point(x: float, y: float, color: str = "red"):
    RADIUS = 2
    x = WIDTH / 2 + x * SCALE
    y = HEIGHT / 2 - y * SCALE
    draw.ellipse([x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS], fill=color)


def draw_points(points: Iterable, color: str = "red"):
    for point in points:
        draw_point(*point, color=color)


def show():
    img.show()


def clear():
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill="black")
