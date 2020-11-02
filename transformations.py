import numpy as np


def rotate(theta: float = 0):
    return np.asarray([[np.cos(theta*2*np.pi), -np.sin(theta*2*np.pi), 0],
                       [np.sin(theta*2*np.pi), np.cos(theta*2*np.pi), 0],
                       [0, 0, 1]])


def skew(skew_x: float = 0, skew_y: float = 0):
    return np.asarray([[1, skew_x, 0],
                       [skew_y, 1, 0],
                       [0, 0, 1]])


def scale(scale_x: float = 0, scale_y: float = 0):
    return np.asarray([[scale_x, 0, 0],
                       [0, scale_y, 0],
                       [0, 0, 1]])


def translate(dx: float = 0, dy: float = 0):
    return np.asarray([[1, 0, dx],
                       [0, 1, dy],
                       [0, 0, 1]])
