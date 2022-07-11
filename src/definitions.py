from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    row: int
    col: int

    @property
    def point(self):
        return np.array([self.row, self.col])


@dataclass
class Box:
    tl: Point  # top left
    tr: Point  # top right
    br: Point  # bottom right
    bl: Point  # bottom left

    @property
    def points(self):
        return np.array([
            self.tl.point,
            self.tr.point,
            self.br.point,
            self.bl.point,
        ], dtype='float32')

    @property
    def width(self):
        return max(self.tr.col, self.br.col)

    @property
    def height(self):
        return max(self.br.row, self.bl.row)


@dataclass
class Polygon:
    left: np.ndarray
    top: np.ndarray
    right: np.ndarray
    bottom: np.ndarray

    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)

        raise Exception(f'Not item {item}')

    @property
    def points(self):
        return np.vstack((self.top, self.right, self.bottom[::-1], self.left[::-1]))

    @property
    def width(self):
        return int(self.right[:, 1].max())

    @property
    def height(self):
        return int(self.bottom[:, 0].max())


@dataclass
class Border:
    right: int
    bottom: int
    top: int = 0
    left: int = 0

    @property
    def slices(self):
        slice_rows = slice(self.top if self.top > 0 else None, self.bottom)
        slice_cols = slice(self.left if self.left > 0 else None, self.right)
        return slice_rows, slice_cols
