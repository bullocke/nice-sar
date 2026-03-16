"""Shared type aliases for the nice-sar package."""

from os import PathLike
from typing import Union

import numpy as np
import numpy.typing as npt

# Path-like types
PathType = Union[str, PathLike[str]]

# Bounding box: (west, south, east, north)
BBox = tuple[float, float, float, float]

# Common array types
ArrayFloat32 = npt.NDArray[np.float32]
ArrayFloat64 = npt.NDArray[np.float64]
ArrayUInt8 = npt.NDArray[np.uint8]
