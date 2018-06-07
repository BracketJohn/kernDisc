"""Module to preprocess time series data."""
import logging
from typing import Tuple

import numpy as np


_LOGGER = logging.getLogger(__package__)


def preprocess(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply all preprocessing steps required to use data for kernel discovery.

    Takes initial data and applies the following preprocessing steps:
        * Reshape `x` and `y` to shape `(-1, 1)`,
        * cast both of them to type `float`,
        * normalize `y` to zero mean,
        * normalize `y` to standard deviation of 1, if standard deviation not 0.

    Also asserts that `x` and `y` have same shape after reshaping.

    Parameters
    ----------
    x: np.ndarray
        Some vector, assumed to be time points, of values `x_1, ..., x_n`.

    y: np.ndarray
        Some vector, assumed to be observations, of values `y_1, ..., y_n`.

    Returns
    -------
    x, y: Tuple[np.ndarray, np.ndarray]
        Both reshaped to `(-1, 1)` and casted to type `float`, `y` with 0 mean and standard deviation 1, if applicable.

    Raises
    ------
    ValueError
        If `x` and `y` are not of same shape after reshaping.

    """
    x, y = x.reshape(-1, 1).astype(float), y.reshape(-1, 1).astype(float)

    if x.shape != y.shape:
        _LOGGER.exception(f'Shapes of x and y do not match! Shape of x is {x.shape}, shape of y is {y.shape}.')
        raise ValueError('Shapes of x and y do not match!')

    y -= y.mean()
    if not np.isclose(y.std(), 0):
        y /= y.std()

    return x, y
