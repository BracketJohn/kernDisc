"""Module to preprocess time series data."""
import logging
from typing import Optional, Tuple

import numpy as np


_LOGGER = logging.getLogger(__package__)


def preprocess(x: np.ndarray, y: np.ndarray, rescale_x_to_upper_bound: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply all preprocessing steps required to use data for kernel discovery.

    Takes initial data and applies the following preprocessing steps:
        * Reshape `x` and `y` to shape `(-1, 1)`,
        * cast both of them to type `float`,
        * normalize `y` to zero mean,
        * normalize `y` to standard deviation of 1, if standard deviation not 0.

    Optionally `x` is rescaled to some upper bound, as specified by `rescale_x_to_upper_bound`.

    Parameters
    ----------
    x: np.ndarray
        Some vector, assumed to be time points, of values `x_1, ..., x_n`.

    y: np.ndarray
        Some vector, assumed to be observations, of values `y_1, ..., y_n`.

    rescale_x_to_upper_bound: Optional[float]
        Rescale `x` to the range `[x.min() / x.max(), 1] * rescale_x_to_upper_bound`.

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

    x = _rescale_x(x, rescale_x_to_upper_bound)

    return x, y


def _rescale_x(x: np.ndarray, rescale_x_to_upper_bound: Optional[float]=None) -> np.ndarray:
    """Rescale `x` to a certain interval.

    `x` is rescaled to some interval keeping the relative distance `c_i` between input
    points `x_i, x_{i + 1}` constant:
    ```
        c_i = x_i / x_{i + 1}
    ```
    This is done by applying:
    ```
        x'_i = rescale_x_to_upper_bound * x_i / x.max()
    ```
    to each point `x_i`.

    This rescaling can lead to improvements for sparse time series with large
    intervals.

    Parameters
    ----------
    x: np.ndarray
        Some vector, assumed to be time points, of values `x_1, ..., x_n`.

    rescale_x_to_upper_bound: Optional[float]
        Rescale `x` to the range `[x.min() / x.max(), 1] * rescale_x_to_upper_bound`.

    Returns
    -------
    rescaled_x: np.ndarray
        `x` rescaled to the specified upper bound.

    Raises
    ------
    ValueError
        If either `rescale_x_to_upper_bound <= 0` or `x.max() == 0`, leading to inconsistent behavior.

    """
    if rescale_x_to_upper_bound is None:
        return x

    if rescale_x_to_upper_bound <= 0 or x.max() == 0:
        _LOGGER.exception(f'Bad upper bound specified for `x`: `{rescale_x_to_upper_bound}`, or bad maximum of `x` to rescale: `{x.max()}`.')
        raise ValueError('Bad upper bound passed for to rescale `x` or bad maximum found for `x`.')

    return rescale_x_to_upper_bound * x / x.max()
