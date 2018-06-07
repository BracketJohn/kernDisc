import numpy as np
import pytest

from kerndisc._preprocessing import preprocess  # noqa: I202, I100


def test_bad_shape():
    with pytest.raises(ValueError):
        preprocess(np.array([0]), np.array([0, 1]))


def test_sd_zero():
    x, y = preprocess(np.array([1, 2]), np.array([1, 1]))
    assert x.shape == y.shape == (2, 1)
    assert x.dtype == y.dtype == 'float'
    assert y.mean() == 0
    assert y.std() == 0


def test_sd_not_zero():
    x, y = preprocess(np.array([1, 2]), np.array([1, 2]))
    assert x.shape == y.shape == (2, 1)
    assert x.dtype == y.dtype == 'float'
    assert y.mean() == 0
    assert y.std() == 1
