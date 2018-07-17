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


def test_rescale_x():
    x, y = preprocess(np.array([1, 2]), np.array([1, 2]), rescale_x_to_upper_bound=None)

    assert np.array_equal(x, np.array([[1], [2]]))
    assert x.shape == y.shape == (2, 1)
    assert x.dtype == y.dtype == 'float'
    assert y.mean() == 0
    assert y.std() == 1

    x_orig = np.array([6, 17, 28, 40, 50])
    x, y = preprocess(x_orig, np.array([1, 2, 3, 4, 5]), rescale_x_to_upper_bound=5)
    for idx in range(x_orig.shape[0] - 1):
        assert np.allclose(x_orig[idx] / x_orig[idx + 1], x[idx, 0] / x[idx + 1, 0])
    assert np.array_equal(x, np.array([[0.6], [1.7], [2.8], [4.0], [5.0]]))
    assert x.shape == y.shape == (5, 1)
    assert x.dtype == y.dtype == 'float'
    assert y.mean() == 0
    assert np.allclose(y.std(), 1)

    x_orig = np.array([6, 17, 28, 40, 50])
    x, y = preprocess(x_orig, np.array([1, 2, 3, 4, 5]), rescale_x_to_upper_bound=11)
    for idx in range(x_orig.shape[0] - 1):
        assert np.allclose(x_orig[idx] / x_orig[idx + 1], x[idx, 0] / x[idx + 1, 0])
    assert np.array_equal(x, np.array([[1.32], [3.74], [6.16], [8.8], [11.]]))
    assert x.shape == y.shape == (5, 1)
    assert x.dtype == y.dtype == 'float'
    assert y.mean() == 0
    assert np.allclose(y.std(), 1)

    with pytest.raises(ValueError) as ex:
        preprocess(np.array([6, 17, 28, 40, 50]), np.array([1, 2, 3, 4, 5]), rescale_x_to_upper_bound=0)
    assert str(ex.value) == 'Bad upper bound passed for to rescale `x` or bad maximum found for `x`.'

    with pytest.raises(ValueError) as ex:
        preprocess(np.array([-6, -17, -28, -40, 0]), np.array([1, 2, 3, 4, 5]), rescale_x_to_upper_bound=0)
    assert str(ex.value) == 'Bad upper bound passed for to rescale `x` or bad maximum found for `x`.'
