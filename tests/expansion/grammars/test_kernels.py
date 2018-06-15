"""Tests here are only for the manually implemented kernels from `_kernels.py`."""
import gpflow

from kerndisc.expansion.grammars._kernels import ChangePoint # noqa: I202, I100


def test_changepoint():
    offset = 5.
    k1 = gpflow.kernels.Constant(1)
    k2 = gpflow.kernels.Constant(1)

    changepoint_k = ChangePoint(k1, k2, offset=offset)
