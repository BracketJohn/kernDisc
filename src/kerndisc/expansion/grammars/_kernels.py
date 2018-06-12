"""Module that maintains all kernels that are available for kernel construction."""
import copy
from functools import reduce
from typing import Dict, Optional

import gpflow
from gpflow import Param, transforms
from gpflow.kernels import (ArcCosine,
                            Combination,
                            Constant,
                            Cosine,
                            Exponential,
                            Linear,
                            Matern12,
                            Matern32,
                            Matern52,
                            Periodic,
                            RationalQuadratic,
                            RBF,
                            White)
import numpy as np
import tensorflow as tf


_CONSTANT: Dict[str, gpflow.kernels.Kernel] = {
    'constant': Constant,
}
SPECIAL_KERNELS: Dict[str, gpflow.kernels.Kernel] = {
    'cp': 'PLACEHOLDER',
    'cw': 'PLACEHOLDER',
}
BASE_KERNELS: Dict[str, gpflow.kernels.Kernel] = {
    'arccosine': ArcCosine,
    'cosine': Cosine,
    'exponential': Exponential,
    'linear': Linear,
    'matern12': Matern12,
    'matern32': Matern32,
    'matern52': Matern52,
    'periodic': Periodic,
    'rbf': RBF,
    'rationalquadratic': RationalQuadratic,
    'white': White,
    **_CONSTANT,
}


class ChangePoint(Combination):
    """Changepoint (CP) Kernel that switches from one kernel to another at a certain offset.

    This kernels output is defined as:
    ```
        CP(k1, k2, θ)(X, X') = σ(X, θ) * k1(X, X') * σ(X', θ) + σ(X, θ) * k2(X, X') * σ(X', θ)
    ```
    Where `k1`, `k2` are kernel functions, `σ` is the sigmoid smooth step function. This kernel is useful
    for quick transitions between two kernel functions. `θ = (offset, variance)` are parameters of the sigmoid function
    that dictate at what (time-)point the transition occurs (`offset`) and what variance is added. `θ` is learned
    during training and applied to `X`, `X2` before `σ` is calculated.

    """

    def __init__(self, k1: gpflow.kernels.Kernel, k2: gpflow.kernels.Kernel, offset: float, variance: float=1.0):
        """Initialize changepoint at location `offset`, with variance `variance`."""
        super(ChangePoint, self).__init__([k1, k2])

        self.offset = Param(offset, transform=transforms.positive)
        self.variance = Param(variance, transform=transforms.positive)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray]=None):  # noqa: N802, N803
        """Calculate covariance matrix."""
        sigmoid_at_X = tf.sigmoid(self.variance * tf.squeeze(tf.expand_dims(X, axis=1) @ tf.expand_dims(X, axis=2)) + self.offset)  # noqa: N806

        if X2 is None:
            sigmoid_at_X2 = sigmoid_at_X  # noqa: N806
        else:
            sigmoid_at_X2 = tf.sigmoid(self.variance * tf.squeeze(tf.expand_dims(X2, axis=1) @  tf.expand_dims(X2, axis=2)) + self.offset)  # noqa: N806

        k1_multiplier = tf.expand_dims(sigmoid_at_X, axis=1) @ tf.expand_dims(sigmoid_at_X2, axis=0)
        k2_multiplier = tf.expand_dims((1. - sigmoid_at_X), axis=1) @ tf.expand_dims((1. - sigmoid_at_X2), axis=0)

        summand_1 = reduce(tf.multiply, [k1_multiplier, self.kern_list[0].K(X, X2)])
        summand_2 = reduce(tf.multiply, [k2_multiplier, self.kern_list[1].K(X, X2)])

        return reduce(tf.add, [summand_1, summand_2])

    def Kdiag(self, X: np.ndarray):  # noqa: N802, N803
        """Calculate diagonal of covariance matrix only, convenience method."""
        sigmoid_at_X = tf.sigmoid(self.variance * tf.squeeze(tf.expand_dims(X, axis=1) @ tf.expand_dims(X, axis=2)) + self.offset)  # noqa: N806

        summand_1 = reduce(tf.multiply, [sigmoid_at_X, self.kern_list[0].Kdiag(X), sigmoid_at_X])
        summand_2 = reduce(tf.multiply, [(1. - sigmoid_at_X), self.kern_list[1].Kdiag(X), 1. - sigmoid_at_X])

        return reduce(tf.add, [summand_1, summand_2])


class ChangeWindow(Combination):
    """Changewindow (CW) Kernel that switches from one kernel to another in a certain range.

    This kernels output is defined as:
    ```
        CW(k1, k2)(X, X') = CP(CP(k1, k2, θ_1), k1, θ_2)(X, X')
    ```
    Where `CP` is the changepoint function from above, `θ_1 = (offset_cp_1, variance_cp_1)` dictates
    at what point the transition from `k1` to `k2` occurs, `θ_2 = (offset_cp_2, variance_cp_2)` then
    forces the second transition from `k2` to `k1` at point `offset_cp_2`. Thus `offset_cp_1 < offset_cp_2`
    must hold.

    """

    def __init__(self, k1: gpflow.kernels.Kernel, k2: gpflow.kernels.Kernel, offset_cp_1: float, offset_cp_2: float,
                 variance: float=1.0, variance_cp_1: float=1.0, variance_cp_2: float=1.0):
        """Initialize changewindow."""
        if offset_cp_1 >= offset_cp_2:
            raise RuntimeError(f'Changewindow kernel was initialized with `offset_cp_1 = {offset_cp_1} >= offset_cp_2 = {offset_cp_2}`.')

        super(ChangeWindow, self).__init__([k1, k2])

        self.variance = Param(variance, transform=transforms.positive)

        self.cp_lower = ChangePoint(k1, k2, offset_cp_1, variance_cp_1)
        self.cp_upper = ChangePoint(self.cp_lower, copy.copy(k1), offset_cp_2, variance_cp_2)

        self.remaining_margin = Param(self.offset_margin, transform=transforms.positive)

    @property
    def offset_margin(self):
        return self.cp_upper.offset.value - self.cp_lower.offset.value

    def K(self, X, X2=None):  # noqa: N802, N803
        """Calculate covariance matrix."""
        return self.variance * self.cp_upper.K(X, X2)

    def Kdiag(self, X):  # noqa: N802, N803
        """Calculate diagonal of covariance matrix only, convenience method."""
        return self.variance * self.cp_upper.Kdiag(X)
