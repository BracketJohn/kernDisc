"""Module that maintains all kernels that are available for kernel construction."""
from functools import reduce
from typing import Dict

import gpflow
from gpflow import Param
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
    """Changepoing Kernel that switches from one kernel to another at a certain offset."""

    def __init__(self, kern_list, offset, variance):
        """Initialize Changepoing at location `offset`, with variance `variance`."""
        super(ChangePoint, self).__init__(kern_list)
        assert len(self.kern_list) == 2  # Has to transition between exactly 2 kernels.
        self.offset = Param(offset)
        self.variance = Param(variance)

    def K(self, X, X2=None, presliced=False):
        """Calculate covariance matrix."""
        if X2 is None:
            X2 = X
        XXT = self.variance * tf.squeeze(tf.matmul(tf.expand_dims(X, axis=1), tf.expand_dims(X, axis=2))) + self.offset
        XXT2 = self.variance * tf.squeeze(tf.matmul(tf.expand_dims(X2, axis=1), tf.expand_dims(X2, axis=2))) + self.offset
        sigm = tf.sigmoid(XXT)
        sigm2 = tf.sigmoid(XXT2)
        sig1 = tf.matmul(tf.expand_dims(sigm, axis=1),
                         tf.expand_dims(sigm2, axis=0))
        sig2 = tf.matmul(tf.expand_dims((1. - sigm), axis=1),
                         tf.expand_dims((1. - sigm2), axis=0))
        a1 = reduce(tf.mul,
                    [sig1, self.kern_list[0].K(X, X2)])
        a2 = reduce(tf.mul,
                    [sig2, self.kern_list[1].K(X, X2)])
        return reduce(tf.add, [a1, a2])

    def Kdiag(self, X, presliced=False):
        """Calculate diagonal of covariance matrix only, convenience method."""
        XXT = self.variance * tf.squeeze(tf.matmul(tf.expand_dims(X, axis=1), tf.expand_dims(X, axis=2))) + self.offset
        sigm = tf.sigmoid(XXT)
        a1 = reduce(tf.mul, [sigm, self.kern_list[0].Kdiag(X),
                             sigm])
        a2 = reduce(tf.mul, [(1. - sigm), self.kern_list[1].Kdiag(X),
                             1. - sigm])
        return reduce(tf.add, [a1, a2])
