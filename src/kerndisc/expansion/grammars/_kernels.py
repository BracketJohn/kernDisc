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
    """Changepoint Kernel that switches from one kernel to another at a certain offset."""

    def __init__(self, kern_list, offset, variance):
        """Initialize changepoint at location `offset`, with variance `variance`."""
        super(ChangePoint, self).__init__(kern_list)
        self.offset = Param(np.array(offset).astype(np.float64))
        self.variance = Param(np.array(variance).astype(np.float64))

    def K(self, X, X2=None):
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
        a1 = reduce(tf.multiply,
                    [sig1, self.kern_list[0].K(X, X2)])
        a2 = reduce(tf.multiply,
                    [sig2, self.kern_list[1].K(X, X2)])
        return reduce(tf.add, [a1, a2])

    def Kdiag(self, X):
        """Calculate diagonal of covariance matrix only, convenience method."""
        XXT = self.variance * tf.squeeze(tf.matmul(tf.expand_dims(X, axis=1), tf.expand_dims(X, axis=2))) + self.offset
        sigm = tf.sigmoid(XXT)
        a1 = reduce(tf.multiply, [sigm, self.kern_list[0].Kdiag(X),
                                  sigm])
        a2 = reduce(tf.multiply, [(1. - sigm), self.kern_list[1].Kdiag(X),
                                  1. - sigm])
        return reduce(tf.add, [a1, a2])


class ChangeWindow(Combination):
    """Changewindow kernel that applies a kernel only in a range from some startingpoint `t_s` to endpoint `t_e`."""

    def __init__(self, kern_list, kernel_start, kernel_end, variance):
        """Initialize changewindow with some kernel which only applies in a certain window, then the other kernel applies again."""
        super(ChangePoint, self).__init__(kern_list)
        self.start = Param(np.array(kernel_start).astype(np.float64))
        self.end = Param(np.array(kernel_end).astype(np.float64))
        self.variance = Param(np.array(variance).astype(np.float64))

    def K(self, X, X2=None):
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
        a1 = reduce(tf.multiply,
                    [sig1, self.kern_list[0].K(X, X2)])
        a2 = reduce(tf.multiply,
                    [sig2, self.kern_list[1].K(X, X2)])
        return reduce(tf.add, [a1, a2])

    def Kdiag(self, X):
        """Calculate diagonal of covariance matrix only, convenience method."""
        XXT = self.variance * tf.squeeze(tf.matmul(tf.expand_dims(X, axis=1), tf.expand_dims(X, axis=2))) + self.offset
        sigm = tf.sigmoid(XXT)
        a1 = reduce(tf.multiply, [sigm, self.kern_list[0].Kdiag(X),
                                  sigm])
        a2 = reduce(tf.multiply, [(1. - sigm), self.kern_list[1].Kdiag(X),
                                  1. - sigm])
        return reduce(tf.add, [a1, a2])
