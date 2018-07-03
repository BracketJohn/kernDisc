"""Module that maintains all kernels that are available for kernel construction."""
from typing import Dict

import gpflow
from gpflow.kernels import (ArcCosine,
                            Constant,
                            Cosine,
                            Exponential,
                            Linear,
                            Matern12,
                            Matern32,
                            Matern52,
                            Periodic,
                            Polynomial,
                            Product,
                            RationalQuadratic,
                            RBF,
                            Sum,
                            White)

BASE_KERNELS: Dict[str, gpflow.kernels.Kernel] = {
    'arccosine': ArcCosine,
    'constant': Constant,
    'cosine': Cosine,
    'exponential': Exponential,
    'linear': Linear,
    'matern12': Matern12,
    'matern32': Matern32,
    'matern52': Matern52,
    'periodic': Periodic,
    'polynomial': Polynomial,
    'rationalquadratic': RationalQuadratic,
    'rbf': RBF,
    'white': White,
}

COMBINATION_KERNELS: Dict[str, gpflow.kernels.Kernel] = {
    'product': Product,
    'sum': Sum,
}
