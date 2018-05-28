"""Module that maintains all kernels that are available for kernel construction."""
from typing import Dict

import gpflow
from gpflow.kernels import (Constant,
                            # ArcCosine,
                            Cosine,
                            # Exponential,
                            Linear,
                            Matern12,
                            Matern32,
                            Matern52,
                            Periodic,
                            RationalQuadratic,
                            RBF,
                            White)

_CONSTANT: Dict[str, gpflow.kernels.Kernel] = {
    'constant': Constant,
}
SPECIAL_KERNELS: Dict[str, gpflow.kernels.Kernel] = {
    'cp': 'PLACEHOLDER',
    'cw': 'PLACEHOLDER',
}
# TODO: Re-add kernels once interesting.
BASE_KERNELS: Dict[str, gpflow.kernels.Kernel] = {
    # 'arccosine': ArcCosine,
    'cosine': Cosine,
    # 'exponential': Exponential,
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
