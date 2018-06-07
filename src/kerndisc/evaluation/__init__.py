"""Package to evaluate performance of kernel expressions.

This package provides:
    * The `evaluate` method, which builds kernels from kernel expressions, then trains and scores them.

Example
-------
To evaluate kernel expressions run:
```
    > from kerndisc.evaluation import evaluate
    > for kernel_expression, score in evaluate(X, Y, kernel_expressions):
    >     print(f'Kernel `{kernel_expressions}` has score `{score}`.')
```

Evaluation uses multiple cores if applicable. How many cores it used can be set via the environment variable `CORES`.
Default is `1` core. `CORES=4` would lead to `4` cores being used.

The models/kernels performance is then scored by the selected metric, which can be set via the environment
variable `METRIC`. See the `scoring` package for more on this.
Default metric is an altered version of the bayesian information criterion.

"""

from ._evaluate import evaluate

__all__ = [
    'evaluate',
]
