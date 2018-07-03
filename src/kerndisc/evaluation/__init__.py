r"""Package to evaluate performance of kernels.

This package provides:
    * The `evaluate` method, which builds kernels from ASTs, then trains and scores them.

Example
-------
To evaluate ASTs run:
```
    > from kerndisc.description import pretty_ast
    > from kerndisc.evaluation import evaluate
    > for ast, score in evaluate(X, Y, asts):
    >     print(f'Ast\n`{pretty_ast(ast)}`\nhas scored `{score:.2f}`.')
```

The models/kernels performance is then scored by the selected metric, which can be set via the environment
variable `METRIC`. See the `scoring` package for more on this. Default metric is an altered version of
the bayesian information criterion.

"""

from ._evaluate import evaluate

__all__ = [
    'evaluate',
]
