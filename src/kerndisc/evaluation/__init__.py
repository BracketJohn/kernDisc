r"""Package to evaluate performance of kernels.

This package provides:
    * The `evaluate_asts` method, which builds kernels from ASTs, then trains and scores them.

Example
-------
To evaluate ASTs run:
```
    > from kerndisc.description import pretty_ast
    > from kerndisc.evaluation import evaluate_asts
    > for ast, model_params, score in evaluate_asts(X, Y, asts):
    >     print(f'Ast\n`{pretty_ast(ast)}`\nhas scored `{score:.2f}`.')
```

The models/kernels performance is then scored by the selected metric, which can be set via the environment
variable `METRIC`. See the `scoring` package for more on this. Default metric is an altered version of
the bayesian information criterion.

"""

from ._evaluate import evaluate_asts

__all__ = [
    'evaluate_asts',
]
