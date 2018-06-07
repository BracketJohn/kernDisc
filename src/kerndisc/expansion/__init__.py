"""Package for kernel expression expansion.

This package prodives two main tools for kernel expansion:
    * A `grammars` package to define a kernel grammar to be used for search space population,
    * actual expansion that can load grammar and use it to expand kernel expressions,
    * a simplification method that simplifies generated kernel expressions according to TODO: to what?

Example
-------
To expand a kernel expression run:
```
    > from kerndisc.expansion import expand_kernel_expressions
    > expanded_expressions = expand_kernel_expressions([k_exp_1, ..., k_exp_2])
```

To simplify kernel expressions:
```
    > from kerndisc.expansion import simplify_kernel_expressions
    > simplified_expressions = simplify_kernel_expressions([k_exp_1, ..., k_exp_2])
```

A `kernel_expression` is any valid kernel expression in the currently selected grammar, i.e., any expression
that can be parsed by the grammars parser. An example of this would be `'linear'` or `'(rbf) * (constant + white)'
in the `duvenaud` grammar. More examples and a deeper explanation of this can be found in the `grammars` package.

"""
from ._expand import expand_kernel_expressions
from ._simplify import simplify_kernel_expressions

__all__ = [
    'expand_kernel_expressions',
    'simplify_kernel_expressions',
]