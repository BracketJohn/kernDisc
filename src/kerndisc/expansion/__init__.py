"""Package for kernel expansion.

This package prodives two main tools for kernel expansion:
    * A `grammars` package to define a kernel grammar to be used for search space population,
    * actual expansion that loads a grammar and uses it to expand kernels.

Example
-------
To expand a kernel run:
```
    > from kerndisc.expansion import expand_kernels
    > expanded_kernels = expand_kernels([k_ast_1, ..., k_ast_2])
```

A `k_ast_n` is any valid gpflow kernel ast generated by the `description` package.

More examples and a deeper explanation of this can be found in the `grammars` package.

"""
from ._expand import expand_kernels

__all__ = [
    'expand_kernels',
]
