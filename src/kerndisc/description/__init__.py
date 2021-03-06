"""Package to describe kernels.

Description is seen as different tasks here:
    * Take a description (aka. representation) of a kernel in one form and transform it into another. E.g., take
      the AST of a kernel and transform it into a gpflow kernel or a string.
    * Take a kernel in its AST form and simplify it.
    * Take a special, simplified kernel AST and describe it using natural language.

This package provides these abilities.

Additionally it also offers some helper functions, e.g., to instantiate models from kernels or ASTs.

Example
-------
1) Transforming form AST to gpflow kernel:
```
    > from kerndisc.description import ast_to_kernel
    > kernel = ast_to_kernel(ast)
```
for any valid AST. A valid AST can be generated by using `kernel_to_ast` on ANY gpflow kernel.

2) Simplifying a kernel:
```
    > from kerndisc.description import ast_to_kernel, kernel_to_ast
    > from kerndisc.description import simplify
    > simplified_kernel = ast_to_kernel(kernel_to_ast(kernel))
```

3) Describing a kernel (in AST form) using natural language:
```
    > from kerndisc.description import describe, pretty_ast
    > print(pretty_ast(kernel))

    Product
    |-- RBF
    |-- RBF
    |-- Linear
    |-- Linear
    +-- Sum
        |-- RBF
        |-- RBF
        +-- Product
            |-- White
            |-- White
            +-- Linear

    > print(describe(kernel))

    Smooth function with polynomially varying amplitude of degree `2`;
    Smooth function with polynomially varying amplitude of degree `2`;
    Uncorrelated noise with polynomially varying amplitude of degree `3`;
```

Note that natural language description is NOT available for all kernels currently,
as it is not clear how to describe effect of some of the kernels. Description is fully
functional for the grammar `duvenaud`, as Duvenaud et al. is the source of the `_describe`
module.

"""
from ._describe import describe
from ._instantiate import instantiate_model_from_ast, instantiate_model_from_kernel
from ._simplify import simplify
from ._transform import ast_to_kernel, ast_to_text, kernel_to_ast
from ._util import pretty_ast


__all__ = [
    'ast_to_text',
    'ast_to_kernel',
    'instantiate_model_from_ast',
    'instantiate_model_from_kernel',
    'describe',
    'kernel_to_ast',
    'pretty_ast',
    'simplify',
]
