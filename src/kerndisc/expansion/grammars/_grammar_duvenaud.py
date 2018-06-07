import logging
from typing import Dict, List

import gpflow
from lark import Lark, ParseError, Transformer, UnexpectedInput
from lark.lexer import Token

from ._kernels import BASE_KERNELS
from ._util import find_closing_bracket


_IMPLEMENTED_KERNEL_EXPRESSIONS = ('cosine', 'linear', 'matern12', 'matern32', 'matern52', 'periodic', 'rbf', 'white', 'rationalquadratic')
_LOGGER = logging.getLogger(__package__)
GRAMMAR = f"""// This kernel grammar is a close implementation of the grammar first defined by David Duvenaud et al. [2013],
              // in their paper: [Structure discovery in Nonparametric Regression through Compositional Kernel Search](https://arxiv.org/pdf/1302.4922.pdf) and
              // also described in `Automatic Model Construction with Gaussian Processes`, the PhD thesis by Duvenaud.

              // Grammar was extended by additional kernels (maternXX), we also force brackets every time we multiply here.

              // CX stands for implementation of rule CX from Appendix C of
              // `Automatic Model Construction with Gaussian Processes`.

              ?kernel: base_kernel // C3, C8
                     | constant // C3, C8
                     | sum
                     | product
                     | lax_product
                     | changepoint
                     | changewindow

              ?sum: kernel "+" base_kernel -> add // C1
                  | kernel "+" constant -> add // C1
                  | kernel "+" changepoint -> add // C1
                  | kernel "+" changewindow -> add // C1

              ?product: "(" kernel ")" "*" base_kernel -> mul // C2
                      | "(" kernel ")" "*" constant -> mul // C2
                      | "(" kernel ")" "*" changepoint -> mul // C2
                      | "(" kernel ")" "*" changewindow -> mul // C2

              ?lax_product: "(" kernel ")" "*" "(" base_kernel "+" constant ")" -> lax_mul // C11
                          | "(" kernel ")" "*" "(" constant "+" base_kernel ")" -> lax_mul // C11
                          | "(" kernel ")" "*" "(" constant "+" constant ")" -> lax_mul // C11

              ?changepoint: "cp" "(" kernel "," kernel ")" -> cp // C4

              ?changewindow: "cw" "(" kernel "," kernel ")" -> cw // C5, C6, C7

              ?base_kernel: BASE_KERNEL -> kernel
              ?constant: CONSTANT -> kernel

              BASE_KERNEL: {' | '.join([f'"{kernel_exp}"' for kernel_exp in _IMPLEMENTED_KERNEL_EXPRESSIONS])}

              CONSTANT: "constant"

              %import common.WS
              %ignore WS
            """


def parser() -> Lark:
    """Instantiate `david_duvenaud` parser.

    Returns
    -------
    parser: Lark
        Parser that converts `david_duvenaud` sentences into `AST`s.

    """
    return Lark(GRAMMAR, start='kernel')


def extender(kernel_expression: str) -> List[str]:
    """Generate a list of kernel expressions that represent all possible one step alterations of a kernel.

    All rules for extension are from Appendix C of `Automatic Model Construction with Gaussian Processes`.

    Changepoints and changewindows are not yet implemented.

    Parameters
    ----------
    kernel_expression: str
        Valid duvenaud kernel expression, such as `(rbf) * rq + linear`.

    Returns
    -------
    kernel_alterations: List[str]
        All possible one step alterations of a kernel expression. `linear` should return:
        ```
        [
            'cosine', 'linear', 'matern12', 'matern32', 'matern52', 'periodic', 'rbf', 'white', 'rationalquadratic',  # May be substituted by any other kernel.
            'linear + cosine', 'linear + linear', ..., 'linear + rationalquadratic',  # May be added to any other kernel.
            '(linear) * cosine', ..., 'linear * rationalquadratic',  # May be multiplied by any other kernel.
            '(linear) * (cosine + constant)', ..., '(linear) * (rationalquadratic + constant)',  # May apply `lax_product`.
            # 'cp(linear, linear)',  # cp may be applied.
            # 'cw(linear, linear)',  # cw may be applied on kernel.
            # 'cw(linear, constant)',  # cw may be applied on kernel, constant.
            # 'cw(constant, linear)',  # cw may be applied on constant, kernel.
        ]
        ```
        If `kernel_expression` were to consist out of products or sums, there would be further alterations:
        ```
        `linear + constant` -> ['linear', 'constant'],  # Splitting sums,
        `(linear) * constant` -> ['linear', 'constant']  # splitting products.
        ```
        Changepoints and changewindows are not yet implemented.

    Raises
    ------
    RuntimeError
        If either a kernel expression passed to the extender or generated by the extender is invalid.

    """
    if not isinstance(kernel_expression, str):
        _LOGGER.exception(f'Called duvenaud extender with non string kernel expression `{kernel_expression}`.')
        raise RuntimeError('Called extender with an invalid kernel expression.')

    p = parser()
    try:
        p.parse(kernel_expression)
    except (ParseError, UnexpectedInput):
        _LOGGER.exception(f'Called duvenaud extender with invalid kernel expression `{kernel_expression}`.')
        raise RuntimeError('Called extender with an invalid kernel expression.')

    kernel_alterations: List[str] = []

    # C3, C8
    kernel_alterations.extend(_IMPLEMENTED_KERNEL_EXPRESSIONS)

    # TODO: Re-add changepoints, changewindows once implemented.
    for kernel_exp in _IMPLEMENTED_KERNEL_EXPRESSIONS:
        kernel_alterations.extend([
            f'{kernel_expression} + {kernel_exp}',                  # C1
            f'({kernel_expression}) * {kernel_exp}',                # C2
            # f'cp({kernel_expression}, {kernel_expression})',      # C4
            # f'cw({kernel_expression}, {kernel_expression}',       # C5
            # f'cw({kernel_expression}, constant)',                 # C6
            # f'cw(constant, {kernel_expression})',                 # C7
            f'({kernel_expression}) * ({kernel_exp} + constant)',   # C11
        ])
    # C9, C10
    kernel_alterations.extend(_extender_subexpressions(kernel_expression))

    for k in kernel_alterations:
        try:
            p.parse(k)
        except (ParseError, UnexpectedInput):  # pragma: no cover
            _LOGGER.exception(f'Duvenaud extender generated invalid kernel expression `{k}`.')
            raise RuntimeError('Extender generated an invalid kernel expression.')

    return kernel_alterations


def _extender_subexpressions(kernel_expression: str) -> List[str]:
    """Search for `+` and `*` in kernel expression that are splittible.

    Method that looks for `+` and `* ` which can be split apart in the form of:
    ```
    `linear + constant` -> ['linear', 'constant'],  # Splitting sums,
    `linear * constant` -> ['linear', 'constant']  # splitting products.
    ```

    This also takes into account `cp`s and `cw`s and parantheses, e.g.:
    ```
    `(cp(linear + constant, white)) * rbf` -> ['cp(linear + constant, white)', 'rbf']
    ```
    The `+` operator in the changepoint cannot be split.

    Parameters
    ----------
    kernel_expression: str
        Valid duvenaud kernel expression, such as `(rbf) * rq + linear`.

    Returns
    -------
    kernel_alterations: List[str]
        All subexpressions that can be generated by splitting apart `+` and `*` in kernel expressions.

    """
    kernel_alterations: List[str] = []
    # Is there a `(`, `cp(` or `cw(`?
    while kernel_expression.find('(') > -1:
        positions: Dict[str, int] = {}

        pos_bracket = kernel_expression.find('(')
        positions['('] = pos_bracket  # There must be a bracket, thus `if pos_bracket > -1` not necessary.

        pos_cp = kernel_expression.find('cp')
        if pos_cp > -1:
            positions['cp'] = pos_cp

        pos_cw = kernel_expression.find('cw')
        if pos_cw > -1:
            positions['cw'] = pos_cw

        # Select token closest to beginning of `kernel_expression`.
        to_eliminate = sorted(positions, key=lambda token: positions[token])[0]

        # Find closing bracket, starting at opening bracket (add len of `cp`/`cw` if required).
        offset = 2 if to_eliminate in ['cw', 'cp'] else 0
        start = positions[to_eliminate]
        end = find_closing_bracket(kernel_expression, start + offset)

        if to_eliminate in ['cw', 'cp']:
            # Keep closing bracket and opening bracket.
            kernel_alterations.append(kernel_expression[start:end + 1])
        else:
            # Discard closing and opening bracket.
            kernel_alterations.append(kernel_expression[start + 1:end])

        # Cut of `(sub_exp) * `.
        kernel_expression = kernel_expression[end + 4:]

    # There is still some part of the `kernel_expression` left,
    # must therefore be a part w.o. `(`, `cp`, `cw`; this part might contain a `+`.
    if kernel_expression:
        kernel_alterations.extend(kernel_expression.split(' + '))

    return kernel_alterations


class KernelTransformer(Transformer):
    """Lark transformer that evaluates an AST generated by the duvenaud grammar."""

    def kernel(self, kernel: List[Token]):
        """Instantiate kernel."""
        return BASE_KERNELS[kernel[0]](1)

    def add(self, kernels: List[gpflow.kernels.Kernel]):
        """Add together a list of kernels."""
        return gpflow.kernels.Sum(kernels)

    def mul(self, kernels: List[gpflow.kernels.Kernel]):
        """Mutiply together a list of kernels."""
        return gpflow.kernels.Product(kernels)

    def lax_mul(self, kernels: List[gpflow.kernels.Kernel]):
        """Multiply together a kernel with a kernel and a constant."""
        _summed = gpflow.kernels.Sum(kernels[1:])
        return gpflow.kernels.Product([kernels[0], _summed])

    def cp(self, kernels: List[gpflow.kernels.Kernel]):  # pragma: no cover
        """Changepoint from the first kernel to the second kernel in the above list."""
        raise NotImplementedError

    def cw(self, kernels: List[gpflow.kernels.Kernel]):  # pragma: no cover
        """Changewindow from the first kernel to the second kernel in the above list."""
        raise NotImplementedError