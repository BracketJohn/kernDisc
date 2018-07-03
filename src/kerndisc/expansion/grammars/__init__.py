"""Package to maintain and load kernel grammars.

Usage
-----
A new grammar can be defined as a new module namend `_grammar_*.py`, similar to `_grammar_duvenaud.py`.

The new module MUST offer:
    * `expand_kernel`: A method that takes a gpflow kernel, applies all possible alterations allowed in the new grammar
                to is and returns those alterations.

After creation of the module, it can be imported here and added to the `_GRAMMARS` dictionary.
Then it can be selected for execution by setting the environment variable `GRAMMAR`.

For an example of a grammar module see `_grammar_duvenaud.py`.

"""
import os
from typing import Callable, Dict, List

import gpflow

from ._grammar_duvenaud import expand_kernel as expand_kernel_duvenaud


_GRAMMARS: Dict[str, Dict[str, Callable]] = {
    'duvenaud': expand_kernel_duvenaud,
}
SELECTED_GRAMMAR_NAME = os.environ.get('GRAMMAR', 'duvenaud')


def expand_kernel(kernel: gpflow.kernels.Kernel) -> List[gpflow.kernels.Kernel]:
    """Expand a kernel using the currently selected grammar.

    Expand takes a kernel, such as `white * constant` and returns
    all possible one step alterations of the kernel, using the current grammar.

    It is advised to instead use the `expand_kernels` method of the `expansion`
    module or call this method in a `gpflow.defer_build` environment, to keep
    expansion time to a minimum.

    Returns
    -------
    kernel_alterations: List[gpflow.kernels.Kernel]
        All expansions possible by applying current grammar.

    """
    _expand = _GRAMMARS[SELECTED_GRAMMAR_NAME]
    return _expand(kernel)
