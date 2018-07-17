"""Package to maintain and load kernel grammars.

Usage
-----
A new grammar can be defined as a new module namend `_grammar_*.py`, similar to `_grammar_duvenaud.py`.

The new module MUST offer:
    * `expand_kernel`: A method that takes a gpflow kernel, applies all possible alterations allowed in the new grammar
                to is and returns those alterations.
    * `IMPLEMENTED_BASE_KERNEL_NAMES`: A global `List[str]`, which contains only `BASE_KERNELS.keys()` from `_kernels.py`.
      The base kernels in this list represent all kernels implemented by the respective grammar.

After creation of the module, it can be imported here and added to the `_GRAMMARS` dictionary.
Then it can be selected for execution by setting the environment variable `GRAMMAR`.

For an example of a grammar module see `_grammar_duvenaud.py`.

"""
import os
from typing import Any, Callable, Dict, List, Optional

import gpflow

from ._grammar_duvenaud import (expand_kernel as expand_kernel_duvenaud,
                                IMPLEMENTED_BASE_KERNEL_NAMES as IMPLEMENTED_BASE_KERNEL_NAMES_DUVENAUD)


_GRAMMARS: Dict[str, Dict[str, Callable]] = {
    'duvenaud': {
        'expand_kernel': expand_kernel_duvenaud,
        'IMPLEMENTED_BASE_KERNEL_NAMES': IMPLEMENTED_BASE_KERNEL_NAMES_DUVENAUD,
    },
}
SELECTED_GRAMMAR_NAME = os.environ.get('GRAMMAR', 'duvenaud')
IMPLEMENTED_BASE_KERNEL_NAMES = _GRAMMARS[SELECTED_GRAMMAR_NAME]['IMPLEMENTED_BASE_KERNEL_NAMES']


def expand_kernel(kernel: gpflow.kernels.Kernel, grammar_kwargs: Optional[Dict[str, Any]]=None) -> List[gpflow.kernels.Kernel]:
    """Expand a kernel using the currently selected grammar.

    Expand takes a kernel, such as `white * constant` and returns
    all possible one step alterations of the kernel, using the current grammar.

    It is advised to instead use the `expand_asts` method of the `expansion`
    module or call this method in a `gpflow.defer_build` environment, to keep
    expansion time to a minimum.

    Parameters
    ----------
    kernel: gpflow.kernels.Kernel
        Kernel to be expanded by current grammar.

    grammar_kwargs: Optional[Dict[str, Any]]
        Options to be passed to grammars, to allow different configurations for manually implemented
        grammars.

    Returns
    -------
    kernel_alterations: List[gpflow.kernels.Kernel]
        All expansions possible by applying current grammar.

    """
    if grammar_kwargs is None:
        grammar_kwargs = {}

    _expand = _GRAMMARS[SELECTED_GRAMMAR_NAME]['expand_kernel']
    return _expand(kernel, **grammar_kwargs)
