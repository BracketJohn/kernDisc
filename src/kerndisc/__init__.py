"""Provides univariate kernel discovery.

This package provides the modules necessary to execute a univariate structured kernel discovery. It
provides two main methods:
    * `discover`, the actual search and main entry point of this library,
    * `preprocess`, which is the preprocessing `discover` applies before executing search.

Example
-------
To execute kernel structure discovery run:
```
    > from kerndisc import discover
    > k = discover(x, y)
```
This will then execute a kernel discovery and return the `find_n_best` best performing kernels and some additional
information about them.

To preprocess values in the same way as `discover` does, run:
```
    > from kerdisc import preprocess
    > X, Y = preprocess(X, Y, find_n_best=5)
```

TODO: Finish this once `kerndisc` is done.

"""
import logging
from os import environ

from ._discover import discover
from ._preprocessing import preprocess


logging.basicConfig(level=environ.get('LOG_LEVEL', 'INFO'),
                    format='%(levelname)-8s [%(asctime)s] %(name)-12s » %(message)s')
logging.getLogger('flake8').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

__all__ = [
    'discover',
    'preprocess',
]
