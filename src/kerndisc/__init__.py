"""Provide univariate kernel dicovery.

This package provides the modules necessary to execute a univariate structured kernel discovery.

Example
-------
To execute kernel structure discovery run:
```
    > from kerndisc import discover
    > k = discover(x, y)
```
This will then execute a kernel discovery search and return the best performing kernel.

TODO: Finish this once `kerndisc` is done.

"""
import logging
from os import environ

from ._discover import discover


logging.basicConfig(level=environ.get('LOG_LEVEL', 'INFO'),
                    format='%(levelname)-8s [%(asctime)s] %(name)-12s » %(message)s')
logging.getLogger('flake8').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

__all__ = [
    'discover',
]
