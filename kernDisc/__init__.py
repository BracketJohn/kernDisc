"""Prodive univariate kernel dicovery.

This package provides the modules necessary to execute a univariate structured kernel discovery.

Example
-------
To execute kernel discovery, just do:

    $ from kerndisc import discovery
    $ k = discovery(X, Y)

This will then execute a kernel discovery search and return the best performing kernel.

"""
from os import environ
import logging

from .discovery import discovery


logging.basicConfig(level=environ.get('LOG_LEVEL', 'INFO'),
                    format='%(levelname)-8s [%(asctime)s] %(name)-12s » %(message)s')

__all__ = [
    'discovery',
]
