# kernDisc

`kernDisc` is a library for automated kernel discovery. It aims to find the best composition of kernels in order to represent a timeseries of data,
written with the univariate case in mind.

It is heavily inspired by the PhD thesis of David Duvenaud et. al, and the [automated statistician](https://github.com/jamesrobertlloyd/gp-structure-search) project.

It aims to do a specced down minimal and pythonic version of their project, maybe bringing down evaluation cost to `O(n^2)` later, by employing upper, lower bound estimation developed by [Kim et. all](https://arxiv.org/abs/1706.02524).

## Usage

Kernel discovery can be used in the following way:

```python
# TODO: Write usage
```

## Install

`kernDisc` can easily be installed via pip:

```
> git clone https://github.com/BracketJohn/kernDisc
> cd kernDisc
> pip install .
```

It supports python3, so adapt to `pip3 install .` if applicable.

## Development

For development purposes `pipenv` is used. Please install it via `pip` if necessary. Usage:

```
> git clone https://github.com/BracketJohn/kernDisc
> cd kernDisc
> pipenv install
> pipenv shell
```

This will install all necessary packages, create a new virtual environment and enter it. From there one can start to develop and test.

### Testing

Tests can be executed running:
```
> pytest
```
