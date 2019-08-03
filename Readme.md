# kernDisc

`kerndisc` is a library for automated kernel structure discovery in **univariate** data. It aims to find the best composition of kernels in order to represent a time series. `kerndisc` currently possesses a test coverage of over 90 %. Still, there is no claim to correctness, contribution and correction is heavily desired.

It is thought to be useful to either:

* Be used repeatedly on different time series of the same variable to discover some recurring, stable structure

* Find a composited kernel that best describes a single time series for some variable


Search and description of kernels is heavily inspired by the PhD thesis of [David Duvenaud et al.](http://www.cs.toronto.edu/~duvenaud/thesis.pdf), the [Automated Statistician](https://github.com/jamesrobertlloyd/gp-structure-search) project and [Lloyd et al.](https://arxiv.org/pdf/1402.4304.pdf).

In the future it is planned to bring down evaluation cost to `O(n^2)`, by employing upper, lower bound estimation as introduced by [Kim et al.](https://arxiv.org/abs/1706.02524).

Currently, this library (development) is in idle mode, however, this is expected to change if there is any interest from the community in this.

## Installation

`kerndisc` can be installed in the following way on mac:

```sh
> git clone https://github.com/BracketJohn/kernDisc
> cd kernDisc
> brew install pipenv
> pipenv install
```

`brew` can be substituted for other package maangers on non-mac systems. Afterwards you can spawn an interactive `kerndisc` session executing:

```sh
> pipenv shell
> cd src
> python
```

This will create a new virtual environment and enter it. From there one can start to develop. Although, I would recommend `ipython` or some similar, enhanced, development environment instead.

An usage example can be found below.

## Usage

`kerndisc` can be used in the following way:

```python
> import numpy as np
> from kerndisc import discover
> X, Y = np.array([0, 1, 2, 3]), np.array([-1, 1, -1, 1])
> discover(X, Y)
...
    Depth `2`: Empty search space, no new asts found.

{'periodic': {'ast': Node("/<class 'gpflow.kernels.Periodic'>", full_name='Periodic'),
  'depth': 0,
  'params': {'GPR/kern/variance': array(1.00037322),
   'GPR/kern/lengthscales': array(0.09897968),
   'GPR/kern/period': array(0.66666667),
   'GPR/likelihood/variance': array(1.00000004e-06)},
  'score': -11.34804081379194},
 'highscore_progression': [inf, -11.34804081379194, -11.34804081379194],
 'termination_reason': 'Depth `2`: Empty search space, no new asts found.'}
```

For scoring the following metrics are available:

* Negative log likelihood (`negative_log_likelihood`),
* bayesian information criterion (BIC, `bayesian_information_criterion`),
* BIC modified to not take "irrelevant" parameters into account (Duvenaud et al., `bayesian_information_criterion_duvenaud`).

BIC is default, a metric can be selected by setting the environment variable `METRIC`. This can also be used to define custom metrics.

To populate the search space, i.e., the possible combinations of kernels that are explored, `kerndisc` uses a grammar from `kerndisc.expansion.grammars`.

It is also possible to define your own grammar for discovery and search space population.

### Defining your own Metric

A new metric can be implemented in the `kerndisc.evaluation.scoring._metrics` module, afterwards it can be imported and added to the `_METRICS` dictionary in the packages `__init__`. Then it can be selected for training by setting the environment variable `METRIC` to its name.

All metrics MUST be minimization problems, i.e., be better when lower.

### Defining your own Grammar

To define a new grammar, please create a new module in `kerndisc.expansion.grammars` called `_grammar_*.py`. This new module MUST offer:

* `expand_kernel`: A method that takes a single gpflow kernel and applies desired alterations to it.
* `IMPLEMENTED_BASE_KERNEL_NAMES`: A global `List[str]`, which contains only `BASE_KERNELS.keys()` from `_kernels.py`.
  The base kernels in this list represent all kernels implemented by the respective grammar.

Once your custom grammar is created, you can select it by adding it to the `_GRAMMARS` dictionary in `kerndisc.expansion.grammars.__init__.py` and then setting the environment variable `GRAMMAR` to your grammars name.

See:
* `kerndisc.expansion.grammars.__init__.py` for general concept and description,
* `kerndisc.expansion.grammars._grammar_duvenaud.py` for an example of a grammar.

## Development

`pipenv` is used for development. Please install it via `pip` if necessary. Usage:

```
> git clone https://github.com/BracketJohn/kernDisc
> cd kernDisc
> pipenv install --dev
> pipenv shell
```

This will install all necessary packages, create a new virtual environment and enter it. From there one can start to develop and test.

### Testing

Tests can be executed by running the following:
```
> pytest
```

Depending on your environment, it might be necessary to do this in a `pipenv shell`.
