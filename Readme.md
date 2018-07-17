# kernDisc

`kerndisc` is a library for automated kernel structure discovery in **univariate** data. It aims to find the best composition of kernels in order to represent a time series. It is thought to be useful to either:

* Be used repeatedly on different time series of the same variable to discover some recurring, stable structure:

TODO: Add some medical variable example here.

* Find a composited kernel that best describes a single time series for some variable:

TODO: Add picture of solar irradiance graph with description here.


Search and description of kernels is heavily inspired by the PhD thesis of [David Duvenaud et al.](http://www.cs.toronto.edu/~duvenaud/thesis.pdf), the [Automated Statistician](https://github.com/jamesrobertlloyd/gp-structure-search) project and [Lloyd et al.](https://arxiv.org/pdf/1402.4304.pdf).

In the future it is planned to bring down evaluation cost to `O(n^2)`, by employing upper, lower bound estimation as introduced by [Kim et al.](https://arxiv.org/abs/1706.02524).

## Usage

`kerndisc` can be used in the following way:

```python
> from kerndisc import discover
> k = discover(X, Y)
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


## Installation

```python
# TODO: Create installation text/package.
```

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
