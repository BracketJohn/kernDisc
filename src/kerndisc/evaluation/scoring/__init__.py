"""Package to maintain metrics.

Usage
-----
A new metric can be implemented in the `_metrics` module, afterwards it can be imported and
added to the `_METRICS` dictionary here. Then it can be selected for training by setting the
environment variable `METRIC` to its name.

All metrics MUST be better when lower, i.e., result in a minimization problem.

Example
-------
To score a model, run:
```
> from kerndisc.evaluation.scoring import score
> s = score(some_model)
> print(s)
```


"""
from functools import lru_cache
import os

import gpflow

from ._metrics import (bayesian_information_criterion,
                       bayesian_information_criterion_duvenaud,
                       negative_log_likelihood)


_STANDARD_METRIC = 'bayesian_information_criterion_duvenaud'
_METRICS = {
    'log_likelihood': negative_log_likelihood,
    'bayesian_information_criterion': bayesian_information_criterion,
    'bayesian_information_criterion_duvenaud': bayesian_information_criterion_duvenaud,
}


@lru_cache(maxsize=1)
def get_current_metric() -> str:
    """Get the name of the currently selected metric.

    Returns
    -------
    metric_func: str
        Name of the currently selected metric.

    """
    return os.environ.get('METRIC', _STANDARD_METRIC)


def score_model(model: gpflow.models.Model) -> float:
    """Score a model using the currently selected metric.

    Metric for scoring can be selected by setting the environment variable `METRIC` to one of
    the metrics available here. This can also be used to add a custom metric.

    Parameters
    ----------
    model: gpflow.models.Model
        Model to be scored using the selected metric.

    Returns
    -------
    score: float
        Score calculated by metric selected via environment variable `METRIC`.

    """
    return _METRICS[get_current_metric()](model)
