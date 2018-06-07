"""Module for evaluation utility functions."""
import gpflow
import numpy as np


def add_jitter_to_model(model: gpflow.models.Model, mean: float=0, sd: float=0.1) -> None:
    """Add randomness (jitter) to a models parameters.

    Randomness is drawn from a normal distribution with mean `mean` and standard deviation `sd`.

    This method works inplace on the model which is passed.

    Parameters
    ----------
    model: gpflow.models.Model
        Model to jitter.

    mean: float
        Mean of normal distribution that randomness is drawn from.

    sd: float
        Standard deviation of normal distribution that randomness is drawn from.
    """
    for param_pathname, param_value in model.read_values().items():
        model.assign({
            param_pathname: param_value + np.random.normal(loc=mean, scale=sd),
        })
