"""Module to maintain all metrics that are available to score models."""
import gpflow
import numpy as np

from ._util import get_prod_count_kernel


def negative_log_likelihood(model: gpflow.models.Model) -> float:
    r"""Calculate the negative logarithmic likelihood of a model.

    Uses gpflow method `compute_log_likelihood`, which returns:
    ```
        LL = \log p(Y | model, theta)
    ```
    With `theta` being the models parameters and `model` usually being a GP regression model,
    `LL` being the *non-negative* log likelihood.

    We then negate `LL` in order to obtain the negative log likelihood.

    Parameters
    ----------
    model: gpflow.models.Model
        Model to be scored.

    Returns
    -------
    score: float
        Negative logarithmic likelihood score of the passed model.

    """
    return -model.compute_log_likelihood()


def bayesian_information_criterion(model: gpflow.models.Model) -> float:
    """Calculate the bayesian information criterion (BIC) value of a model.

    Calculate:
    ```
        BIC = -2 * LL + |theta| * log(n) = 2 * neg(LL) + |theta| * log(n)
    ```
    Where `LL = log_likelihood(model)`, `|theta|` is the number of parameters of a model
    and `n` is the numer of data points the model was trained on.

    `n` is obtained by using the data shape information naturally stored by gpflow models.

    Parameters
    ----------
    model: gpflow.models.Model
        Model to be scored.

    Returns
    -------
    score: float
        BIC score of the passed model.

    """
    # `model.parameters` returns a generator, thus we have to exhaust it first.
    return 2 * negative_log_likelihood(model) + len(list(model.parameters)) * np.log(model.X.shape[0])


def bayesian_information_criterion_duvenaud(model: gpflow.models.Model) -> float:
    """Calculate the bayesian information criterion (BIC) value of a model.

    Here Duvenauds BIC is defined as:
    ```
        D_BIC = -2 * LL + |effective_theta| * log(n) = 2 * neg(LL) + |effective_theta| * log(n)
    ```
    Where `LL = log_likelihood(model)`, `|effective_theta|` is the number of parameters of a model *that matter*
    and `n` is the numer of data points the model was trained on.

    `n` is obtained by using the data shape information naturally stored by gpflow models.

    `effective_theta`, i.e., the effective parameters of a model, are all parameters of each of its base kernels,
        * minus `l_i - 1` for each product `i` in the models kernels made out of `l` sub kernels;
        (as only one variance per multiplied kernel matters, hence we subtract `l - 1` from the parameter count)
        * plus 2 for each changepoint (`location` and `steepness`);
        * plus 3 for each changewindow (`location`, `steepness` and `width`);
        * minus 1 for the variance of the models likelihood, which is not counted.

    A model with products will therefore usually statisfy: D_BIC < BIC, whereas changepoints and
    changewindows lead to the opposite.

    Parameters
    ----------
    model: gpflow.models.Model
        Model to be scored.

    Returns
    -------
    score: float
        Duvenaud D_BIC score of the passed model.

    """
    effective_theta_cnt = len(list(model.parameters)) - 1  # Minus 1 for variance of likelihood.

    effective_theta_cnt -= get_prod_count_kernel(model.kern) - 1
    # TODO: Add param addition for CPs and CWs (if their params aren't automatically recognized).
    return 2 * negative_log_likelihood(model) + effective_theta_cnt * np.log(model.X.shape[0])
