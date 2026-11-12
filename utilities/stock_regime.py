"""Two-state switching AR(1) with a common contemporaneous market loading."""
import numpy as np
from scipy.special import logsumexp
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def fit_regime(y, lag, market, maxiter=200):
    # Percentage units improve numerical conditioning; AR and beta are unchanged.
    model = MarkovRegression(
        np.asarray(y) * 100, k_regimes=2, trend="c",
        exog=np.column_stack([lag, market]) * 100,
        switching_exog=[True, False], switching_variance=True,
    )
    result = model.fit(maxiter=maxiter, disp=False, search_reps=0)
    if not result.mle_retvals.get("converged", False):
        raise ValueError("optimizer did not converge")
    p = np.asarray(result.params)
    coefficients = np.array([p[model.parameters[s, "exog"]] for s in range(2)])
    variance = np.array([p[model.parameters[s, "variance"]][0] for s in range(2)]) / 10000
    coefficients[:, 0] /= 100
    transition = model.regime_transition_matrix(p)[:, :, 0]
    probabilities = np.asarray(result.filtered_marginal_probabilities)[-1]
    occupancy = np.asarray(result.filtered_marginal_probabilities).mean(axis=0)
    if (not np.isfinite(p).all() or np.any(variance < 1e-10)
            or np.any(np.abs(coefficients[:, 1]) >= 1) or np.any(occupancy < .05)):
        raise ValueError("degenerate, unstable, or poorly occupied states")
    # Stable reporting convention: state 0 has the lower AR coefficient.
    order = np.argsort(coefficients[:, 1])
    return {"coef": coefficients[order], "variance": variance[order],
            "transition": transition[np.ix_(order, order)],
            "probability": probabilities[order], "llf": float(result.llf)}


def filter_step(model, probability, observed, lag, market):
    prior = model["transition"] @ probability
    means = model["coef"] @ np.array([1., lag, market])
    variance = model["variance"]
    logp = (np.log(np.maximum(prior, 1e-300))
            - .5 * (np.log(2 * np.pi * variance) + (observed - means) ** 2 / variance))
    return np.exp(logp - logsumexp(logp))


def forecast(model, probability, current_return, prediction_horizon=1):
    """Cumulative log-return moments, conditional on zero future market returns.

    Track state-weighted joint moments of the latest return and accumulated
    return. This preserves serial covariance and dependence on regime paths.
    """
    if (isinstance(prediction_horizon, bool) or not np.isfinite(prediction_horizon)
            or prediction_horizon < 1 or int(prediction_horizon) != prediction_horizon):
        raise ValueError("prediction_horizon must be a positive integer")
    transition = model["transition"]
    alpha, phi = model["coef"][:, 0], model["coef"][:, 1]
    p = np.asarray(probability, dtype=float)
    r = p * current_return
    r2 = p * current_return ** 2
    total = np.zeros_like(p)
    total2 = np.zeros_like(p)
    cross = np.zeros_like(p)
    for _ in range(int(prediction_horizon)):
        p, r, r2, total, total2, cross = (
            transition @ value for value in (p, r, r2, total, total2, cross))
        new_r = alpha * p + phi * r
        new_r2 = ((alpha ** 2 + model["variance"]) * p
                  + 2 * alpha * phi * r + phi ** 2 * r2)
        previous_total_new_r = alpha * total + phi * cross
        total2 = total2 + 2 * previous_total_new_r + new_r2
        total = total + new_r
        cross = previous_total_new_r + new_r2
        r, r2 = new_r, new_r2
    mean = float(total.sum())
    variance = float(total2.sum() - mean ** 2)
    return mean, np.sqrt(max(0., variance)), p
