import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


def _standardize_frame(frame):
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    filled = numeric.fillna(numeric.mean()).fillna(0.0)
    centered = filled - filled.mean()
    scale = centered.std(ddof=0).where(lambda values: values > 1e-12)
    return centered.div(scale, axis=1).fillna(0.0)


def screen_lagged_leaders(returns, lags=(1, 2, 3), maximum_leaders=15):
    """Screen source stocks by their strongest absolute lagged correlation."""
    values = returns.apply(pd.to_numeric, errors="coerce")
    lags = tuple(sorted({int(lag) for lag in lags}))
    if not lags or lags[0] < 1:
        raise ValueError("lags must contain positive integers.")
    if int(maximum_leaders) < 1:
        raise ValueError("maximum_leaders must be positive.")

    maximum_lag = max(lags)
    targets = _standardize_frame(values.iloc[maximum_lag:])
    strongest = pd.DataFrame(0.0, index=values.columns, columns=values.columns)
    for lag in lags:
        sources = _standardize_frame(values.shift(lag).iloc[maximum_lag:])
        correlations = sources.to_numpy(dtype=float).T @ targets.to_numpy(dtype=float)
        correlations /= max(1, len(targets))
        strongest = strongest.combine(
            pd.DataFrame(
                np.abs(correlations),
                index=values.columns,
                columns=values.columns,
            ),
            np.maximum,
        )
    np.fill_diagonal(strongest.values, 0.0)
    return {
        target: strongest[target].nlargest(int(maximum_leaders)).index.tolist()
        for target in strongest.columns
    }, strongest


def estimate_lead_lag_network(
    returns,
    lags=(1, 2, 3),
    maximum_leaders=15,
    alpha=0.0001,
    minimum_observations=126,
    minimum_source_coverage=0.80,
    max_iter=5000,
):
    """Estimate sparse directed edges after controlling for target autoregression."""
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least 2.")
    if not 0.0 < minimum_source_coverage <= 1.0:
        raise ValueError("minimum_source_coverage must be in (0, 1].")

    numeric = returns.apply(pd.to_numeric, errors="coerce")
    lags = tuple(sorted({int(lag) for lag in lags}))
    candidates, screening_scores = screen_lagged_leaders(
        numeric,
        lags=lags,
        maximum_leaders=maximum_leaders,
    )
    stocks = numeric.columns
    coverage = numeric.notna().mean()
    coefficient_matrices = {
        lag: pd.DataFrame(0.0, index=stocks, columns=stocks, dtype=float)
        for lag in lags
    }

    for target in stocks:
        leaders = [
            source
            for source in candidates[target]
            if coverage.get(source, 0.0) >= minimum_source_coverage
        ]
        if not leaders:
            continue
        own_lags = pd.concat(
            [numeric[target].shift(lag).rename(f"own_{lag}") for lag in lags],
            axis=1,
        )
        predictor_keys = [(source, lag) for source in leaders for lag in lags]
        predictors = pd.concat(
            [numeric[source].shift(lag) for source, lag in predictor_keys],
            axis=1,
        )
        predictors.columns = pd.MultiIndex.from_tuples(
            predictor_keys, names=["Source", "Lag"]
        )
        valid = numeric[target].notna() & own_lags.notna().all(axis=1)
        if int(valid.sum()) < minimum_observations:
            continue

        y = numeric.loc[valid, target].to_numpy(dtype=float)
        controls = np.column_stack([
            np.ones(len(y)),
            own_lags.loc[valid].to_numpy(dtype=float),
        ])
        x_frame = predictors.loc[valid]
        x_frame = x_frame.fillna(x_frame.mean()).fillna(0.0)
        x = x_frame.to_numpy(dtype=float)
        y_residual = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
        x_residual = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
        scale = x_residual.std(axis=0, ddof=0)
        usable = np.isfinite(scale) & (scale > 1e-12)
        if not usable.any():
            continue

        standardized = x_residual[:, usable] / scale[usable]
        model = Lasso(
            alpha=float(alpha),
            fit_intercept=False,
            max_iter=int(max_iter),
            selection="cyclic",
        )
        model.fit(standardized, y_residual)
        raw_coefficients = model.coef_ / scale[usable]
        usable_keys = np.asarray(predictor_keys, dtype=object)[usable]
        for (source, lag), coefficient in zip(usable_keys, raw_coefficients):
            if coefficient != 0.0:
                coefficient_matrices[int(lag)].loc[source, target] = coefficient

    return coefficient_matrices, candidates, screening_scores


def lead_lag_edge_table(coefficient_matrices):
    rows = []
    for lag, matrix in coefficient_matrices.items():
        for source in matrix.index:
            coefficients = matrix.loc[source]
            for target, coefficient in coefficients[coefficients.ne(0.0)].items():
                rows.append({
                    "Source": source,
                    "Target": target,
                    "Lag": int(lag),
                    "Coefficient": float(coefficient),
                    "AbsoluteCoefficient": abs(float(coefficient)),
                })
    columns = ["Source", "Target", "Lag", "Coefficient", "AbsoluteCoefficient"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "AbsoluteCoefficient", ascending=False, ignore_index=True
    )
