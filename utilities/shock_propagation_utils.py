import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


def residualize_common_effects(returns, sector_map=None, minimum_sector_size=3):
    """Remove leave-one-out market and, where reliable, sector returns."""
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty pandas DataFrame.")
    if minimum_sector_size < 2:
        raise ValueError("minimum_sector_size must be at least 2.")

    numeric = returns.apply(pd.to_numeric, errors="coerce")
    valid_values = numeric.notna().to_numpy(dtype=int)
    counts = valid_values.sum(axis=1)
    totals = numeric.sum(axis=1, skipna=True).to_numpy(dtype=float)
    peer_counts = counts[:, None] - valid_values
    peer_totals = totals[:, None] - numeric.fillna(0.0).to_numpy(dtype=float)
    market_peer_return = np.divide(
        peer_totals,
        peer_counts.astype(float),
        out=np.full(numeric.shape, np.nan, dtype=float),
        where=peer_counts > 0,
    )
    adjusted = numeric - pd.DataFrame(
        market_peer_return,
        index=numeric.index,
        columns=numeric.columns,
    )
    adjustment_mode = pd.Series("MARKET", index=numeric.columns, dtype=object)

    if sector_map is None:
        return adjusted, adjustment_mode
    if not hasattr(sector_map, "get"):
        raise ValueError("sector_map must map ticker symbols to sector labels.")

    groups = {}
    for stock in numeric.columns:
        sector = sector_map.get(stock)
        if sector is not None and pd.notna(sector):
            groups.setdefault(str(sector), []).append(stock)

    for members in groups.values():
        if len(members) < minimum_sector_size:
            continue
        sector_values = adjusted[members]
        sector_counts = sector_values.notna().sum(axis=1)
        sector_totals = sector_values.sum(axis=1, skipna=True)
        for stock in members:
            peer_count = sector_counts - sector_values[stock].notna().astype(int)
            peer_total = sector_totals - sector_values[stock].fillna(0.0)
            sector_peer_return = peer_total.div(peer_count.where(peer_count > 0))
            adjusted[stock] = adjusted[stock] - sector_peer_return
            adjustment_mode.loc[stock] = "MARKET_AND_SECTOR"

    return adjusted, adjustment_mode


def estimate_sparse_lag_network(
    residual_returns,
    lags=(1, 2, 3),
    alpha=0.0005,
    minimum_observations=126,
    minimum_source_coverage=0.80,
    max_iter=5000,
):
    """Estimate directed lag coefficients after partialling out target autoregression."""
    if not isinstance(residual_returns, pd.DataFrame) or residual_returns.empty:
        raise ValueError("residual_returns must be a non-empty pandas DataFrame.")
    lag_values = tuple(sorted({int(lag) for lag in lags}))
    if not lag_values or lag_values[0] < 1:
        raise ValueError("lags must contain positive integers.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least 2.")
    if not 0.0 < minimum_source_coverage <= 1.0:
        raise ValueError("minimum_source_coverage must be in (0, 1].")

    returns = residual_returns.apply(pd.to_numeric, errors="coerce")
    stocks = returns.columns
    coefficient_matrices = {
        lag: pd.DataFrame(0.0, index=stocks, columns=stocks, dtype=float)
        for lag in lag_values
    }
    source_coverage = returns.notna().mean()
    usable_sources = source_coverage[source_coverage >= minimum_source_coverage].index

    for target in stocks:
        own_lags = pd.concat(
            [returns[target].shift(lag).rename(f"own_lag_{lag}") for lag in lag_values],
            axis=1,
        )
        predictor_parts = []
        predictor_keys = []
        for source in usable_sources:
            if source == target:
                continue
            for lag in lag_values:
                predictor_parts.append(returns[source].shift(lag))
                predictor_keys.append((source, lag))
        if not predictor_parts:
            continue

        predictors = pd.concat(predictor_parts, axis=1)
        predictors.columns = pd.MultiIndex.from_tuples(
            predictor_keys, names=["Source", "Lag"]
        )
        valid = returns[target].notna() & own_lags.notna().all(axis=1)
        if int(valid.sum()) < minimum_observations:
            continue

        y = returns.loc[valid, target].to_numpy(dtype=float)
        own = own_lags.loc[valid].to_numpy(dtype=float)
        x_frame = predictors.loc[valid]
        x_frame = x_frame.fillna(x_frame.mean()).fillna(0.0)
        x = x_frame.to_numpy(dtype=float)

        controls = np.column_stack([np.ones(len(y)), own])
        y_residual = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
        x_residual = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
        x_mean = x_residual.mean(axis=0)
        x_scale = x_residual.std(axis=0, ddof=0)
        usable_columns = np.isfinite(x_scale) & (x_scale > 1e-12)
        if not usable_columns.any():
            continue

        x_standardized = (
            x_residual[:, usable_columns] - x_mean[usable_columns]
        ) / x_scale[usable_columns]
        model = Lasso(
            alpha=float(alpha),
            fit_intercept=False,
            max_iter=int(max_iter),
            selection="cyclic",
        )
        model.fit(x_standardized, y_residual)
        raw_coefficients = model.coef_ / x_scale[usable_columns]
        usable_keys = np.asarray(predictor_keys, dtype=object)[usable_columns]
        for (source, lag), coefficient in zip(usable_keys, raw_coefficients):
            if coefficient != 0.0:
                coefficient_matrices[int(lag)].loc[source, target] = coefficient

    return coefficient_matrices


def lag_network_edge_table(coefficient_matrices):
    """Convert lag coefficient matrices to a directed edge table."""
    rows = []
    for lag, matrix in coefficient_matrices.items():
        for source in matrix.index:
            nonzero = matrix.loc[source]
            nonzero = nonzero[nonzero.ne(0.0)]
            rows.extend(
                {
                    "Source": source,
                    "Target": target,
                    "Lag": int(lag),
                    "Coefficient": float(coefficient),
                    "AbsoluteCoefficient": abs(float(coefficient)),
                }
                for target, coefficient in nonzero.items()
            )
    if not rows:
        return pd.DataFrame(
            columns=["Source", "Target", "Lag", "Coefficient", "AbsoluteCoefficient"]
        )
    return pd.DataFrame(rows).sort_values(
        "AbsoluteCoefficient", ascending=False, ignore_index=True
    )
