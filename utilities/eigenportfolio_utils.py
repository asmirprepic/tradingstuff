import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.covariance import LedoitWolf


def align_eigenvectors(current, previous=None):
    """Match component order and sign to a previous eigenvector estimate."""
    if not isinstance(current, pd.DataFrame) or current.empty:
        raise ValueError("current must be a non-empty eigenvector DataFrame.")
    aligned = current.copy()
    if previous is None or previous.empty:
        for column in aligned:
            anchor = aligned[column].abs().idxmax()
            if aligned.loc[anchor, column] < 0.0:
                aligned[column] *= -1.0
        return aligned, list(range(aligned.shape[1]))

    previous_aligned = previous.reindex(index=aligned.index).fillna(0.0)
    common_count = min(previous_aligned.shape[1], aligned.shape[1])
    previous_values = previous_aligned.iloc[:, :common_count].to_numpy(dtype=float)
    current_values = aligned.to_numpy(dtype=float)
    previous_norm = np.linalg.norm(previous_values, axis=0)
    current_norm = np.linalg.norm(current_values, axis=0)
    denominator = previous_norm[:, None] * current_norm[None, :]
    similarity = np.divide(
        np.abs(previous_values.T @ current_values),
        denominator,
        out=np.zeros((common_count, aligned.shape[1]), dtype=float),
        where=denominator > 0.0,
    )
    previous_indices, current_indices = linear_sum_assignment(-similarity)
    matched = dict(zip(previous_indices.tolist(), current_indices.tolist()))
    order = [matched[index] for index in range(common_count) if index in matched]
    order.extend(index for index in range(aligned.shape[1]) if index not in order)
    aligned = aligned.iloc[:, order]
    aligned.columns = [f"PC{index + 1}" for index in range(aligned.shape[1])]

    for index in range(common_count):
        column = aligned.columns[index]
        previous_column = previous_aligned.iloc[:, index]
        if float(np.dot(previous_column, aligned[column])) < 0.0:
            aligned[column] *= -1.0
    return aligned, order


def estimate_shrunk_eigenportfolios(
    returns,
    component_count=8,
    minimum_coverage=0.90,
    previous_eigenvectors=None,
):
    """Estimate correlation eigenvectors and tradable volatility-adjusted loadings."""
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty pandas DataFrame.")
    if component_count < 1:
        raise ValueError("component_count must be positive.")
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in (0, 1].")

    numeric = returns.apply(pd.to_numeric, errors="coerce")
    coverage = numeric.notna().mean()
    volatility = numeric.std(ddof=1)
    usable = coverage.ge(minimum_coverage) & volatility.gt(1e-12)
    usable_stocks = numeric.columns[usable]
    if len(usable_stocks) < 2:
        raise ValueError("At least two sufficiently complete return series are required.")

    values = numeric[usable_stocks]
    standardized = values.sub(values.mean()).div(volatility[usable_stocks])
    standardized = standardized.fillna(0.0)
    covariance = LedoitWolf().fit(standardized.to_numpy(dtype=float)).covariance_
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    descending = np.argsort(eigenvalues)[::-1]
    count = min(int(component_count), len(usable_stocks))
    descending = descending[:count]
    eigenvalues = eigenvalues[descending]
    vectors = pd.DataFrame(
        0.0,
        index=numeric.columns,
        columns=[f"PC{index + 1}" for index in range(count)],
    )
    vectors.loc[usable_stocks] = eigenvectors[:, descending]
    vectors, order = align_eigenvectors(vectors, previous_eigenvectors)
    eigenvalues = eigenvalues[np.asarray(order, dtype=int)]
    explained = eigenvalues / np.trace(covariance)

    tradable = vectors.div(volatility.reindex(vectors.index), axis=0).fillna(0.0)
    return {
        "eigenvectors": vectors,
        "tradable_loadings": tradable,
        "eigenvalues": pd.Series(eigenvalues, index=vectors.columns),
        "explained_variance": pd.Series(explained, index=vectors.columns),
        "volatility": volatility.reindex(vectors.index),
    }


def long_only_component_sleeves(tradable_loadings):
    """Create positive and negative long-only sleeves for each component."""
    if not isinstance(tradable_loadings, pd.DataFrame) or tradable_loadings.empty:
        raise ValueError("tradable_loadings must be a non-empty DataFrame.")
    sleeves = {}
    for component in tradable_loadings.columns:
        loading = tradable_loadings[component]
        positive = loading.clip(lower=0.0)
        negative = (-loading.clip(upper=0.0))
        if positive.sum() > 0.0:
            sleeves[f"{component}_POS"] = positive / positive.sum()
        if negative.sum() > 0.0:
            sleeves[f"{component}_NEG"] = negative / negative.sum()
    return pd.DataFrame(sleeves, index=tradable_loadings.index).fillna(0.0)


def cap_and_normalize_long_weights(weights, maximum_weight=0.10):
    """Normalize positive weights subject to a hard per-stock cap."""
    maximum_weight = float(maximum_weight)
    if not 0.0 < maximum_weight <= 1.0:
        raise ValueError("maximum_weight must be in (0, 1].")
    original = pd.Series(weights, dtype=float).clip(lower=0.0).fillna(0.0)
    if original.sum() <= 0.0:
        return original

    result = pd.Series(0.0, index=original.index, dtype=float)
    remaining = original[original.gt(0.0)].copy()
    budget = 1.0
    while not remaining.empty and budget > 1e-12:
        proposed = remaining / remaining.sum() * budget
        over = proposed.gt(maximum_weight)
        if not over.any():
            result.loc[proposed.index] = proposed
            break
        capped = proposed.index[over]
        result.loc[capped] = maximum_weight
        budget = 1.0 - result.sum()
        remaining = remaining.drop(index=capped)
    return result.clip(lower=0.0, upper=maximum_weight)
