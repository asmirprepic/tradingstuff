import numpy as np
import pandas as pd


def build_correlation_graph(
    returns,
    top_k=8,
    minimum_correlation=0.25,
    remove_market_mode=True,
    minimum_observations=None,
):
    """Build a symmetric, non-negative top-k correlation adjacency matrix."""
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty pandas DataFrame.")
    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if not 0.0 <= float(minimum_correlation) <= 1.0:
        raise ValueError("minimum_correlation must be between 0 and 1.")

    numeric = returns.apply(pd.to_numeric, errors="coerce")
    if remove_market_mode:
        numeric = numeric.sub(numeric.mean(axis=1, skipna=True), axis=0)
    if minimum_observations is None:
        minimum_observations = max(10, len(numeric) // 2)
    minimum_observations = int(minimum_observations)
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least 2.")

    correlation = numeric.corr(min_periods=minimum_observations)
    correlation = correlation.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    correlation = correlation.clip(lower=0.0)
    np.fill_diagonal(correlation.values, 0.0)
    correlation = correlation.where(correlation >= float(minimum_correlation), 0.0)

    directed = pd.DataFrame(0.0, index=correlation.index, columns=correlation.columns)
    keep_count = min(top_k, max(0, len(correlation.columns) - 1))
    if keep_count:
        for stock in correlation.index:
            strongest = correlation.loc[stock].nlargest(keep_count)
            strongest = strongest[strongest > 0.0]
            directed.loc[stock, strongest.index] = strongest

    adjacency = directed.combine(directed.T, np.maximum)
    np.fill_diagonal(adjacency.values, 0.0)
    return adjacency


def neighbour_weighted_values(adjacency, values):
    """Calculate weighted neighbour values while ignoring unavailable nodes."""
    if not isinstance(adjacency, pd.DataFrame):
        raise ValueError("adjacency must be a pandas DataFrame.")
    if not adjacency.index.equals(adjacency.columns):
        raise ValueError("adjacency must have identical index and columns.")
    values = pd.Series(values, dtype=float).reindex(adjacency.columns)
    valid = values.notna().to_numpy(dtype=float)
    weights = adjacency.to_numpy(dtype=float)
    denominator = weights @ valid
    numerator = weights @ values.fillna(0.0).to_numpy(dtype=float)
    result = np.divide(
        numerator,
        denominator,
        out=np.full(len(adjacency), np.nan, dtype=float),
        where=denominator > 0.0,
    )
    return pd.Series(result, index=adjacency.index, name="NeighbourValue")


def random_walk_laplacian(adjacency):
    """Return I - D^-1 A, with undefined rows for isolated nodes."""
    if not isinstance(adjacency, pd.DataFrame):
        raise ValueError("adjacency must be a pandas DataFrame.")
    if not adjacency.index.equals(adjacency.columns):
        raise ValueError("adjacency must have identical index and columns.")
    weights = adjacency.to_numpy(dtype=float)
    degree = weights.sum(axis=1)
    transition = np.divide(
        weights,
        degree[:, None],
        out=np.zeros_like(weights),
        where=degree[:, None] > 0.0,
    )
    laplacian = np.eye(len(adjacency)) - transition
    laplacian[degree <= 0.0, :] = np.nan
    return pd.DataFrame(laplacian, index=adjacency.index, columns=adjacency.columns)


def graph_deviation(adjacency, values):
    """Calculate node value minus its weighted-neighbour value."""
    values = pd.Series(values, dtype=float).reindex(adjacency.index)
    neighbours = neighbour_weighted_values(adjacency, values)
    return (values - neighbours).rename("GraphDeviation")
