import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def normalize_return_pattern(values, epsilon=1e-12):
    """Z-normalize one finite return sequence."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        return None
    scale = float(array.std(ddof=0))
    if scale <= float(epsilon):
        return None
    return (array - array.mean()) / scale


def build_pattern_library(
    close,
    pattern_window=30,
    forecast_horizons=(5, 10, 20),
    candidate_stride=5,
):
    """Precompute local pattern vectors and outcomes; callers enforce maturity."""
    if not isinstance(close, pd.DataFrame) or close.empty:
        raise ValueError("close must be a non-empty DataFrame.")
    pattern_window = int(pattern_window)
    candidate_stride = int(candidate_stride)
    horizons = tuple(sorted({int(value) for value in forecast_horizons}))
    if pattern_window < 2:
        raise ValueError("pattern_window must be at least 2.")
    if candidate_stride < 1:
        raise ValueError("candidate_stride must be positive.")
    if not horizons or horizons[0] < 1:
        raise ValueError("forecast_horizons must contain positive integers.")

    prices = close.apply(pd.to_numeric, errors="coerce")
    returns = np.log(prices / prices.shift(1))
    patterns = []
    rows = []
    for stock in prices.columns:
        stock_returns = returns[stock].to_numpy(dtype=float)
        stock_prices = prices[stock].to_numpy(dtype=float)
        for end_position in range(pattern_window - 1, len(prices), candidate_stride):
            start_position = end_position - pattern_window + 1
            pattern = normalize_return_pattern(
                stock_returns[start_position:end_position + 1]
            )
            if pattern is None:
                continue
            row = {
                "Stock": stock,
                "StartPosition": start_position,
                "EndPosition": end_position,
                "StartDate": prices.index[start_position],
                "EndDate": prices.index[end_position],
            }
            for horizon in horizons:
                maturity_position = end_position + horizon
                outcome = np.nan
                if (
                    maturity_position < len(prices)
                    and np.isfinite(stock_prices[end_position])
                    and np.isfinite(stock_prices[maturity_position])
                    and stock_prices[end_position] > 0.0
                    and stock_prices[maturity_position] > 0.0
                ):
                    outcome = np.log(
                        stock_prices[maturity_position] / stock_prices[end_position]
                    )
                row[f"ForwardReturn_{horizon}"] = outcome
            patterns.append(pattern)
            rows.append(row)

    if not patterns:
        return np.empty((0, pattern_window)), pd.DataFrame(rows), returns
    return np.vstack(patterns), pd.DataFrame(rows), returns


def find_pattern_matches(
    query_pattern,
    query_stock,
    query_position,
    library_patterns,
    library_metadata,
    forecast_horizons=(5, 10, 20),
    neighbours=30,
    maximum_matches_per_stock=3,
    maximum_distance=1.25,
    history_window=500,
    search_multiplier=10,
    search_index=None,
):
    """Find causal, matured analogues while excluding overlapping self-matches."""
    query = normalize_return_pattern(query_pattern)
    if query is None or len(library_metadata) == 0:
        return pd.DataFrame()
    horizons = tuple(sorted({int(value) for value in forecast_horizons}))
    query_start = int(query_position) - len(query) + 1
    maximum_horizon = max(horizons)
    if search_index is None:
        search_index = build_pattern_search_index(
            query_position,
            library_patterns,
            library_metadata,
            forecast_horizons=horizons,
            neighbours=neighbours,
            history_window=history_window,
            search_multiplier=search_multiplier,
        )
    if search_index is None:
        return pd.DataFrame()
    candidate_positions = search_index["CandidatePositions"]
    requested = search_index["RequestedNeighbours"]
    model = search_index["Model"]
    distances, indices = model.kneighbors(query.reshape(1, -1))
    normalized_distances = distances[0] / np.sqrt(len(query))
    rows = []
    source_counts = {}
    for distance, local_position in zip(normalized_distances, indices[0]):
        if maximum_distance is not None and distance > float(maximum_distance):
            continue
        metadata_position = candidate_positions[local_position]
        row = library_metadata.iloc[metadata_position].to_dict()
        row["LibraryPosition"] = int(metadata_position)
        stock = row["Stock"]
        if (
            stock == query_stock
            and int(row["EndPosition"]) + maximum_horizon >= query_start
        ):
            continue
        if source_counts.get(stock, 0) >= int(maximum_matches_per_stock):
            continue
        row["Distance"] = float(distance)
        rows.append(row)
        source_counts[stock] = source_counts.get(stock, 0) + 1
        if len(rows) >= int(neighbours):
            break
    return pd.DataFrame(rows)


def build_pattern_search_index(
    query_position,
    library_patterns,
    library_metadata,
    forecast_horizons=(5, 10, 20),
    neighbours=30,
    history_window=500,
    search_multiplier=10,
):
    """Build one matured historical neighbour index reusable across query stocks."""
    if len(library_metadata) == 0:
        return None
    horizons = tuple(sorted({int(value) for value in forecast_horizons}))
    maximum_horizon = max(horizons)
    minimum_end = max(0, int(query_position) - int(history_window))
    eligible = (
        library_metadata["EndPosition"].ge(minimum_end)
        & library_metadata["EndPosition"].add(maximum_horizon).le(query_position)
    )
    candidate_positions = np.flatnonzero(eligible.to_numpy())
    if not len(candidate_positions):
        return None
    requested = min(
        len(candidate_positions),
        max(int(neighbours), int(neighbours) * int(search_multiplier)),
    )
    model = NearestNeighbors(
        n_neighbors=requested,
        metric="euclidean",
        algorithm="brute",
        n_jobs=1,
    )
    model.fit(library_patterns[candidate_positions])
    return {
        "Model": model,
        "CandidatePositions": candidate_positions,
        "RequestedNeighbours": requested,
    }


def forecast_from_matches(
    matches,
    forecast_horizons=(5, 10, 20),
    distance_temperature=0.25,
    primary_horizon=None,
):
    """Aggregate matched outcomes and return forecast diagnostics."""
    horizons = tuple(sorted({int(value) for value in forecast_horizons}))
    primary_horizon = int(primary_horizon or horizons[0])
    if primary_horizon not in horizons:
        raise ValueError("primary_horizon must be included in forecast_horizons.")
    result = {
        "NeighbourCount": 0,
        "MeanDistance": np.nan,
        "Confidence": np.nan,
    }
    result.update({f"Forecast_{horizon}": np.nan for horizon in horizons})
    if matches is None or matches.empty:
        return result
    temperature = float(distance_temperature)
    if temperature <= 0.0:
        raise ValueError("distance_temperature must be positive.")
    weights = np.exp(-matches["Distance"].to_numpy(dtype=float) / temperature)
    result["NeighbourCount"] = len(matches)
    result["MeanDistance"] = float(np.average(matches["Distance"], weights=weights))
    primary_values = None
    for horizon in horizons:
        values = pd.to_numeric(
            matches[f"ForwardReturn_{horizon}"], errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.any():
            forecast = float(np.average(values[valid], weights=weights[valid]))
            result[f"Forecast_{horizon}"] = forecast
            if horizon == primary_horizon:
                primary_values = values[valid]
    if primary_values is not None:
        primary_forecast = result[f"Forecast_{primary_horizon}"]
        agreement = np.mean(np.sign(primary_values) == np.sign(primary_forecast))
        result["Confidence"] = float(
            agreement * np.exp(-result["MeanDistance"])
        )
    return result
