import numpy as np
import pandas as pd

from utilities.matrix_profile_utils import (
    build_pattern_library,
    normalize_return_pattern,
)


def relative_log_volume(volume, lookback=20):
    """Express volume relative to its trailing median on a log scale."""
    lookback = int(lookback)
    if lookback < 2:
        raise ValueError("volume_lookback must be at least 2.")
    numeric = volume.apply(pd.to_numeric, errors="coerce").where(lambda values: values > 0.0)
    log_volume = np.log(numeric)
    baseline = log_volume.rolling(lookback, min_periods=lookback).median()
    return log_volume - baseline


def build_volume_pattern_library(
    close,
    volume,
    pattern_window=30,
    volume_lookback=20,
    forecast_horizons=(5, 10, 20),
    candidate_stride=5,
):
    """Build aligned return and relative-volume pattern libraries."""
    return_patterns, metadata, returns = build_pattern_library(
        close,
        pattern_window=pattern_window,
        forecast_horizons=forecast_horizons,
        candidate_stride=candidate_stride,
    )
    relative_volume = relative_log_volume(volume, lookback=volume_lookback)
    volume_patterns = []
    keep = []
    for position, row in metadata.iterrows():
        values = relative_volume[row["Stock"]].iloc[
            int(row["StartPosition"]):int(row["EndPosition"]) + 1
        ]
        pattern = normalize_return_pattern(values.to_numpy(dtype=float))
        if pattern is None:
            continue
        keep.append(position)
        volume_patterns.append(pattern)
    if not keep:
        return (
            np.empty((0, int(pattern_window))),
            np.empty((0, int(pattern_window))),
            metadata.iloc[:0].copy(),
            returns,
            relative_volume,
        )
    keep_array = np.asarray(keep, dtype=int)
    return (
        return_patterns[keep_array],
        np.vstack(volume_patterns),
        metadata.iloc[keep_array].reset_index(drop=True),
        returns,
        relative_volume,
    )


def rerank_with_volume(
    return_matches,
    query_volume_pattern,
    library_volume_patterns,
    volume_weight=0.25,
    neighbours=30,
    maximum_matches_per_stock=3,
    maximum_distance=1.40,
):
    """Re-rank return analogues using a weighted relative-volume distance."""
    query = normalize_return_pattern(query_volume_pattern)
    if query is None or return_matches is None or return_matches.empty:
        return pd.DataFrame()
    volume_weight = float(volume_weight)
    if volume_weight < 0.0:
        raise ValueError("volume_weight cannot be negative.")
    positions = return_matches["LibraryPosition"].astype(int).to_numpy()
    volume_distances = np.linalg.norm(
        library_volume_patterns[positions] - query,
        axis=1,
    ) / np.sqrt(len(query))
    result = return_matches.copy()
    result["ReturnDistance"] = result["Distance"].astype(float)
    result["VolumeDistance"] = volume_distances
    result["Distance"] = np.sqrt(
        result["ReturnDistance"].pow(2)
        + volume_weight * result["VolumeDistance"].pow(2)
    )
    if maximum_distance is not None:
        result = result.loc[result["Distance"].le(float(maximum_distance))]
    result = result.sort_values("Distance")
    rows = []
    source_counts = {}
    for _, row in result.iterrows():
        stock = row["Stock"]
        if source_counts.get(stock, 0) >= int(maximum_matches_per_stock):
            continue
        rows.append(row)
        source_counts[stock] = source_counts.get(stock, 0) + 1
        if len(rows) >= int(neighbours):
            break
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()
