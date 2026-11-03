import numpy as np
import pandas as pd

from agents.technical.matrix_profile_agent import MatrixProfileAgent
from utilities.matrix_profile_utils import (
    build_pattern_search_index,
    find_pattern_matches,
    forecast_from_matches,
    normalize_return_pattern,
)
from utilities.volume_matrix_profile_utils import (
    build_volume_pattern_library,
    rerank_with_volume,
)


class VolumeMatrixProfileAgent(MatrixProfileAgent):
    """Return-pattern analogue forecasts refined by relative-volume similarity."""

    def __init__(
        self,
        data,
        pattern_window=30,
        volume_lookback=20,
        volume_weight=0.25,
        return_candidate_pool=100,
        forecast_horizons=(5, 10, 20),
        primary_horizon=5,
        neighbours=30,
        maximum_matches_per_stock=3,
        minimum_neighbours=10,
        return_maximum_distance=1.25,
        maximum_distance=1.40,
        distance_temperature=0.25,
        history_window=500,
        candidate_stride=5,
        decision_frequency="W-FRI",
        price_type="Close",
        auto_generate=True,
    ):
        self.volume_lookback = int(volume_lookback)
        self.volume_weight = float(volume_weight)
        self.return_candidate_pool = int(return_candidate_pool)
        self.return_maximum_distance = (
            None
            if return_maximum_distance is None
            else float(return_maximum_distance)
        )
        super().__init__(
            data,
            pattern_window=pattern_window,
            forecast_horizons=forecast_horizons,
            primary_horizon=primary_horizon,
            neighbours=neighbours,
            maximum_matches_per_stock=maximum_matches_per_stock,
            minimum_neighbours=minimum_neighbours,
            maximum_distance=maximum_distance,
            distance_temperature=distance_temperature,
            history_window=history_window,
            candidate_stride=candidate_stride,
            decision_frequency=decision_frequency,
            price_type=price_type,
            auto_generate=False,
        )
        self.algorithm_name = "VolumeMatrixProfile"
        self.score_column = "VolumePatternForecast"
        if self.volume_lookback < 2:
            raise ValueError("volume_lookback must be at least 2.")
        if self.volume_weight < 0.0:
            raise ValueError("volume_weight cannot be negative.")
        if self.return_candidate_pool < self.neighbours:
            raise ValueError("return_candidate_pool cannot be smaller than neighbours.")
        if (
            self.return_maximum_distance is not None
            and self.return_maximum_distance <= 0.0
        ):
            raise ValueError("return_maximum_distance must be positive or None.")
        fields = self.data.columns.get_level_values(1)
        if "Volume" not in fields:
            raise ValueError("VolumeMatrixProfileAgent requires Volume data.")
        if auto_generate:
            self.run_all()

    def _generate_all_signals(self):
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        volume = self.data.xs("Volume", level=1, axis=1).astype(float)
        volume = volume.reindex(columns=close.columns)
        (
            return_patterns,
            volume_patterns,
            metadata,
            returns,
            relative_volume,
        ) = build_volume_pattern_library(
            close,
            volume,
            pattern_window=self.pattern_window,
            volume_lookback=self.volume_lookback,
            forecast_horizons=self.forecast_horizons,
            candidate_stride=self.candidate_stride,
        )
        index = close.index
        stocks = close.columns
        forecast_matrices = {
            horizon: pd.DataFrame(np.nan, index=index, columns=stocks)
            for horizon in self.forecast_horizons
        }
        confidence = pd.DataFrame(np.nan, index=index, columns=stocks)
        neighbour_counts = pd.DataFrame(0, index=index, columns=stocks, dtype=int)
        mean_distances = pd.DataFrame(np.nan, index=index, columns=stocks)
        return_distances = pd.DataFrame(np.nan, index=index, columns=stocks)
        volume_distances = pd.DataFrame(np.nan, index=index, columns=stocks)
        history_rows = []

        for date in self._decision_dates():
            position = index.get_loc(date)
            if position < self.pattern_window + self.volume_lookback - 1:
                continue
            search_index = build_pattern_search_index(
                position,
                return_patterns,
                metadata,
                forecast_horizons=self.forecast_horizons,
                neighbours=self.return_candidate_pool,
                history_window=self.history_window,
            )
            if search_index is None:
                continue
            date_matches = {}
            for stock in stocks:
                query_return = returns[stock].iloc[
                    position - self.pattern_window + 1:position + 1
                ].to_numpy(dtype=float)
                query_volume = relative_volume[stock].iloc[
                    position - self.pattern_window + 1:position + 1
                ].to_numpy(dtype=float)
                if (
                    normalize_return_pattern(query_return) is None
                    or normalize_return_pattern(query_volume) is None
                ):
                    continue
                return_matches = find_pattern_matches(
                    query_return,
                    query_stock=stock,
                    query_position=position,
                    library_patterns=return_patterns,
                    library_metadata=metadata,
                    forecast_horizons=self.forecast_horizons,
                    neighbours=self.return_candidate_pool,
                    maximum_matches_per_stock=self.return_candidate_pool,
                    maximum_distance=self.return_maximum_distance,
                    history_window=self.history_window,
                    search_index=search_index,
                )
                matches = rerank_with_volume(
                    return_matches,
                    query_volume,
                    volume_patterns,
                    volume_weight=self.volume_weight,
                    neighbours=self.neighbours,
                    maximum_matches_per_stock=self.maximum_matches_per_stock,
                    maximum_distance=self.maximum_distance,
                )
                diagnostics = forecast_from_matches(
                    matches,
                    forecast_horizons=self.forecast_horizons,
                    distance_temperature=self.distance_temperature,
                    primary_horizon=self.primary_horizon,
                )
                count = int(diagnostics["NeighbourCount"])
                neighbour_counts.loc[date, stock] = count
                mean_distances.loc[date, stock] = diagnostics["MeanDistance"]
                confidence.loc[date, stock] = diagnostics["Confidence"]
                if not matches.empty:
                    return_distances.loc[date, stock] = matches["ReturnDistance"].mean()
                    volume_distances.loc[date, stock] = matches["VolumeDistance"].mean()
                if count >= self.minimum_neighbours:
                    for horizon in self.forecast_horizons:
                        forecast_matrices[horizon].loc[date, stock] = diagnostics[
                            f"Forecast_{horizon}"
                        ]
                    if self.first_score_position is None:
                        self.first_score_position = position
                date_matches[stock] = matches
                history_rows.append({"Date": date, "Stock": stock, **diagnostics})
            self.latest_matches = date_matches

        self.forecast_matrix = forecast_matrices[self.primary_horizon]
        self.confidence_matrix = confidence
        self.neighbour_count_matrix = neighbour_counts
        self.mean_distance_matrix = mean_distances
        self.mean_return_distance_matrix = return_distances
        self.mean_volume_distance_matrix = volume_distances
        self.forecast_history = (
            pd.DataFrame(history_rows).set_index(["Date", "Stock"])
            if history_rows else pd.DataFrame()
        )
        carried_forecast = self.forecast_matrix.ffill()
        self.signal_data = {}
        for stock in stocks:
            signals = pd.DataFrame(index=index)
            signals["price"] = close[stock]
            signals["return"] = returns[stock]
            signals["RelativeLogVolume"] = relative_volume[stock]
            for horizon in self.forecast_horizons:
                signals[f"Forecast_{horizon}"] = forecast_matrices[horizon][stock]
            signals["VolumePatternForecast"] = carried_forecast[stock]
            signals["SignalStrength"] = carried_forecast[stock]
            signals["PatternConfidence"] = confidence[stock].ffill()
            signals["NeighbourCount"] = neighbour_counts[stock].replace(0, np.nan).ffill().fillna(0).astype(int)
            signals["MeanReturnDistance"] = return_distances[stock].ffill()
            signals["MeanVolumeDistance"] = volume_distances[stock].ffill()
            signals["Valid"] = signals["VolumePatternForecast"].notna()
            signals["Position"] = (
                signals["Valid"] & signals["VolumePatternForecast"].gt(0.0)
            ).astype(int)
            signals["Signal"] = np.sign(signals["Position"].diff().fillna(0)).astype(int)
            self.signal_data[stock] = signals
