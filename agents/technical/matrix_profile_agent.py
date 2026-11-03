import numpy as np
import pandas as pd

from agents.base_agents.trading_agent import TradingAgent
from utilities.matrix_profile_utils import (
    build_pattern_library,
    build_pattern_search_index,
    find_pattern_matches,
    forecast_from_matches,
    normalize_return_pattern,
)


class MatrixProfileAgent(TradingAgent):
    """Cross-stock analogue forecasts from normalized return subsequences."""

    def __init__(
        self,
        data,
        pattern_window=30,
        forecast_horizons=(5, 10, 20),
        primary_horizon=5,
        neighbours=30,
        maximum_matches_per_stock=3,
        minimum_neighbours=10,
        maximum_distance=1.25,
        distance_temperature=0.25,
        history_window=500,
        candidate_stride=5,
        decision_frequency="W-FRI",
        price_type="Close",
        auto_generate=True,
    ):
        super().__init__(data)
        self.algorithm_name = "MatrixProfile"
        self.score_column = "PatternForecast"
        self.pattern_window = int(pattern_window)
        self.forecast_horizons = tuple(
            sorted({int(value) for value in forecast_horizons})
        )
        self.primary_horizon = int(primary_horizon)
        self.neighbours = int(neighbours)
        self.maximum_matches_per_stock = int(maximum_matches_per_stock)
        self.minimum_neighbours = int(minimum_neighbours)
        self.maximum_distance = (
            None if maximum_distance is None else float(maximum_distance)
        )
        self.distance_temperature = float(distance_temperature)
        self.history_window = int(history_window)
        self.candidate_stride = int(candidate_stride)
        self.decision_frequency = decision_frequency
        self.price_type = price_type
        self._validate_parameters()

        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        self.forecast_matrix = pd.DataFrame()
        self.confidence_matrix = pd.DataFrame()
        self.neighbour_count_matrix = pd.DataFrame()
        self.mean_distance_matrix = pd.DataFrame()
        self.forecast_history = pd.DataFrame()
        self.latest_matches = {}
        self.first_score_position = None
        if auto_generate:
            self.run_all()

    def _validate_parameters(self):
        if self.pattern_window < 2:
            raise ValueError("pattern_window must be at least 2.")
        if not self.forecast_horizons or self.forecast_horizons[0] < 1:
            raise ValueError("forecast_horizons must contain positive integers.")
        if self.primary_horizon not in self.forecast_horizons:
            raise ValueError("primary_horizon must be included in forecast_horizons.")
        if self.neighbours < 1 or self.minimum_neighbours < 1:
            raise ValueError("neighbour counts must be positive.")
        if self.minimum_neighbours > self.neighbours:
            raise ValueError("minimum_neighbours cannot exceed neighbours.")
        if self.maximum_matches_per_stock < 1:
            raise ValueError("maximum_matches_per_stock must be positive.")
        if self.maximum_distance is not None and self.maximum_distance <= 0.0:
            raise ValueError("maximum_distance must be positive or None.")
        if self.distance_temperature <= 0.0:
            raise ValueError("distance_temperature must be positive.")
        if self.history_window < self.pattern_window:
            raise ValueError("history_window cannot be shorter than pattern_window.")
        if self.candidate_stride < 1:
            raise ValueError("candidate_stride must be positive.")
        try:
            pd.tseries.frequencies.to_offset(self.decision_frequency)
        except (TypeError, ValueError) as exc:
            raise ValueError("decision_frequency must be a valid pandas frequency.") from exc

    def _decision_dates(self):
        dates = pd.Series(self.data.index, index=self.data.index)
        grouped = dates.resample(self.decision_frequency).last().dropna()
        completed = grouped.index <= self.data.index[-1]
        return pd.DatetimeIndex(grouped.loc[completed].tolist())

    def _generate_all_signals(self):
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        library_patterns, library_metadata, returns = build_pattern_library(
            close,
            pattern_window=self.pattern_window,
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
        history_rows = []

        for date in self._decision_dates():
            position = index.get_loc(date)
            if position < self.pattern_window:
                continue
            search_index = build_pattern_search_index(
                position,
                library_patterns,
                library_metadata,
                forecast_horizons=self.forecast_horizons,
                neighbours=self.neighbours,
                history_window=self.history_window,
            )
            if search_index is None:
                continue
            date_matches = {}
            for stock in stocks:
                query = returns[stock].iloc[
                    position - self.pattern_window + 1:position + 1
                ].to_numpy(dtype=float)
                if normalize_return_pattern(query) is None:
                    continue
                matches = find_pattern_matches(
                    query,
                    query_stock=stock,
                    query_position=position,
                    library_patterns=library_patterns,
                    library_metadata=library_metadata,
                    forecast_horizons=self.forecast_horizons,
                    neighbours=self.neighbours,
                    maximum_matches_per_stock=self.maximum_matches_per_stock,
                    maximum_distance=self.maximum_distance,
                    history_window=self.history_window,
                    search_index=search_index,
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
                if count >= self.minimum_neighbours:
                    for horizon in self.forecast_horizons:
                        forecast_matrices[horizon].loc[date, stock] = diagnostics[
                            f"Forecast_{horizon}"
                        ]
                    if self.first_score_position is None:
                        self.first_score_position = position
                date_matches[stock] = matches
                history_rows.append({
                    "Date": date,
                    "Stock": stock,
                    **diagnostics,
                })
            self.latest_matches = date_matches

        self.forecast_matrix = forecast_matrices[self.primary_horizon]
        self.confidence_matrix = confidence
        self.neighbour_count_matrix = neighbour_counts
        self.mean_distance_matrix = mean_distances
        self.forecast_history = (
            pd.DataFrame(history_rows).set_index(["Date", "Stock"])
            if history_rows else pd.DataFrame()
        )
        carried_forecast = self.forecast_matrix.ffill()
        carried_confidence = confidence.ffill()
        self.signal_data = {}
        for stock in stocks:
            signals = pd.DataFrame(index=index)
            signals["price"] = close[stock]
            signals["return"] = returns[stock]
            for horizon in self.forecast_horizons:
                signals[f"Forecast_{horizon}"] = forecast_matrices[horizon][stock]
            signals["PatternForecast"] = carried_forecast[stock]
            signals["SignalStrength"] = carried_forecast[stock]
            signals["PatternConfidence"] = carried_confidence[stock]
            signals["NeighbourCount"] = neighbour_counts[stock].replace(0, np.nan).ffill().fillna(0).astype(int)
            signals["MeanDistance"] = mean_distances[stock].ffill()
            signals["Valid"] = signals["PatternForecast"].notna()
            signals["Position"] = (
                signals["Valid"] & signals["PatternForecast"].gt(0.0)
            ).astype(int)
            signals["Signal"] = np.sign(signals["Position"].diff().fillna(0)).astype(int)
            self.signal_data[stock] = signals

    def generate_signal_strategy(self, stock, mode="backtest"):
        if mode not in {"backtest", "live"}:
            raise ValueError("mode must be 'backtest' or 'live'.")
        if stock not in self.signal_data:
            raise KeyError(f"No signal data for stock={stock}")
        return self.signal_data[stock].tail(1) if mode == "live" else self.signal_data[stock]

    def get_matches(self, stock):
        if stock not in self.latest_matches:
            raise KeyError(f"No latest pattern matches for stock={stock}")
        return self.latest_matches[stock].copy()

    def run_all(self, mode="backtest"):
        self._generate_all_signals()
        self.calculate_returns()
        return self.signal_data
