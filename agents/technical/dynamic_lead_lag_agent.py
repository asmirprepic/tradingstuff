import numpy as np
import pandas as pd

from agents.base_agents.trading_agent import TradingAgent
from utilities.graph_plotting import plot_directed_graph
from utilities.lead_lag_network import (
    estimate_lead_lag_network,
    lead_lag_edge_table,
)
from utilities.shock_propagation_utils import residualize_common_effects


class DynamicLeadLagAgent(TradingAgent):
    """Unconditional next-day forecasts from a rolling directed lead-lag network."""

    def __init__(
        self,
        data,
        estimation_window=252,
        max_lag=3,
        maximum_leaders_per_target=15,
        lasso_alpha=0.0001,
        minimum_observations=126,
        minimum_source_coverage=0.80,
        network_refit_frequency="monthly",
        decision_frequency="B",
        adjust_common_effects=True,
        sector_map=None,
        minimum_sector_size=3,
        price_type="Close",
        auto_generate=True,
    ):
        super().__init__(data)
        self.algorithm_name = "DynamicLeadLag"
        self.score_column = "LeadLagForecast"
        self.estimation_window = int(estimation_window)
        self.max_lag = int(max_lag)
        self.maximum_leaders_per_target = int(maximum_leaders_per_target)
        self.lasso_alpha = float(lasso_alpha)
        self.minimum_observations = int(minimum_observations)
        self.minimum_source_coverage = float(minimum_source_coverage)
        self.network_refit_frequency = str(network_refit_frequency).lower()
        self.decision_frequency = decision_frequency
        self.adjust_common_effects = bool(adjust_common_effects)
        self.sector_map = sector_map
        self.minimum_sector_size = int(minimum_sector_size)
        self.price_type = price_type
        self._validate_parameters()

        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        self.model_returns = pd.DataFrame()
        self.score_matrix = pd.DataFrame()
        self.edge_snapshots = {}
        self.network_history = pd.DataFrame()
        self.latest_edge_table = pd.DataFrame()
        self.first_score_position = None
        if auto_generate:
            self.run_all()

    def _validate_parameters(self):
        if self.estimation_window < 20:
            raise ValueError("estimation_window must be at least 20.")
        if not 1 <= self.max_lag < self.estimation_window:
            raise ValueError("max_lag must be positive and smaller than estimation_window.")
        if self.maximum_leaders_per_target < 1:
            raise ValueError("maximum_leaders_per_target must be positive.")
        if self.lasso_alpha <= 0.0:
            raise ValueError("lasso_alpha must be positive.")
        if not 2 <= self.minimum_observations < self.estimation_window:
            raise ValueError("minimum_observations must be between 2 and estimation_window.")
        if not 0.0 < self.minimum_source_coverage <= 1.0:
            raise ValueError("minimum_source_coverage must be in (0, 1].")
        if self.network_refit_frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError("network_refit_frequency must be daily, weekly, or monthly.")

    def _decision_dates(self):
        dates = pd.Series(self.data.index, index=self.data.index)
        grouped = dates.resample(self.decision_frequency).last().dropna()
        return set(grouped.loc[grouped.index <= self.data.index[-1]].tolist())

    def _refit_key(self, date):
        if self.network_refit_frequency == "daily":
            return date
        if self.network_refit_frequency == "weekly":
            iso = date.isocalendar()
            return int(iso.year), int(iso.week)
        return date.year, date.month

    def _generate_all_signals(self):
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        log_returns = np.log(close / close.shift(1))
        if self.adjust_common_effects:
            model_returns, _ = residualize_common_effects(
                log_returns,
                sector_map=self.sector_map,
                minimum_sector_size=self.minimum_sector_size,
            )
        else:
            model_returns = log_returns

        score_matrix = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        incoming_count = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=int)
        decisions = self._decision_dates()
        coefficients = None
        refit_key = None
        network_rows = []
        lags = tuple(range(1, self.max_lag + 1))

        for position, date in enumerate(close.index):
            if date not in decisions or position < self.estimation_window:
                continue
            current_key = self._refit_key(date)
            if coefficients is None or current_key != refit_key:
                window = model_returns.iloc[
                    position - self.estimation_window + 1:position + 1
                ]
                coefficients, candidates, _ = estimate_lead_lag_network(
                    window,
                    lags=lags,
                    maximum_leaders=self.maximum_leaders_per_target,
                    alpha=self.lasso_alpha,
                    minimum_observations=self.minimum_observations,
                    minimum_source_coverage=self.minimum_source_coverage,
                )
                edges = lead_lag_edge_table(coefficients)
                self.edge_snapshots[date] = edges.copy()
                network_rows.append({
                    "Date": date,
                    "Edges": len(edges),
                    "Sources": edges["Source"].nunique() if len(edges) else 0,
                    "Targets": edges["Target"].nunique() if len(edges) else 0,
                    "CandidateLeaders": sum(len(value) for value in candidates.values()),
                })
                refit_key = current_key

            forecast = pd.Series(0.0, index=close.columns, dtype=float)
            counts = pd.Series(0, index=close.columns, dtype=int)
            for lag, matrix in coefficients.items():
                source_position = position - (lag - 1)
                source_values = model_returns.iloc[source_position].fillna(0.0)
                forecast += source_values.dot(matrix)
                counts += matrix.ne(0.0).sum(axis=0).astype(int)
            score_matrix.loc[date] = forecast.where(counts.gt(0))
            incoming_count.loc[date] = counts
            if self.first_score_position is None and score_matrix.loc[date].notna().any():
                self.first_score_position = position

        self.model_returns = model_returns
        self.score_matrix = score_matrix
        self.latest_edge_table = lead_lag_edge_table(coefficients or {})
        self.network_history = (
            pd.DataFrame(network_rows).set_index("Date")
            if network_rows else pd.DataFrame()
        )
        self.signal_data = {}
        for stock in close.columns:
            signals = pd.DataFrame(index=close.index)
            signals["price"] = close[stock]
            signals["return"] = log_returns[stock]
            signals["ModelReturn"] = model_returns[stock]
            signals["LeadLagForecast"] = score_matrix[stock]
            signals["SignalStrength"] = score_matrix[stock]
            signals["IncomingLeaderEdges"] = incoming_count[stock]
            signals["Valid"] = score_matrix[stock].notna()
            signals["Position"] = (
                signals["Valid"] & signals["LeadLagForecast"].gt(0.0)
            ).astype(int)
            signals["Signal"] = np.sign(signals["Position"].diff().fillna(0)).astype(int)
            self.signal_data[stock] = signals

    def generate_signal_strategy(self, stock, mode="backtest"):
        if mode not in {"backtest", "live"}:
            raise ValueError("mode must be 'backtest' or 'live'.")
        if stock not in self.signal_data:
            raise KeyError(f"No signal data for stock={stock}")
        return self.signal_data[stock].tail(1) if mode == "live" else self.signal_data[stock]

    def get_edge_table(self, date=None):
        if not self.edge_snapshots:
            raise RuntimeError("No fitted network snapshots are available.")
        dates = pd.DatetimeIndex(self.edge_snapshots).sort_values()
        if date is None:
            resolved = dates[-1]
        else:
            requested = pd.Timestamp(date)
            if dates.tz is not None and requested.tzinfo is None:
                requested = requested.tz_localize(dates.tz)
            eligible = dates[dates <= requested]
            if eligible.empty:
                raise ValueError("No network snapshot is available on or before date.")
            resolved = eligible[-1]
        return self.edge_snapshots[resolved].copy()

    def leaders_for_target(self, stock, date=None):
        edges = self.get_edge_table(date)
        return edges.loc[edges["Target"].eq(stock)].reset_index(drop=True)

    def plot_network(self, date=None, max_nodes=30, max_edges=100, ax=None):
        edges = self.get_edge_table(date)
        plot_edges = edges[["Source", "Target", "Coefficient"]].rename(
            columns={"Coefficient": "Weight"}
        )
        return plot_directed_graph(
            plot_edges,
            max_nodes=max_nodes,
            max_edges=max_edges,
            title="Dynamic Lead-Lag Network",
            ax=ax,
        )

    def run_all(self, mode="backtest"):
        self._generate_all_signals()
        self.calculate_returns()
        return self.signal_data
