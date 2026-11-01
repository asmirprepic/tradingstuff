import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents.base_agents.trading_agent import TradingAgent
from utilities.shock_propagation_utils import (
    estimate_sparse_lag_network,
    lag_network_edge_table,
    residualize_common_effects,
)
from utilities.graph_plotting import plot_directed_graph


class GraphShockPropagationAgent(TradingAgent):
    """Directed sparse-lag network forecasts triggered by idiosyncratic shocks."""

    def __init__(
        self,
        data,
        estimation_window=252,
        max_lag=3,
        lasso_alpha=0.0005,
        shock_lookback=60,
        shock_z_threshold=2.0,
        minimum_observations=126,
        minimum_source_coverage=0.80,
        network_refit_frequency="monthly",
        decision_frequency="B",
        sector_map=None,
        minimum_sector_size=3,
        price_type="Close",
        auto_generate=True,
    ):
        super().__init__(data)
        self.algorithm_name = "GraphShockPropagation"
        self.score_column = "PropagationScore"
        self.estimation_window = int(estimation_window)
        self.max_lag = int(max_lag)
        self.lasso_alpha = float(lasso_alpha)
        self.shock_lookback = int(shock_lookback)
        self.shock_z_threshold = float(shock_z_threshold)
        self.minimum_observations = int(minimum_observations)
        self.minimum_source_coverage = float(minimum_source_coverage)
        self.network_refit_frequency = str(network_refit_frequency).lower()
        self.decision_frequency = decision_frequency
        self.sector_map = sector_map
        self.minimum_sector_size = int(minimum_sector_size)
        self.price_type = price_type
        self._validate_parameters()

        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        self.adjusted_returns = pd.DataFrame()
        self.adjustment_mode = pd.Series(dtype=object)
        self.latest_coefficients = {}
        self.latest_edge_table = pd.DataFrame()
        self.network_history = pd.DataFrame()
        self.edge_snapshots = {}
        self.first_score_position = None
        if auto_generate:
            self.run_all()

    def _validate_parameters(self):
        if self.estimation_window < 20:
            raise ValueError("estimation_window must be at least 20.")
        if self.max_lag < 1:
            raise ValueError("max_lag must be positive.")
        if self.max_lag >= self.estimation_window:
            raise ValueError("max_lag must be smaller than estimation_window.")
        if self.lasso_alpha <= 0.0:
            raise ValueError("lasso_alpha must be positive.")
        if self.shock_lookback < 2:
            raise ValueError("shock_lookback must be at least 2.")
        if self.shock_z_threshold <= 0.0:
            raise ValueError("shock_z_threshold must be positive.")
        if not 2 <= self.minimum_observations < self.estimation_window:
            raise ValueError("minimum_observations must be between 2 and estimation_window.")
        if not 0.0 < self.minimum_source_coverage <= 1.0:
            raise ValueError("minimum_source_coverage must be in (0, 1].")
        if self.network_refit_frequency not in {"monthly", "weekly", "daily"}:
            raise ValueError("network_refit_frequency must be monthly, weekly, or daily.")
        if self.minimum_sector_size < 2:
            raise ValueError("minimum_sector_size must be at least 2.")
        try:
            pd.tseries.frequencies.to_offset(self.decision_frequency)
        except (TypeError, ValueError) as exc:
            raise ValueError("decision_frequency must be a valid pandas frequency.") from exc

    def _decision_dates(self):
        indexed_dates = pd.Series(self.data.index, index=self.data.index)
        grouped = indexed_dates.resample(self.decision_frequency).last().dropna()
        completed = grouped.index <= self.data.index[-1]
        return pd.DatetimeIndex(grouped.loc[completed].tolist())

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
        adjusted, adjustment_mode = residualize_common_effects(
            log_returns,
            sector_map=self.sector_map,
            minimum_sector_size=self.minimum_sector_size,
        )
        scale = adjusted.shift(1).rolling(
            self.shock_lookback,
            min_periods=self.shock_lookback,
        ).std(ddof=1)
        shock_z = adjusted.div(scale.where(scale > 1e-12))
        shock_flags = shock_z.abs().ge(self.shock_z_threshold)

        index = close.index
        stocks = close.columns
        decision_dates = set(self._decision_dates())
        score_matrix = pd.DataFrame(np.nan, index=index, columns=stocks)
        incoming_count = pd.DataFrame(0, index=index, columns=stocks, dtype=int)
        network_rows = []
        coefficients = None
        refit_key = None
        lags = tuple(range(1, self.max_lag + 1))
        minimum_position = max(self.estimation_window - 1, self.shock_lookback)

        for position, date in enumerate(index):
            if date not in decision_dates or position < minimum_position:
                continue
            current_refit_key = self._refit_key(date)
            if coefficients is None or current_refit_key != refit_key:
                start = position - self.estimation_window + 1
                training_window = adjusted.iloc[start:position + 1]
                coefficients = estimate_sparse_lag_network(
                    training_window,
                    lags=lags,
                    alpha=self.lasso_alpha,
                    minimum_observations=self.minimum_observations,
                    minimum_source_coverage=self.minimum_source_coverage,
                )
                edge_table = lag_network_edge_table(coefficients)
                self.edge_snapshots[date] = edge_table.copy()
                network_rows.append({
                    "Date": date,
                    "Edges": len(edge_table),
                    "Sources": edge_table["Source"].nunique() if len(edge_table) else 0,
                    "Targets": edge_table["Target"].nunique() if len(edge_table) else 0,
                })
                refit_key = current_refit_key

            forecast = pd.Series(0.0, index=stocks, dtype=float)
            counts = pd.Series(0, index=stocks, dtype=int)
            for lag, matrix in coefficients.items():
                source_position = position - (lag - 1)
                source_values = adjusted.iloc[source_position].where(
                    shock_flags.iloc[source_position], 0.0
                ).fillna(0.0)
                forecast += source_values.dot(matrix)
                active_sources = source_values.ne(0.0).astype(int)
                counts += matrix.ne(0.0).mul(active_sources, axis=0).sum(axis=0).astype(int)

            score_matrix.loc[date] = forecast.where(counts.gt(0))
            incoming_count.loc[date] = counts
            if self.first_score_position is None and score_matrix.loc[date].notna().any():
                self.first_score_position = position

        self.adjusted_returns = adjusted
        self.adjustment_mode = adjustment_mode
        self.latest_coefficients = coefficients or {}
        self.latest_edge_table = lag_network_edge_table(self.latest_coefficients)
        self.network_history = (
            pd.DataFrame(network_rows).set_index("Date")
            if network_rows
            else pd.DataFrame(columns=["Edges", "Sources", "Targets"])
        )
        self.signal_data = {}
        for stock in stocks:
            signals = pd.DataFrame(index=index)
            signals["price"] = close[stock]
            signals["return"] = log_returns[stock]
            signals["AdjustedReturn"] = adjusted[stock]
            signals["ShockZ"] = shock_z[stock]
            signals["IsShock"] = shock_flags[stock]
            signals["AdjustmentMode"] = adjustment_mode.loc[stock]
            signals["PredictedResidualReturn"] = score_matrix[stock]
            signals["PropagationScore"] = score_matrix[stock]
            signals["IncomingShockCount"] = incoming_count[stock]
            signals["Valid"] = signals["PropagationScore"].notna()
            signals["Position"] = np.where(
                signals["Valid"] & signals["PropagationScore"].gt(0.0), 1, 0
            ).astype(int)
            signals["Signal"] = np.sign(signals["Position"].diff().fillna(0)).astype(int)
            self.signal_data[stock] = signals

    def generate_signal_strategy(self, stock, mode="backtest"):
        if mode not in {"backtest", "live"}:
            raise ValueError("mode must be 'backtest' or 'live'.")
        if stock not in self.signal_data:
            raise KeyError(f"No signal data for stock={stock}")
        if mode == "live":
            return self.signal_data[stock].tail(1)
        return self.signal_data[stock]

    @staticmethod
    def _align_timestamp(value, index):
        timestamp = pd.Timestamp(value)
        if index.tz is not None and timestamp.tzinfo is None:
            return timestamp.tz_localize(index.tz)
        if index.tz is None and timestamp.tzinfo is not None:
            return timestamp.tz_localize(None)
        return timestamp

    def get_edge_table(self, date=None):
        """Return the network fitted on or most recently before a requested date."""
        if not self.edge_snapshots:
            raise RuntimeError("No fitted network snapshots are available.")
        snapshot_dates = pd.DatetimeIndex(self.edge_snapshots.keys()).sort_values()
        if date is None:
            resolved = snapshot_dates[-1]
        else:
            requested = self._align_timestamp(date, snapshot_dates)
            eligible = snapshot_dates[snapshot_dates <= requested]
            if eligible.empty:
                raise ValueError("No network snapshot is available on or before date.")
            resolved = eligible[-1]
        return self.edge_snapshots[resolved].copy()

    def _resolve_forecast_date(self, date=None):
        score_frame = pd.DataFrame(
            {stock: signals["PropagationScore"] for stock, signals in self.signal_data.items()}
        )
        available = score_frame.index[score_frame.notna().any(axis=1)]
        if available.empty:
            raise ValueError("No propagation forecast dates are available.")
        resolved = (
            available[-1]
            if date is None
            else self._align_timestamp(date, score_frame.index)
        )
        if resolved not in available:
            raise ValueError("The requested date has no propagation forecasts.")
        return resolved

    def propagation_contributions(self, stock, date=None):
        """Explain a target forecast as active directed-edge contributions."""
        if stock not in self.signal_data:
            raise KeyError(f"No signal data for stock={stock}")
        scores = self.signal_data[stock]["PropagationScore"].dropna()
        if scores.empty:
            raise ValueError(f"No propagation forecasts are available for stock={stock}.")
        resolved = (
            scores.index[-1]
            if date is None
            else self._align_timestamp(date, scores.index)
        )
        if resolved not in scores.index:
            raise ValueError("The requested date has no propagation forecast for this stock.")

        edges = self.get_edge_table(resolved)
        edges = edges.loc[edges["Target"].eq(stock)].copy()
        position = self.data.index.get_loc(resolved)
        rows = []
        for edge in edges.itertuples(index=False):
            source_position = position - (edge.Lag - 1)
            if source_position < 0:
                continue
            source_date = self.data.index[source_position]
            source_signal = self.signal_data[edge.Source].loc[source_date]
            if not bool(source_signal["IsShock"]):
                continue
            contribution = float(edge.Coefficient * source_signal["AdjustedReturn"])
            rows.append({
                "Source": edge.Source,
                "Target": stock,
                "Lag": edge.Lag,
                "SourceDate": source_date,
                "Coefficient": edge.Coefficient,
                "AdjustedShock": source_signal["AdjustedReturn"],
                "ShockZ": source_signal["ShockZ"],
                "Contribution": contribution,
            })
        columns = [
            "Source", "Target", "Lag", "SourceDate", "Coefficient",
            "AdjustedShock", "ShockZ", "Contribution",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        result = pd.DataFrame(rows, columns=columns)
        return result.loc[
            result["Contribution"].abs().sort_values(ascending=False).index
        ].reset_index(drop=True)

    def active_propagation_edges(self, date=None):
        """Return every directed edge activated by a shock on a forecast date."""
        resolved = self._resolve_forecast_date(date)
        score_frame = pd.DataFrame(
            {stock: signals["PropagationScore"] for stock, signals in self.signal_data.items()}
        )
        parts = []
        for stock in score_frame.columns[score_frame.loc[resolved].notna()]:
            contributions = self.propagation_contributions(stock, resolved)
            if not contributions.empty:
                parts.append(contributions)
        if not parts:
            return pd.DataFrame(columns=[
                "Source", "Target", "Lag", "SourceDate", "Coefficient",
                "AdjustedShock", "ShockZ", "Contribution",
            ])
        return pd.concat(parts, ignore_index=True).sort_values(
            "Contribution", key=lambda values: values.abs(), ascending=False
        ).reset_index(drop=True)

    def plot_network(
        self,
        date=None,
        active_only=False,
        node_metric=None,
        max_nodes=30,
        max_edges=100,
        minimum_absolute_weight=0.0,
        ax=None,
    ):
        """Plot either the fitted directed network or currently activated edges."""
        if active_only:
            resolved = self._resolve_forecast_date(date)
            active = self.active_propagation_edges(date)
            if active.empty:
                raise ValueError("No shock-activated edges are available on this date.")
            edges = active[["Source", "Target", "Contribution"]].rename(
                columns={"Contribution": "Weight"}
            )
            metric = node_metric or "PropagationScore"
            title = "Active Shock Propagation"
        else:
            structural = self.get_edge_table(date)
            edges = structural[["Source", "Target", "Coefficient"]].rename(
                columns={"Coefficient": "Weight"}
            )
            resolved = self.data.index[-1] if date is None else date
            metric = node_metric or "ShockZ"
            title = "Estimated Directed Lag Network"

        resolved = self._align_timestamp(resolved, self.data.index)
        if metric not in {"ShockZ", "PropagationScore", "AdjustedReturn"}:
            raise ValueError(
                "node_metric must be ShockZ, PropagationScore, or AdjustedReturn."
            )
        node_values = pd.Series(
            {
                stock: signals.loc[resolved, metric]
                for stock, signals in self.signal_data.items()
            },
            name=metric,
            dtype=float,
        )
        return plot_directed_graph(
            edges,
            node_values=node_values,
            max_nodes=max_nodes,
            max_edges=max_edges,
            minimum_absolute_weight=minimum_absolute_weight,
            title=f"{title} ({resolved.date()})",
            ax=ax,
        )

    def plot_propagation_contributions(self, stock, date=None, max_sources=15, ax=None):
        """Plot the signed edge contributions behind one target forecast."""
        contributions = self.propagation_contributions(stock, date).head(int(max_sources))
        if contributions.empty:
            raise ValueError("No active incoming shock contributions are available.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(contributions))))
        else:
            fig = ax.figure
        labels = contributions.apply(
            lambda row: f"{row['Source']} (lag {int(row['Lag'])})", axis=1
        )
        values = contributions["Contribution"]
        colors = np.where(values.ge(0.0), "#17845f", "#c44536")
        positions = np.arange(len(contributions))
        ax.barh(positions, values, color=colors, alpha=0.85)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.axvline(0.0, color="#263238", linewidth=0.8)
        resolved = self._resolve_forecast_date(date)
        ax.set_title(f"Incoming Shock Contributions: {stock} ({resolved.date()})")
        ax.set_xlabel("Predicted residual-return contribution")
        fig.tight_layout()
        return fig, ax

    def run_all(self, mode="backtest"):
        self._generate_all_signals()
        self.calculate_returns()
        return self.signal_data
