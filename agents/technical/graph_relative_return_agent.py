import numpy as np
import pandas as pd

from agents.base_agents.trading_agent import TradingAgent
from utilities.graph_utils import build_correlation_graph, graph_deviation, neighbour_weighted_values
from utilities.graph_plotting import adjacency_to_edge_list, plot_weighted_graph


class GraphRelativeReturnAgent(TradingAgent):
    """Dynamic graph signal with data-driven persistence or mean-reversion direction."""

    def __init__(
        self,
        data,
        graph_window=126,
        top_k=8,
        minimum_correlation=0.25,
        recent_return_window=5,
        prediction_horizon=5,
        calibration_weeks=52,
        minimum_calibration_weeks=12,
        direction_shrinkage=0.10,
        decision_frequency="W-FRI",
        remove_market_mode=True,
        price_type="Close",
        auto_generate=True,
    ):
        super().__init__(data)
        self.algorithm_name = "GraphRelativeReturn"
        self.score_column = "GraphScore"
        self.graph_window = int(graph_window)
        self.top_k = int(top_k)
        self.minimum_correlation = float(minimum_correlation)
        self.recent_return_window = int(recent_return_window)
        self.prediction_horizon = int(prediction_horizon)
        self.calibration_weeks = int(calibration_weeks)
        self.minimum_calibration_weeks = int(minimum_calibration_weeks)
        self.direction_shrinkage = float(direction_shrinkage)
        self.decision_frequency = decision_frequency
        self.remove_market_mode = bool(remove_market_mode)
        self.price_type = price_type
        self._validate_parameters()

        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        if self.top_k >= len(self.stocks_in_data):
            raise ValueError("top_k must be smaller than the number of stocks.")
        self.latest_adjacency = pd.DataFrame()
        self.graph_snapshots = {}
        self.graph_history = pd.DataFrame()
        self.direction_history = pd.DataFrame()
        self.first_score_position = None
        if auto_generate:
            self.run_all()

    def _validate_parameters(self):
        positive = {
            "graph_window": self.graph_window,
            "top_k": self.top_k,
            "recent_return_window": self.recent_return_window,
            "prediction_horizon": self.prediction_horizon,
            "calibration_weeks": self.calibration_weeks,
            "minimum_calibration_weeks": self.minimum_calibration_weeks,
        }
        if any(value < 1 for value in positive.values()):
            raise ValueError("All graph windows, horizons, and counts must be positive.")
        if self.minimum_calibration_weeks > self.calibration_weeks:
            raise ValueError("minimum_calibration_weeks cannot exceed calibration_weeks.")
        if not 0.0 <= self.minimum_correlation <= 1.0:
            raise ValueError("minimum_correlation must be between 0 and 1.")
        if self.direction_shrinkage < 0.0:
            raise ValueError("direction_shrinkage cannot be negative.")
        try:
            pd.tseries.frequencies.to_offset(self.decision_frequency)
        except (TypeError, ValueError) as exc:
            raise ValueError("decision_frequency must be a valid pandas frequency.") from exc

    def _decision_dates(self):
        indexed_dates = pd.Series(self.data.index, index=self.data.index)
        grouped = indexed_dates.resample(self.decision_frequency).last().dropna()
        completed = grouped.index <= self.data.index[-1]
        dates = grouped.loc[completed]
        return pd.DatetimeIndex(dates.tolist())

    def _fit_direction(self, completed):
        sample = completed[-self.calibration_weeks:]
        if len(sample) < self.minimum_calibration_weeks:
            return np.nan, 0
        x = np.concatenate([record["x"] for record in sample])
        y = np.concatenate([record["y"] for record in sample])
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 2:
            return np.nan, 0
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        sum_squares = float(np.dot(x_centered, x_centered))
        if sum_squares <= 1e-16:
            return np.nan, len(x)
        beta = float(np.dot(x_centered, y_centered) / sum_squares)
        beta /= 1.0 + self.direction_shrinkage
        return beta, len(x)

    def _generate_all_signals(self):
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        daily_returns = np.log(close / close.shift(1))
        index = close.index
        stocks = close.columns
        decision_dates = set(self._decision_dates())

        neighbour_matrix = pd.DataFrame(np.nan, index=index, columns=stocks)
        deviation_matrix = pd.DataFrame(np.nan, index=index, columns=stocks)
        score_matrix = pd.DataFrame(np.nan, index=index, columns=stocks)
        neighbour_count_matrix = pd.DataFrame(0, index=index, columns=stocks, dtype=int)
        beta_series = pd.Series(np.nan, index=index, dtype=float)
        mode_series = pd.Series("UNAVAILABLE", index=index, dtype=object)
        pending = []
        completed = []
        graph_rows = []
        direction_rows = []
        adjacency = None
        graph_month = None
        self.graph_snapshots = {}

        minimum_position = max(self.graph_window, self.recent_return_window)
        for position, date in enumerate(index):
            if date not in decision_dates or position < minimum_position:
                continue

            still_pending = []
            for record in pending:
                if record["maturity_position"] <= position:
                    future_return = np.log(
                        close.iloc[record["maturity_position"]] / close.iloc[record["position"]]
                    )
                    outcome = graph_deviation(record["adjacency"], future_return)
                    completed.append({
                        "date": record["date"],
                        "x": record["deviation"].to_numpy(dtype=float),
                        "y": outcome.to_numpy(dtype=float),
                    })
                else:
                    still_pending.append(record)
            pending = still_pending

            beta, training_samples = self._fit_direction(completed)
            month = (date.year, date.month)
            if adjacency is None or month != graph_month:
                graph_returns = daily_returns.iloc[position - self.graph_window:position]
                adjacency = build_correlation_graph(
                    graph_returns,
                    top_k=self.top_k,
                    minimum_correlation=self.minimum_correlation,
                    remove_market_mode=self.remove_market_mode,
                )
                graph_month = month
                self.graph_snapshots[date] = adjacency_to_edge_list(adjacency)
                edge_count = int(np.count_nonzero(np.triu(adjacency.to_numpy(), k=1)))
                graph_rows.append({
                    "Date": date,
                    "Edges": edge_count,
                    "ConnectedStocks": int(adjacency.sum(axis=1).gt(0.0).sum()),
                    "AverageDegree": float(adjacency.gt(0.0).sum(axis=1).mean()),
                })

            recent_return = np.log(
                close.iloc[position] / close.iloc[position - self.recent_return_window]
            )
            neighbour_return = neighbour_weighted_values(adjacency, recent_return)
            deviation = graph_deviation(adjacency, recent_return)
            score = deviation * beta if np.isfinite(beta) else deviation * np.nan
            neighbour_matrix.loc[date] = neighbour_return
            deviation_matrix.loc[date] = deviation
            score_matrix.loc[date] = score
            neighbour_count_matrix.loc[date] = adjacency.gt(0.0).sum(axis=1).astype(int)
            beta_series.loc[date] = beta
            if np.isfinite(beta):
                mode_series.loc[date] = "PERSISTENCE" if beta > 0.0 else "MEAN_REVERSION"
                if self.first_score_position is None:
                    self.first_score_position = position
            direction_rows.append({
                "Date": date,
                "DirectionBeta": beta,
                "DirectionMode": mode_series.loc[date],
                "CalibrationWeeks": len(completed[-self.calibration_weeks:]),
                "TrainingSamples": training_samples,
            })

            maturity_position = position + self.prediction_horizon
            if maturity_position < len(index):
                pending.append({
                    "date": date,
                    "position": position,
                    "maturity_position": maturity_position,
                    "deviation": deviation,
                    "adjacency": adjacency,
                })

        self.latest_adjacency = adjacency if adjacency is not None else pd.DataFrame()
        self.graph_history = pd.DataFrame(graph_rows).set_index("Date") if graph_rows else pd.DataFrame()
        self.direction_history = (
            pd.DataFrame(direction_rows).set_index("Date") if direction_rows else pd.DataFrame()
        )
        self.signal_data = {}
        for stock in stocks:
            signals = pd.DataFrame(index=index)
            signals["price"] = close[stock]
            signals["return"] = daily_returns[stock]
            signals["NeighbourReturn"] = neighbour_matrix[stock]
            signals["GraphDeviation"] = deviation_matrix[stock]
            signals["DirectionBeta"] = beta_series
            signals["DirectionMode"] = mode_series
            signals["GraphScore"] = score_matrix[stock]
            signals["SignalStrength"] = score_matrix[stock]
            signals["NeighbourCount"] = neighbour_count_matrix[stock]
            signals["Valid"] = signals["GraphScore"].notna()
            signals["Position"] = np.where(
                signals["Valid"] & signals["GraphScore"].gt(0.0), 1, 0
            ).astype(int)
            signals["Signal"] = np.sign(signals["Position"].diff().fillna(0)).astype(int)
            self.signal_data[stock] = signals

    def generate_signal_strategy(self, stock, mode="backtest"):
        if mode not in {"backtest", "live"}:
            raise ValueError("mode must be 'backtest' or 'live'.")
        if not self.signal_data:
            self._generate_all_signals()
        if stock not in self.signal_data:
            raise KeyError(f"Stock {stock} not found in signal data.")
        return self.signal_data[stock]

    def run_all(self, mode="backtest"):
        self.first_score_position = None
        self._generate_all_signals()
        self.calculate_returns()

    def _resolve_graph_snapshot_date(self, date=None):
        if not self.graph_snapshots:
            raise RuntimeError("No graph snapshots are available. Run the agent first.")
        dates = pd.DatetimeIndex(self.graph_snapshots.keys()).sort_values()
        if date is None:
            return dates[-1]
        requested = pd.Timestamp(date)
        if dates.tz is not None and requested.tzinfo is None:
            requested = requested.tz_localize(dates.tz)
        elif dates.tz is None and requested.tzinfo is not None:
            requested = requested.tz_localize(None)
        eligible = dates[dates <= requested]
        if eligible.empty:
            raise ValueError("No graph snapshot exists on or before the requested date.")
        return eligible[-1]

    def plot_graph(self, date=None, node_metric="GraphScore", max_nodes=40,
                   max_edges=250, minimum_weight=0.0, ax=None):
        snapshot_date = self._resolve_graph_snapshot_date(date)
        node_values = None
        if node_metric is not None:
            values = {}
            for stock, signals in self.signal_data.items():
                if node_metric not in signals.columns:
                    raise ValueError(f"Unknown node_metric: {node_metric}")
                values[stock] = signals.at[snapshot_date, node_metric]
            node_values = pd.Series(values, name=node_metric, dtype=float)
        return plot_weighted_graph(
            self.graph_snapshots[snapshot_date],
            node_values=node_values,
            max_nodes=max_nodes,
            max_edges=max_edges,
            minimum_weight=minimum_weight,
            title=f"Equity Graph - {snapshot_date.date()}",
            ax=ax,
        )

    def plot_direction_history(self, ax=None):
        import matplotlib.pyplot as plt

        if self.direction_history.empty:
            raise RuntimeError("No direction history is available. Run the agent first.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        else:
            fig = ax.figure
        beta = self.direction_history["DirectionBeta"]
        ax.plot(beta.index, beta, color="#145a6a", linewidth=2.0)
        ax.axhline(0.0, color="#202020", linewidth=1.0)
        ax.fill_between(beta.index, 0.0, beta, where=beta.ge(0.0), color="#2e8b57", alpha=0.25)
        ax.fill_between(beta.index, 0.0, beta, where=beta.lt(0.0), color="#c44e52", alpha=0.25)
        ax.set_title("Graph Direction: Persistence (+) vs Mean Reversion (-)")
        ax.set_ylabel("Direction beta")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig, ax

    def plot_stock_signal(self, stock, ax=None):
        import matplotlib.pyplot as plt

        if stock not in self.signal_data:
            raise KeyError(f"Stock {stock} not found in signal data.")
        signals = self.signal_data[stock].dropna(subset=["GraphDeviation"])
        if signals.empty:
            raise ValueError(f"No graph observations are available for {stock}.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure
        ax.plot(signals.index, signals["GraphDeviation"], label="Graph deviation", color="#c44e52")
        ax.plot(signals.index, signals["NeighbourReturn"], label="Neighbour return", color="#4c72b0")
        ax.axhline(0.0, color="#202020", linewidth=0.8)
        ax.set_ylabel("Log return")
        score_axis = ax.twinx()
        score_axis.plot(signals.index, signals["GraphScore"], label="Graph score", color="#2e8b57")
        score_axis.set_ylabel("Graph score")
        lines = ax.get_lines()[:2] + score_axis.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="best")
        ax.set_title(f"Graph Signal - {stock}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig, (ax, score_axis)
