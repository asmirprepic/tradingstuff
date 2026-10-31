import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents.base_agents.trading_agent import TradingAgent
from utilities.eigenportfolio_utils import (
    cap_and_normalize_long_weights,
    estimate_shrunk_eigenportfolios,
    long_only_component_sleeves,
)


class EigenportfolioRotationAgent(TradingAgent):
    """Rotate among momentum-ranked long-only sleeves of correlation eigenportfolios."""

    def __init__(
        self,
        data,
        estimation_window=252,
        component_count=8,
        include_first_component=False,
        short_score_window=20,
        long_score_window=60,
        selected_sleeves=3,
        component_refit_frequency="monthly",
        decision_frequency="W-FRI",
        minimum_coverage=0.90,
        maximum_stock_weight=0.10,
        price_type="Close",
        auto_generate=True,
    ):
        super().__init__(data)
        self.algorithm_name = "EigenportfolioRotation"
        self.score_column = "TargetWeight"
        self.estimation_window = int(estimation_window)
        self.component_count = int(component_count)
        self.include_first_component = bool(include_first_component)
        self.short_score_window = int(short_score_window)
        self.long_score_window = int(long_score_window)
        self.selected_sleeves = int(selected_sleeves)
        self.component_refit_frequency = str(component_refit_frequency).lower()
        self.decision_frequency = decision_frequency
        self.minimum_coverage = float(minimum_coverage)
        self.maximum_stock_weight = float(maximum_stock_weight)
        self.price_type = price_type
        self._validate_parameters()

        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        self.target_weight_matrix = pd.DataFrame()
        self.sleeve_returns = pd.DataFrame()
        self.sleeve_score_history = pd.DataFrame()
        self.component_history = pd.DataFrame()
        self.selected_sleeves_history = pd.DataFrame()
        self.component_weight_snapshots = {}
        self.latest_eigenvectors = pd.DataFrame()
        self.latest_sleeve_weights = pd.DataFrame()
        self.first_weight_position = None
        if auto_generate:
            self.run_all()

    def _validate_parameters(self):
        if self.estimation_window < 20:
            raise ValueError("estimation_window must be at least 20.")
        if self.component_count < 1:
            raise ValueError("component_count must be positive.")
        if self.short_score_window < 2:
            raise ValueError("short_score_window must be at least 2.")
        if self.long_score_window < self.short_score_window:
            raise ValueError("long_score_window cannot be shorter than short_score_window.")
        if self.selected_sleeves < 1:
            raise ValueError("selected_sleeves must be positive.")
        if self.component_refit_frequency not in {"monthly", "weekly", "daily"}:
            raise ValueError("component_refit_frequency must be monthly, weekly, or daily.")
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be in (0, 1].")
        if not 0.0 < self.maximum_stock_weight <= 1.0:
            raise ValueError("maximum_stock_weight must be in (0, 1].")
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
        if self.component_refit_frequency == "daily":
            return date
        if self.component_refit_frequency == "weekly":
            iso = date.isocalendar()
            return int(iso.year), int(iso.week)
        return date.year, date.month

    @staticmethod
    def _risk_adjusted_score(returns, short_window, long_window):
        clean = returns.dropna()
        if len(clean) < long_window:
            return np.nan
        short = clean.iloc[-short_window:]
        long = clean.iloc[-long_window:]

        def ratio(sample):
            volatility = sample.std(ddof=1)
            return sample.mean() / volatility if volatility > 1e-12 else np.nan

        short_score = ratio(short)
        long_score = ratio(long)
        if not np.isfinite(short_score) or not np.isfinite(long_score):
            return np.nan
        return 0.5 * short_score + 0.5 * long_score

    def _generate_all_signals(self):
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        daily_returns = close.pct_change(fill_method=None)
        index = close.index
        stocks = close.columns
        decision_dates = set(self._decision_dates())
        target_weights = pd.DataFrame(np.nan, index=index, columns=stocks)
        sleeve_return_rows = []
        score_rows = []
        component_rows = []
        selection_rows = []
        current_sleeves = None
        previous_eigenvectors = None
        refit_key = None

        for position, date in enumerate(index):
            if current_sleeves is not None:
                sleeve_return = current_sleeves.mul(daily_returns.iloc[position], axis=0).sum()
                sleeve_return.name = date
                sleeve_return_rows.append(sleeve_return)

            if position >= self.estimation_window - 1:
                current_refit_key = self._refit_key(date)
                if current_sleeves is None or current_refit_key != refit_key:
                    start = position - self.estimation_window + 1
                    estimate = estimate_shrunk_eigenportfolios(
                        daily_returns.iloc[start:position + 1],
                        component_count=self.component_count,
                        minimum_coverage=self.minimum_coverage,
                        previous_eigenvectors=previous_eigenvectors,
                    )
                    previous_eigenvectors = estimate["eigenvectors"]
                    current_sleeves = long_only_component_sleeves(
                        estimate["tradable_loadings"]
                    )
                    self.component_weight_snapshots[date] = current_sleeves.copy()
                    for component in estimate["eigenvalues"].index:
                        component_rows.append({
                            "Date": date,
                            "Component": component,
                            "Eigenvalue": estimate["eigenvalues"].loc[component],
                            "ExplainedVariance": estimate["explained_variance"].loc[component],
                        })
                    refit_key = current_refit_key

            if date not in decision_dates or current_sleeves is None:
                continue
            sleeve_returns = (
                pd.DataFrame(sleeve_return_rows)
                if sleeve_return_rows
                else pd.DataFrame()
            )
            scores = pd.Series(
                {
                    sleeve: self._risk_adjusted_score(
                        sleeve_returns.get(sleeve, pd.Series(dtype=float)),
                        self.short_score_window,
                        self.long_score_window,
                    )
                    for sleeve in current_sleeves.columns
                },
                dtype=float,
            )
            if not self.include_first_component:
                scores = scores.loc[~scores.index.str.startswith("PC1_")]
            positive_scores = scores.dropna().loc[lambda values: values.gt(0.0)]
            best_sides = []
            components = positive_scores.index.str.rsplit("_", n=1).str[0].unique()
            for component in components:
                component_scores = positive_scores.loc[
                    positive_scores.index.str.startswith(f"{component}_")
                ]
                best_sides.append(component_scores.idxmax())
            component_candidates = positive_scores.reindex(best_sides).dropna()
            selected = component_candidates.nlargest(
                self.selected_sleeves
            ).index.tolist()
            target_weights.loc[date] = 0.0
            if selected:
                combined = current_sleeves[selected].mean(axis=1)
                target_weights.loc[date] = cap_and_normalize_long_weights(
                    combined,
                    maximum_weight=self.maximum_stock_weight,
                )
            score_rows.append(pd.Series(scores, name=date))
            selection_rows.append({
                "Date": date,
                "SelectedSleeves": selected,
                "BestSleeveScore": (
                    component_candidates.max()
                    if not component_candidates.empty
                    else np.nan
                ),
            })
            if self.first_weight_position is None and scores.notna().any():
                self.first_weight_position = position

        self.target_weight_matrix = target_weights
        self.sleeve_returns = (
            pd.DataFrame(sleeve_return_rows).sort_index()
            if sleeve_return_rows
            else pd.DataFrame(index=index)
        )
        self.sleeve_score_history = (
            pd.DataFrame(score_rows).sort_index() if score_rows else pd.DataFrame()
        )
        self.component_history = (
            pd.DataFrame(component_rows).set_index(["Date", "Component"])
            if component_rows
            else pd.DataFrame(columns=["Eigenvalue", "ExplainedVariance"])
        )
        self.selected_sleeves_history = (
            pd.DataFrame(selection_rows).set_index("Date")
            if selection_rows
            else pd.DataFrame(columns=["SelectedSleeves", "BestSleeveScore"])
        )
        self.latest_eigenvectors = (
            previous_eigenvectors if previous_eigenvectors is not None else pd.DataFrame()
        )
        self.latest_sleeve_weights = (
            current_sleeves if current_sleeves is not None else pd.DataFrame()
        )
        self.signal_data = {}
        carried_weights = target_weights.ffill().fillna(0.0)
        for stock in stocks:
            signals = pd.DataFrame(index=index)
            signals["price"] = close[stock]
            signals["return"] = np.log(close[stock] / close[stock].shift(1))
            signals["TargetWeight"] = carried_weights[stock]
            signals["SignalStrength"] = carried_weights[stock]
            signals["Position"] = carried_weights[stock].gt(0.0).astype(int)
            signals["Signal"] = np.sign(signals["Position"].diff().fillna(0)).astype(int)
            self.signal_data[stock] = signals

    @staticmethod
    def _align_timestamp(value, index):
        timestamp = pd.Timestamp(value)
        if index.tz is not None and timestamp.tzinfo is None:
            return timestamp.tz_localize(index.tz)
        if index.tz is None and timestamp.tzinfo is not None:
            return timestamp.tz_localize(None)
        return timestamp

    def get_component_weights(self, date=None):
        """Return long-only component sleeves from the latest applicable refit."""
        if not self.component_weight_snapshots:
            raise RuntimeError("No component weight snapshots are available.")
        dates = pd.DatetimeIndex(self.component_weight_snapshots.keys()).sort_values()
        if date is None:
            resolved = dates[-1]
        else:
            requested = self._align_timestamp(date, dates)
            eligible = dates[dates <= requested]
            if eligible.empty:
                raise ValueError("No component snapshot is available on or before date.")
            resolved = eligible[-1]
        return self.component_weight_snapshots[resolved].copy()

    def component_summary(self, date=None):
        """Return eigenvalues and explained variance for one component refit."""
        if self.component_history.empty:
            raise RuntimeError("No component history is available.")
        dates = self.component_history.index.get_level_values("Date").unique().sort_values()
        if date is None:
            resolved = dates[-1]
        else:
            requested = self._align_timestamp(date, dates)
            eligible = dates[dates <= requested]
            if eligible.empty:
                raise ValueError("No component estimate is available on or before date.")
            resolved = eligible[-1]
        return self.component_history.xs(resolved, level="Date").copy()

    def plot_explained_variance(self, date=None, ax=None):
        summary = self.component_summary(date)
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4.5))
        else:
            fig = ax.figure
        ax.bar(summary.index, summary["ExplainedVariance"] * 100.0, color="#2f6f73")
        ax.set_ylabel("Explained variance (%)")
        ax.set_title("Eigenportfolio Variance Structure")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig, ax

    def plot_target_weights(self, date=None, max_stocks=25, ax=None):
        available = self.target_weight_matrix.index[
            self.target_weight_matrix.notna().any(axis=1)
        ]
        if available.empty:
            raise ValueError("No target portfolio weights are available.")
        resolved = (
            available[-1]
            if date is None
            else self._align_timestamp(date, self.target_weight_matrix.index)
        )
        if resolved not in available:
            raise ValueError("The requested date has no portfolio decision.")
        weights = self.target_weight_matrix.loc[resolved].dropna().nlargest(int(max_stocks))
        weights = weights.loc[weights.gt(0.0)].sort_values()
        if weights.empty:
            raise ValueError("The requested portfolio is entirely in cash.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(weights))))
        else:
            fig = ax.figure
        ax.barh(weights.index, weights * 100.0, color="#b25b3f")
        ax.set_xlabel("Target weight (%)")
        ax.set_title(f"Eigenportfolio Rotation Weights ({resolved.date()})")
        fig.tight_layout()
        return fig, ax

    def generate_signal_strategy(self, stock, mode="backtest"):
        if mode not in {"backtest", "live"}:
            raise ValueError("mode must be 'backtest' or 'live'.")
        if stock not in self.signal_data:
            raise KeyError(f"No signal data for stock={stock}")
        if mode == "live":
            return self.signal_data[stock].tail(1)
        return self.signal_data[stock]

    def run_all(self, mode="backtest"):
        self._generate_all_signals()
        self.calculate_returns()
        return self.signal_data
