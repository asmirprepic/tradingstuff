from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseTopNStrategy(ABC):
    """Shared portfolio construction and evaluation for long-only top-N strategies."""

    def __init__(self, data, strategy_name, top_n=10, rebalance_frequency="W-FRI",
                 price_type="Close", transaction_cost=0.0, evaluation_start=None):
        self.strategy_name = strategy_name
        self.data = data
        self.top_n = int(top_n)
        self.rebalance_frequency = rebalance_frequency
        self.price_type = price_type
        self.transaction_cost = float(transaction_cost)
        self.requested_evaluation_start = (
            None if evaluation_start is None else pd.Timestamp(evaluation_start)
        )
        self._validate_inputs()
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        if self.top_n > len(self.stocks_in_data):
            raise ValueError("top_n cannot exceed the number of stocks in the dataset.")

        self.agent = None
        self.score_matrix = pd.DataFrame()
        self.eligibility_matrix = pd.DataFrame()
        self.target_weights = pd.DataFrame()
        self.effective_weights = pd.DataFrame()
        self.selection_log = pd.DataFrame()
        self.portfolio_returns = pd.Series(dtype=float)
        self.equity_curve = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)
        self.benchmark_equity_curve = pd.Series(dtype=float)
        self.evaluation_start = None

    def _validate_inputs(self):
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            raise ValueError("data must be a non-empty pandas DataFrame.")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("data must use a DatetimeIndex.")
        if not self.data.index.is_monotonic_increasing or not self.data.index.is_unique:
            raise ValueError("data index must be sorted and unique.")
        if not isinstance(self.data.columns, pd.MultiIndex) or self.data.columns.nlevels < 2:
            raise ValueError("data must have (ticker, field) MultiIndex columns.")
        if self.top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost cannot be negative.")
        if self.price_type not in self.data.columns.get_level_values(1):
            raise ValueError(f"Price field {self.price_type!r} was not found in data.")
        try:
            pd.tseries.frequencies.to_offset(self.rebalance_frequency)
        except (TypeError, ValueError) as exc:
            raise ValueError("rebalance_frequency must be a valid pandas period frequency.") from exc

    def _rebalance_dates(self):
        indexed_dates = pd.Series(self.data.index, index=self.data.index)
        dates = indexed_dates.resample(self.rebalance_frequency).last().dropna()
        return pd.DatetimeIndex(dates.tolist())

    def _resolve_evaluation_start(self, natural_start):
        if self.requested_evaluation_start is None:
            return natural_start

        index = self.data.index
        try:
            position = index.searchsorted(self.requested_evaluation_start, side="left")
        except TypeError as exc:
            raise ValueError(
                "evaluation_start must have the same timezone convention as the data index."
            ) from exc
        if position >= len(index):
            raise ValueError("evaluation_start is later than the final data row.")

        resolved = index[position]
        if natural_start is None or resolved < natural_start:
            required = natural_start if natural_start is not None else "unavailable"
            raise ValueError(
                f"evaluation_start resolves to {resolved}, but this configuration's "
                f"first executable date is {required}. Provide more pre-start history "
                "or choose a later evaluation_start."
            )
        return resolved

    def _build_target_weights(self, minimum_history):
        if self.score_matrix.empty or self.eligibility_matrix.empty:
            raise RuntimeError("Signal matrices must be populated before building weights.")
        index = self.data.index
        weights = pd.DataFrame(np.nan, index=index, columns=self.stocks_in_data, dtype=float)
        valid_dates = [
            date for date in self._rebalance_dates() if index.get_loc(date) >= minimum_history
        ]
        for date in valid_dates:
            weights.loc[date] = 0.0
            scores = self.score_matrix.loc[date].where(self.eligibility_matrix.loc[date]).dropna()
            selected = scores.nlargest(self.top_n).index
            weights.loc[date, selected] = 1.0 / self.top_n

        self.target_weights = weights.ffill().fillna(0.0)
        self.effective_weights = self.target_weights.shift(1).fillna(0.0)
        natural_start = None
        if valid_dates:
            first_position = index.get_loc(valid_dates[0]) + 1
            if first_position < len(index):
                natural_start = index[first_position]
        self.evaluation_start = self._resolve_evaluation_start(natural_start)
        return self.target_weights

    @abstractmethod
    def generate_target_weights(self):
        """Generate signal matrices and convert them to target weights."""

    def calculate_returns(self):
        if self.target_weights.empty:
            raise RuntimeError("Generate target weights before calculating returns.")
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        daily_returns = close.pct_change(fill_method=None)
        gross_returns = (daily_returns * self.effective_weights).sum(axis=1, min_count=1).fillna(0.0)
        turnover = self.effective_weights.diff().abs().sum(axis=1).fillna(0.0)
        self.portfolio_returns = gross_returns - turnover * self.transaction_cost

        benchmark_returns = pd.Series(0.0, index=close.index, dtype=float)
        if self.evaluation_start is not None:
            evaluation_close = close.loc[self.evaluation_start:]
            start_prices = evaluation_close.iloc[0].replace(0.0, np.nan).dropna()
            benchmark_equity = evaluation_close[start_prices.index].div(start_prices).mean(axis=1)
            benchmark_returns.loc[benchmark_equity.index] = benchmark_equity.pct_change().fillna(0.0)
        self.benchmark_returns = benchmark_returns

        if self.evaluation_start is not None:
            self.portfolio_returns.loc[self.portfolio_returns.index < self.evaluation_start] = 0.0
        else:
            self.portfolio_returns[:] = 0.0
        self.equity_curve = (1.0 + self.portfolio_returns).cumprod()
        self.benchmark_equity_curve = (1.0 + self.benchmark_returns).cumprod()
        rebalance_set = set(self._rebalance_dates())
        self.selection_log = pd.DataFrame({
            "Selected_Stocks": self.target_weights.apply(
                lambda row: list(row.index[row.gt(0.0)]), axis=1
            ),
            "InvestedWeight": self.target_weights.sum(axis=1),
            "N_Held": self.target_weights.gt(0.0).sum(axis=1).astype(int),
            "Rebalance": [date in rebalance_set for date in self.data.index],
            "Turnover": turnover,
            "PortfolioReturn": self.portfolio_returns,
            "Equity": self.equity_curve,
            "BenchmarkEquity": self.benchmark_equity_curve,
        }, index=self.data.index)
        return self.portfolio_returns

    @staticmethod
    def _performance_metrics(returns):
        returns = returns.dropna()
        if returns.empty:
            return {name: np.nan for name in (
                "Total_Return_%", "Annualized_Return_%", "Annualized_Volatility_%",
                "Sharpe_Ratio", "Max_Drawdown_%"
            )}
        equity = (1.0 + returns).cumprod()
        volatility = returns.std(ddof=1)
        return {
            "Total_Return_%": (equity.iloc[-1] - 1.0) * 100.0,
            "Annualized_Return_%": (equity.iloc[-1] ** (252.0 / len(returns)) - 1.0) * 100.0,
            "Annualized_Volatility_%": volatility * np.sqrt(252.0) * 100.0,
            "Sharpe_Ratio": returns.mean() / volatility * np.sqrt(252.0) if volatility > 0 else np.nan,
            "Max_Drawdown_%": (equity / equity.cummax() - 1.0).min() * 100.0,
        }

    def evaluate_performance(self):
        if self.equity_curve.empty:
            raise RuntimeError("Run the strategy before evaluating performance.")
        if self.evaluation_start is None:
            return pd.DataFrame()
        strategy = self._performance_metrics(self.portfolio_returns.loc[self.evaluation_start:])
        benchmark = self._performance_metrics(self.benchmark_returns.loc[self.evaluation_start:])
        result = pd.DataFrame([
            {"Portfolio": self.strategy_name, **strategy},
            {"Portfolio": "EqualWeightBuyHold", **benchmark},
        ])
        result["Outperformance_%"] = result["Total_Return_%"] - benchmark["Total_Return_%"]
        return result.round(4)

    def run_all(self):
        self.generate_target_weights()
        self.calculate_returns()
        return self
