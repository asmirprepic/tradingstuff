import numpy as np
import pandas as pd

from agents.technical.moving_average_agent import MovingAverageAgent


class MovingAverageTopNStrategy:
    """Equal-slot portfolio of the strongest moving-average trends."""

    def __init__(
        self,
        data,
        short_window=50,
        long_window=200,
        top_n=10,
        rebalance_frequency="W-FRI",
        price_type="Close",
        transaction_cost=0.0,
        auto_generate=True,
    ):
        self.strategy_name = "MovingAverageTopN"
        self.data = data
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.top_n = int(top_n)
        self.rebalance_frequency = rebalance_frequency
        self.price_type = price_type
        self.transaction_cost = float(transaction_cost)

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

        if auto_generate:
            self.run_all()

    def _validate_inputs(self):
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            raise ValueError("data must be a non-empty pandas DataFrame.")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("data must use a DatetimeIndex.")
        if not self.data.index.is_monotonic_increasing or not self.data.index.is_unique:
            raise ValueError("data index must be sorted and unique.")
        if not isinstance(self.data.columns, pd.MultiIndex) or self.data.columns.nlevels < 2:
            raise ValueError("data must have (ticker, field) MultiIndex columns.")
        if self.short_window < 1 or self.long_window < 1:
            raise ValueError("short_window and long_window must be positive integers.")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window.")
        if self.top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost cannot be negative.")

        fields = self.data.columns.get_level_values(1)
        if self.price_type not in fields:
            raise ValueError(f"Price field {self.price_type!r} was not found in data.")
        try:
            self.data.index.to_period(self.rebalance_frequency)
        except (TypeError, ValueError) as exc:
            raise ValueError("rebalance_frequency must be a valid pandas period frequency.") from exc

    def _rebalance_dates(self):
        periods = self.data.index.to_period(self.rebalance_frequency)
        dates = pd.Series(self.data.index, index=self.data.index).groupby(periods).last()
        return pd.DatetimeIndex(dates.to_numpy())

    def generate_target_weights(self):
        """Run the agent and convert its signals into portfolio target weights."""
        self.agent = MovingAverageAgent(
            self.data,
            short_window=self.short_window,
            long_window=self.long_window,
            price_type=self.price_type,
            auto_generate=True,
        )

        index = self.data.index
        columns = self.stocks_in_data
        self.score_matrix = pd.DataFrame(index=index, columns=columns, dtype=float)
        self.eligibility_matrix = pd.DataFrame(False, index=index, columns=columns)

        for stock in columns:
            signals = self.agent.signal_data[stock]
            self.score_matrix[stock] = signals["SignalStrength"]
            self.eligibility_matrix[stock] = signals["Position"].eq(1)

        weights = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
        rebalance_dates = self._rebalance_dates()
        valid_rebalance_dates = [
            date for date in rebalance_dates if index.get_loc(date) >= self.long_window - 1
        ]

        for date in valid_rebalance_dates:
            weights.loc[date] = 0.0
            scores = self.score_matrix.loc[date].where(self.eligibility_matrix.loc[date]).dropna()
            selected = scores.nlargest(self.top_n).index
            weights.loc[date, selected] = 1.0 / self.top_n

        self.target_weights = weights.ffill().fillna(0.0)
        self.effective_weights = self.target_weights.shift(1).fillna(0.0)

        if valid_rebalance_dates:
            first_position = index.get_loc(valid_rebalance_dates[0]) + 1
            if first_position < len(index):
                self.evaluation_start = index[first_position]
        return self.target_weights

    def calculate_returns(self):
        """Calculate next-bar portfolio returns and an equal-weight buy-and-hold benchmark."""
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
            valid_start_prices = evaluation_close.iloc[0].replace(0.0, np.nan).dropna()
            benchmark_equity = evaluation_close[valid_start_prices.index].div(valid_start_prices).mean(axis=1)
            benchmark_returns.loc[benchmark_equity.index] = benchmark_equity.pct_change().fillna(0.0)
        self.benchmark_returns = benchmark_returns

        strategy_returns = self.portfolio_returns.copy()
        if self.evaluation_start is not None:
            strategy_returns.loc[strategy_returns.index < self.evaluation_start] = 0.0
        else:
            strategy_returns[:] = 0.0
        self.portfolio_returns = strategy_returns
        self.equity_curve = (1.0 + self.portfolio_returns).cumprod()
        self.benchmark_equity_curve = (1.0 + self.benchmark_returns).cumprod()

        rebalance_set = set(self._rebalance_dates())
        self.selection_log = pd.DataFrame(
            {
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
            },
            index=self.data.index,
        )
        return self.portfolio_returns

    @staticmethod
    def _performance_metrics(returns):
        returns = returns.dropna()
        if returns.empty:
            return {
                "Total_Return_%": np.nan,
                "Annualized_Return_%": np.nan,
                "Annualized_Volatility_%": np.nan,
                "Sharpe_Ratio": np.nan,
                "Max_Drawdown_%": np.nan,
            }

        equity = (1.0 + returns).cumprod()
        periods = len(returns)
        total_return = equity.iloc[-1] - 1.0
        annualized_return = equity.iloc[-1] ** (252.0 / periods) - 1.0
        annualized_volatility = returns.std(ddof=1) * np.sqrt(252.0)
        sharpe = (
            returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
            if returns.std(ddof=1) > 0
            else np.nan
        )
        max_drawdown = (equity / equity.cummax() - 1.0).min()
        return {
            "Total_Return_%": total_return * 100.0,
            "Annualized_Return_%": annualized_return * 100.0,
            "Annualized_Volatility_%": annualized_volatility * 100.0,
            "Sharpe_Ratio": sharpe,
            "Max_Drawdown_%": max_drawdown * 100.0,
        }

    def evaluate_performance(self):
        """Return comparable portfolio and benchmark metrics."""
        if self.equity_curve.empty:
            raise RuntimeError("Run the strategy before evaluating performance.")
        if self.evaluation_start is None:
            return pd.DataFrame()

        strategy = self._performance_metrics(self.portfolio_returns.loc[self.evaluation_start:])
        benchmark = self._performance_metrics(self.benchmark_returns.loc[self.evaluation_start:])
        rows = [
            {"Portfolio": self.strategy_name, **strategy},
            {"Portfolio": "EqualWeightBuyHold", **benchmark},
        ]
        result = pd.DataFrame(rows)
        result["Outperformance_%"] = result["Total_Return_%"] - benchmark["Total_Return_%"]
        return result.round(4)

    def run_all(self):
        self.generate_target_weights()
        self.calculate_returns()
        return self
