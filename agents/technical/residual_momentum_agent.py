import numpy as np
import pandas as pd

from agents.base_agents.trading_agent import TradingAgent


class ResidualMomentumAgent(TradingAgent):
    """Momentum signal based on returns unexplained by an estimated market factor."""

    VARIANCE_EPSILON = 1e-12
    MOMENTUM_EPSILON = 1e-12

    def __init__(self, data, regression_window=126, momentum_window=20,
                 benchmark_returns=None, price_type="Close", auto_generate=True):
        super().__init__(data)
        self.algorithm_name = "ResidualMomentum"
        self.score_column = "ResidualMomentum"
        self.regression_window = int(regression_window)
        self.momentum_window = int(momentum_window)
        self.price_type = price_type
        if self.regression_window < 2:
            raise ValueError("regression_window must be at least 2.")
        if self.momentum_window < 1:
            raise ValueError("momentum_window must be a positive integer.")

        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        stock_returns = np.log(close / close.shift(1))
        self.benchmark_returns = self._resolve_benchmark_returns(benchmark_returns, stock_returns)
        if auto_generate:
            self.run_all()

    def _resolve_benchmark_returns(self, benchmark_returns, stock_returns):
        if benchmark_returns is None:
            return stock_returns.mean(axis=1, skipna=True).rename("BenchmarkReturn")
        if not isinstance(benchmark_returns, pd.Series):
            raise ValueError("benchmark_returns must be a pandas Series of log returns.")
        if not benchmark_returns.index.is_unique:
            raise ValueError("benchmark_returns index must be unique.")
        return pd.to_numeric(benchmark_returns, errors="coerce").reindex(
            self.data.index
        ).rename("BenchmarkReturn")

    def generate_signal_strategy(self, stock, mode="backtest"):
        if mode not in {"backtest", "live"}:
            raise ValueError("mode must be 'backtest' or 'live'.")
        price = self.data[(stock, self.price_type)].astype(float)
        stock_returns = np.log(price / price.shift(1))
        benchmark = self.benchmark_returns

        # The coefficients used at t are estimated only with observations through t-1.
        historical_stock = stock_returns.shift(1)
        historical_benchmark = benchmark.shift(1)
        stock_mean = historical_stock.rolling(self.regression_window).mean()
        benchmark_mean = historical_benchmark.rolling(self.regression_window).mean()
        covariance = historical_stock.rolling(self.regression_window).cov(historical_benchmark)
        variance = historical_benchmark.rolling(self.regression_window).var()
        beta = covariance.div(variance.where(variance.abs() > self.VARIANCE_EPSILON))
        alpha = stock_mean - beta * benchmark_mean
        residual_return = stock_returns - (alpha + beta * benchmark)
        residual_momentum = residual_return.rolling(self.momentum_window).sum()

        signals = pd.DataFrame(index=self.data.index)
        signals["price"] = price
        signals["return"] = stock_returns
        signals["BenchmarkReturn"] = benchmark
        signals["Alpha"] = alpha
        signals["Beta"] = beta
        signals["ResidualReturn"] = residual_return
        signals["ResidualMomentum"] = residual_momentum
        signals["SignalStrength"] = residual_momentum
        signals["Valid"] = residual_momentum.notna()
        signals["Position"] = np.where(
            signals["Valid"]
            & signals["ResidualMomentum"].gt(self.MOMENTUM_EPSILON),
            1,
            0,
        ).astype(int)
        signals["Signal"] = np.sign(signals["Position"].diff().fillna(0)).astype(int)
        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest"):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock, mode=mode)
        self.calculate_returns()
