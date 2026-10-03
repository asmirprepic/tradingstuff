import numpy as np
import pandas as pd

from agents.base_agents.trading_agent import TradingAgent


class RelativeStrengthAgent(TradingAgent):
    """Cross-sectional relative momentum portfolio with trend and risk filters."""

    def __init__(self, data, lookback_period=126, trend_period=200, volatility_period=20,
                 rebalance_period=20, top_n=20, price_type="Close", auto_generate=True):
        super().__init__(data)
        parameters = (lookback_period, trend_period, volatility_period, rebalance_period, top_n)
        if any(int(value) < 1 for value in parameters):
            raise ValueError("All period parameters and top_n must be positive integers.")
        self.algorithm_name = "RelativeStrength"
        self.score_column = "RelativeStrength"
        self.lookback_period = int(lookback_period)
        self.trend_period = int(trend_period)
        self.volatility_period = int(volatility_period)
        self.rebalance_period = int(rebalance_period)
        self.top_n = int(top_n)
        self.price_type = price_type
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        if self.top_n > len(self.stocks_in_data):
            raise ValueError("top_n cannot exceed the number of stocks in the dataset.")
        self.holdings_matrix = pd.DataFrame()
        self.weights_matrix = pd.DataFrame()
        self.selection_matrix = pd.DataFrame()
        self.selection_log = pd.DataFrame()
        self.portfolio_log_returns = pd.Series(dtype=float)
        self.cumulative_returns = pd.Series(dtype=float)
        if auto_generate:
            self.run_all()

    def _build_portfolio(self):
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        daily_returns = close.pct_change(fill_method=None)
        lookback_returns = close.div(close.shift(self.lookback_period)).sub(1.0)
        relative_strength = lookback_returns.sub(lookback_returns.mean(axis=1), axis=0)
        trend_average = close.rolling(self.trend_period, min_periods=self.trend_period).mean()
        volatility = daily_returns.rolling(
            self.volatility_period, min_periods=self.volatility_period
        ).std()
        weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        selections = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=np.int8)
        first_position = max(self.lookback_period, self.trend_period - 1, self.volatility_period)

        for position in range(first_position, len(close), self.rebalance_period):
            date = close.index[position]
            valid = (relative_strength.loc[date].gt(0)
                     & close.loc[date].gt(trend_average.loc[date])
                     & volatility.loc[date].gt(0))
            candidates = relative_strength.loc[date, valid].dropna().nlargest(self.top_n)
            end = min(position + self.rebalance_period, len(close))
            if candidates.empty:
                continue
            inverse_volatility = volatility.loc[date, candidates.index].pow(-1).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if inverse_volatility.empty or inverse_volatility.sum() <= 0:
                continue
            target_weights = inverse_volatility / inverse_volatility.sum()
            selections.loc[date, target_weights.index] = 1
            weights.loc[close.index[position:end], target_weights.index] = target_weights.to_numpy()

        holdings = weights.gt(0).astype(np.int8)
        ranks = relative_strength.rank(axis=1, method="first", ascending=False)
        return close, daily_returns, relative_strength, ranks, selections, holdings, weights

    def generate_signal_strategy(self, stock=None, mode="backtest"):
        (close, daily_returns, relative_strength, ranks, self.selection_matrix,
         self.holdings_matrix, self.weights_matrix) = self._build_portfolio()
        self.signal_data = {}
        for ticker in self.stocks_in_data:
            frame = pd.DataFrame(index=close.index)
            frame["return"] = np.log1p(daily_returns[ticker])
            frame["RelativeStrength"] = relative_strength[ticker]
            frame["Rank"] = ranks[ticker]
            frame["Weight"] = self.weights_matrix[ticker]
            frame["Position"] = self.holdings_matrix[ticker]
            frame["Signal"] = np.sign(frame["Position"].diff().fillna(0)).astype(int)
            self.signal_data[ticker] = frame
        if stock is None:
            return self.signal_data
        if stock not in self.signal_data:
            raise KeyError(f"Stock {stock} not found in signal data.")
        return self.signal_data[stock]

    def calculate_returns(self):
        super().calculate_returns()
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        daily_returns = close.pct_change(fill_method=None)
        effective_weights = self.weights_matrix.reindex(close.index).shift(1).fillna(0.0)
        portfolio_returns = (daily_returns * effective_weights).sum(axis=1, min_count=1)
        portfolio_returns = portfolio_returns.fillna(0.0).clip(lower=-0.999999)
        self.portfolio_log_returns = np.log1p(portfolio_returns)
        self.cumulative_returns = self.portfolio_log_returns.cumsum()
        self.selection_log = pd.DataFrame({
            "Selected_Stocks": self.holdings_matrix.apply(
                lambda row: list(row.index[row.eq(1)]), axis=1
            ),
            "InvestedWeight": self.weights_matrix.sum(axis=1),
            "Log_Return": self.portfolio_log_returns,
            "Cum_Log_Return": self.cumulative_returns,
            "N_Held": self.holdings_matrix.sum(axis=1).astype(int),
        }, index=close.index)

    def run_all(self, mode="backtest"):
        self.generate_signal_strategy(mode=mode)
        self.calculate_returns()
