import numpy as np
import pandas as pd

from agents.base_agents.trading_agent import TradingAgent


class VolatilityAdjustedBreakoutAgent(TradingAgent):
    """Long-only breakout signal normalized by Average True Range."""

    ATR_EPSILON = 1e-12

    def __init__(self, data, breakout_window=60, atr_window=20,
                 price_type="Close", auto_generate=True):
        super().__init__(data)
        self.algorithm_name = "VolatilityAdjustedBreakout"
        self.score_column = "BreakoutScore"
        self.breakout_window = int(breakout_window)
        self.atr_window = int(atr_window)
        self.price_type = price_type
        if self.breakout_window < 1:
            raise ValueError("breakout_window must be a positive integer.")
        if self.atr_window < 1:
            raise ValueError("atr_window must be a positive integer.")
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        if auto_generate:
            self.run_all()

    def generate_signal_strategy(self, stock, mode="backtest"):
        if mode not in {"backtest", "live"}:
            raise ValueError("mode must be 'backtest' or 'live'.")
        required = {(stock, "High"), (stock, "Low"), (stock, self.price_type)}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"{stock}: missing required columns {sorted(missing)}")

        high = self.data[(stock, "High")].astype(float)
        low = self.data[(stock, "Low")].astype(float)
        price = self.data[(stock, self.price_type)].astype(float)
        previous_close = price.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(self.atr_window, min_periods=self.atr_window).mean()

        # shift(1) ensures today's price is not included in its own breakout threshold.
        prior_high = price.shift(1).rolling(
            self.breakout_window,
            min_periods=self.breakout_window,
        ).max()
        valid = prior_high.notna() & atr.gt(self.ATR_EPSILON)
        breakout_score = ((price - prior_high) / atr).where(valid)

        signals = pd.DataFrame(index=self.data.index)
        signals["price"] = price
        signals["return"] = np.log(price / price.shift(1))
        signals["TrueRange"] = true_range
        signals["ATR"] = atr
        signals["PriorHigh"] = prior_high
        signals["BreakoutScore"] = breakout_score
        signals["SignalStrength"] = breakout_score
        signals["Valid"] = valid
        signals["Breakout"] = valid & breakout_score.gt(0.0)
        signals["Position"] = np.where(valid & breakout_score.gt(0.0), 1, 0).astype(int)
        signals["Signal"] = np.sign(signals["Position"].diff().fillna(0)).astype(int)
        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest"):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock, mode=mode)
        self.calculate_returns()
