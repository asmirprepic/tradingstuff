import numpy as np
import pandas as pd

from agents.base_agents.trading_agent import TradingAgent


class ChangePointAgent(TradingAgent):
    """Detect persistent return shifts with a causal two-sided CUSUM test."""

    def __init__(
        self,
        data,
        lookback=60,
        threshold=5.0,
        drift=0.25,
        score_decay=20.0,
        auto_generate=True,
    ):
        super().__init__(data)
        self.algorithm_name = "ChangePoint"
        self.score_column = "SignalStrength"

        self.lookback = int(lookback)
        self.threshold = float(threshold)
        self.drift = float(drift)
        self.score_decay = float(score_decay)
        if self.lookback < 2:
            raise ValueError("lookback must be an integer of at least 2.")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive.")
        if self.drift < 0:
            raise ValueError("drift must be non-negative.")
        if self.score_decay <= 0:
            raise ValueError("score_decay must be positive.")

        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        if auto_generate:
            self.run_all()

    def generate_signal_strategy(self, stock, mode="backtest"):
        close = self.data[(stock, "Close")].astype(float)
        returns = np.log(close / close.shift(1))

        # Shift the baseline so today's observation cannot influence its own test.
        baseline_mean = returns.rolling(self.lookback).mean().shift(1)
        baseline_std = returns.rolling(self.lookback).std(ddof=0).shift(1).replace(0, np.nan)
        z_score = (returns - baseline_mean) / baseline_std
        valid = z_score.notna() & np.isfinite(z_score)

        positive = np.zeros(len(close), dtype=float)
        negative = np.zeros(len(close), dtype=float)
        change_point = np.zeros(len(close), dtype=int)
        position = np.zeros(len(close), dtype=int)
        bars_since_change = np.full(len(close), np.nan)

        regime = 0
        age = None
        for i, value in enumerate(z_score.to_numpy()):
            if not valid.iloc[i]:
                continue

            previous_positive = positive[i - 1] if i else 0.0
            previous_negative = negative[i - 1] if i else 0.0
            positive[i] = max(0.0, previous_positive + value - self.drift)
            negative[i] = min(0.0, previous_negative + value + self.drift)

            if positive[i] >= self.threshold:
                change_point[i] = 1
                regime = 1
                age = 0
                positive[i] = 0.0
                negative[i] = 0.0
            elif negative[i] <= -self.threshold:
                change_point[i] = -1
                regime = -1
                age = 0
                positive[i] = 0.0
                negative[i] = 0.0
            elif age is not None:
                age += 1

            position[i] = regime
            if age is not None:
                bars_since_change[i] = age

        confidence = np.exp(-bars_since_change / self.score_decay)
        signal_strength = position * confidence

        signals = pd.DataFrame(index=close.index)
        signals["price"] = close
        signals["return"] = returns
        signals["BaselineMean"] = baseline_mean
        signals["BaselineVolatility"] = baseline_std
        signals["ZScore"] = z_score
        signals["PositiveCUSUM"] = positive
        signals["NegativeCUSUM"] = negative
        signals["ChangePoint"] = change_point
        signals["BarsSinceChange"] = bars_since_change
        signals["Valid"] = valid
        signals["Position"] = position
        signals["Signal"] = change_point
        signals["SignalStrength"] = signal_strength

        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest"):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock, mode=mode)
        self.calculate_returns()
