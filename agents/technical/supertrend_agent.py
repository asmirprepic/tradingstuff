from agents.base_agents.trading_agent import TradingAgent
import numpy as np
import pandas as pd


class SupertrendAgent(TradingAgent):
    """
    Supertrend-based trend-following agent.

    Uses ATR bands derived from High/Low/Close and flips long/short when price
    crosses the active supertrend line.
    """

    def __init__(
        self,
        data,
        period=10,
        multiplier=3.0,
        price_type="Close",
        auto_generate=True,
    ):
        super().__init__(data)
        self.algorithm_name = "Supertrend"
        self.score_column = "SignalStrength"

        self.period = int(period)
        self.multiplier = float(multiplier)
        if self.period < 1:
            raise ValueError("period must be a positive integer.")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive.")

        self.price_type = price_type
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        if auto_generate:
            self.run_all()

    def generate_signal_strategy(self, stock, mode="backtest"):
        high = self.data[(stock, "High")]
        low = self.data[(stock, "Low")]
        close = self.data[(stock, self.price_type)]

        signals = pd.DataFrame(index=close.index)
        signals["price"] = close
        signals["return"] = np.log(close / close.shift(1))

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(window=self.period).mean()
        hl2 = (high + low) / 2.0
        basic_upper = hl2 + self.multiplier * atr
        basic_lower = hl2 - self.multiplier * atr

        final_upper = pd.Series(np.nan, index=close.index, dtype=float)
        final_lower = pd.Series(np.nan, index=close.index, dtype=float)
        supertrend = pd.Series(np.nan, index=close.index, dtype=float)
        direction = pd.Series(0, index=close.index, dtype=int)

        valid_idx = np.flatnonzero(atr.notna().to_numpy())
        if len(valid_idx) > 0:
            first_valid = int(valid_idx[0])
            final_upper.iloc[first_valid] = basic_upper.iloc[first_valid]
            final_lower.iloc[first_valid] = basic_lower.iloc[first_valid]
            if close.iloc[first_valid] >= hl2.iloc[first_valid]:
                supertrend.iloc[first_valid] = final_lower.iloc[first_valid]
                direction.iloc[first_valid] = 1
            else:
                supertrend.iloc[first_valid] = final_upper.iloc[first_valid]
                direction.iloc[first_valid] = -1

            for i in range(first_valid + 1, len(close)):
                prev_final_upper = final_upper.iloc[i - 1]
                prev_final_lower = final_lower.iloc[i - 1]
                prev_supertrend = supertrend.iloc[i - 1]
                prev_close_i = close.iloc[i - 1]

                if basic_upper.iloc[i] < prev_final_upper or prev_close_i > prev_final_upper:
                    final_upper.iloc[i] = basic_upper.iloc[i]
                else:
                    final_upper.iloc[i] = prev_final_upper

                if basic_lower.iloc[i] > prev_final_lower or prev_close_i < prev_final_lower:
                    final_lower.iloc[i] = basic_lower.iloc[i]
                else:
                    final_lower.iloc[i] = prev_final_lower

                if prev_supertrend == prev_final_upper:
                    if close.iloc[i] <= final_upper.iloc[i]:
                        supertrend.iloc[i] = final_upper.iloc[i]
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = final_lower.iloc[i]
                        direction.iloc[i] = 1
                else:
                    if close.iloc[i] >= final_lower.iloc[i]:
                        supertrend.iloc[i] = final_lower.iloc[i]
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = final_upper.iloc[i]
                        direction.iloc[i] = -1

        valid = atr.notna() & supertrend.notna()

        signals["ATR"] = atr
        signals["UpperBand"] = final_upper
        signals["LowerBand"] = final_lower
        signals["Supertrend"] = supertrend
        signals["Valid"] = valid
        signals["Position"] = np.where(valid, direction, 0).astype(int)
        sig = signals["Position"].diff().fillna(0).astype(int)
        signals["Signal"] = sig.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        signals["SignalStrength"] = ((close - supertrend) / close).where(valid)

        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest"):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock, mode=mode)
        self.calculate_returns()
