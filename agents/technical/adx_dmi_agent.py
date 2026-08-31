from agents.base_agents.trading_agent import TradingAgent
import numpy as np
import pandas as pd


class ADXDMIAgent(TradingAgent):
    """
    Trend-following agent using Directional Movement Index and ADX.

    Takes long exposure when +DI dominates -DI and ADX confirms trend strength,
    short exposure on the inverse, and stays flat when the trend is weak.
    """

    def __init__(
        self,
        data,
        period=14,
        adx_threshold=20.0,
        auto_generate=True,
    ):
        super().__init__(data)
        self.algorithm_name = "ADX_DMI"
        self.score_column = "SignalStrength"

        self.period = int(period)
        self.adx_threshold = float(adx_threshold)
        if self.period < 1:
            raise ValueError("period must be a positive integer.")
        if self.adx_threshold < 0:
            raise ValueError("adx_threshold must be non-negative.")

        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        if auto_generate:
            self.run_all()

    def generate_signal_strategy(self, stock, mode="backtest"):
        high = self.data[(stock, "High")]
        low = self.data[(stock, "Low")]
        close = self.data[(stock, "Close")]

        signals = pd.DataFrame(index=close.index)
        signals["return"] = np.log(close / close.shift(1))

        up_move = high.diff()
        down_move = low.shift(1) - low

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=close.index,
            dtype=float,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=close.index,
            dtype=float,
        )

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.rolling(window=self.period).sum()
        plus_di = 100.0 * plus_dm.rolling(window=self.period).sum().div(atr.replace(0, np.nan))
        minus_di = 100.0 * minus_dm.rolling(window=self.period).sum().div(atr.replace(0, np.nan))
        di_sum = plus_di + minus_di
        dx = 100.0 * (plus_di - minus_di).abs().div(di_sum.replace(0, np.nan))
        adx = dx.rolling(window=self.period).mean()

        valid = plus_di.notna() & minus_di.notna() & adx.notna()
        trend_up = valid & (adx >= self.adx_threshold) & (plus_di > minus_di)
        trend_down = valid & (adx >= self.adx_threshold) & (minus_di > plus_di)

        position = np.where(trend_up, 1, np.where(trend_down, -1, 0))
        signed_strength = ((plus_di - minus_di) / di_sum.replace(0, np.nan)) * (adx / 100.0)

        signals["PlusDI"] = plus_di
        signals["MinusDI"] = minus_di
        signals["DX"] = dx
        signals["ADX"] = adx
        signals["Valid"] = valid
        signals["Position"] = np.where(valid, position, 0).astype(int)
        signals["Signal"] = 0
        signals.loc[signals["Position"] > signals["Position"].shift(1), "Signal"] = 1
        signals.loc[signals["Position"] < signals["Position"].shift(1), "Signal"] = -1
        signals["SignalStrength"] = signed_strength.where(valid)

        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest"):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock, mode=mode)
        self.calculate_returns()
