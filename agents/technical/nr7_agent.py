import numpy as np
import pandas as pd
from agents.base_agents.trading_agent import TradingAgent

class NR7BreakoutAgent(TradingAgent):
    """
    Simple NR7 breakout:
      - Identify day t-1 with narrowest range over last 7 days (NR7).
      - On day t:
          Close[t] > High[t-1] -> long
          Close[t] < Low[t-1]  -> short (toggleable)
      - Hold for `hold_days` bars, then exit.
      - Next-bar execution: StrategyRet = Position.shift(1) * AssetRet.
    """
    def __init__(self, data, hold_days=5, take_shorts=False):
        super().__init__(data)
        self.algorithm_name = "NR7Breakout"
        self.hold_days = int(hold_days)
        self.take_shorts = bool(take_shorts)
        self.signal_data = {}

    def _build_signals_for_stock(self, stock: str) -> pd.DataFrame:
        df = self.data[stock].copy()
        if not {'Open','High','Low','Close'}.issubset(df.columns):
            raise ValueError(f"{stock}: missing OHLC columns")

        hi, lo, cl = df['High'], df['Low'], df['Close']
        rng = (hi - lo)
        # NR7 at t-1: range is the smallest of the last 7 days ending at t-1
        nr7 = rng.shift(1) == rng.shift(1).rolling(7).min()

        # Breakout signals at t, based on NR7 at t-1
        long_entry  = nr7 & (cl > hi.shift(1))
        short_entry = nr7 & (cl < lo.shift(1)) if self.take_shorts else pd.Series(False, index=df.index)

        pos = pd.Series(0, index=df.index, dtype=int)
        days_left = 0

        for t in range(1, len(df)):
            i = df.index[t]
            if pos.iloc[t-1] == 0 and days_left == 0:
                if long_entry.iloc[t]:
                    pos.iloc[t] = 1
                    days_left = self.hold_days
                elif short_entry.iloc[t]:
                    pos.iloc[t] = -1
                    days_left = self.hold_days
                else:
                    pos.iloc[t] = 0
            else:
                # continue holding
                pos.iloc[t] = pos.iloc[t-1]
                days_left = max(0, days_left - 1)
                if days_left == 0:
                    pos.iloc[t] = 0

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[pos.gt(pos.shift(1))] = 1
        signal[pos.lt(pos.shift(1))] = -1

        asset_ret = np.log(cl / cl.shift(1)).fillna(0.0)
        strat_ret = pos.shift(1).fillna(0) * asset_ret

        return pd.DataFrame({
            'Position': pos,
            'Signal': signal,
            'AssetRet': asset_ret,
            'StrategyRet': strat_ret,
            'NR7_prev': nr7.astype(int)
        }, index=df.index)

    def generate_signal_strategy(self, stock, mode='backtest'):
        self.signal_data[stock] = self._build_signals_for_stock(stock)
        return self.signal_data[stock]

    def run_all(self, mode='backtest'):
        self.signal_data = {}
        for stock in self.data.columns.get_level_values(0).unique():
            try:
                self.generate_signal_strategy(stock, mode=mode)
            except Exception as e:
                print(f"[WARN] {stock}: {e}")
        self.calculate_returns()
