from agents.base_agents.trading_agent import TradingAgent
import numpy as np
import pandas as pd


class MomentumAgent(TradingAgent):
    """
    Momentum agent with optional multi-lookback ensemble scoring.
    """

    def __init__(
        self,
        data,
        back_length=1,
        lookbacks=None,
        long_only: bool = False,
        score_mode: str = "z",
        price_type: str = "Close",
        auto_generate: bool = True,
    ):
        super().__init__(data)
        self.algorithm_name = "Momentum"
        self.score_column = "SignalStrength"

        if lookbacks is None:
            if isinstance(back_length, (list, tuple, set, np.ndarray)):
                lookbacks = list(back_length)
            else:
                lookbacks = [back_length]

        lookbacks = [int(x) for x in lookbacks if int(x) > 0]
        if not lookbacks:
            lookbacks = [1]

        self.lookbacks = sorted(set(lookbacks))
        self.back_length = int(self.lookbacks[0])
        self.long_only = bool(long_only)

        score_mode = str(score_mode).lower().strip()
        if score_mode not in {"z", "raw"}:
            raise ValueError("score_mode must be 'z' or 'raw'")
        self.score_mode = score_mode

        self.price_type = price_type
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        if auto_generate:
            self.run_all()

    def _momentum_components(self, price, returns, lookback):
        total_log_return = np.log(price / price.shift(lookback))
        dailyized_momentum = total_log_return / float(lookback)
        horizon_vol = returns.rolling(lookback).std(ddof=0) * np.sqrt(lookback)
        horizon_vol = horizon_vol.replace(0, np.nan)
        z_score = total_log_return / horizon_vol
        return total_log_return, dailyized_momentum, z_score

    def generate_signal_strategy(self, stock, mode="backtest"):
        price = self.data[(stock, self.price_type)]
        signals = pd.DataFrame(index=price.index)
        signals["return"] = np.log(price / price.shift(1))

        momentum_cols = []
        raw_score_cols = []
        z_score_cols = []

        for lookback in self.lookbacks:
            momentum_col = f"Momentum_{lookback}"
            raw_score_col = f"MomentumDaily_{lookback}"
            z_score_col = f"MomentumZ_{lookback}"

            momentum, dailyized_momentum, z_score = self._momentum_components(
                price,
                signals["return"],
                lookback,
            )

            signals[momentum_col] = momentum
            signals[raw_score_col] = dailyized_momentum
            signals[z_score_col] = z_score

            momentum_cols.append(momentum_col)
            raw_score_cols.append(raw_score_col)
            z_score_cols.append(z_score_col)

        ensemble_valid = signals[momentum_cols + z_score_cols].notna().all(axis=1)

        if len(momentum_cols) == 1:
            momentum_ensemble = signals[momentum_cols[0]]
        else:
            momentum_ensemble = signals[momentum_cols].mean(axis=1)
        signals["Momentum"] = momentum_ensemble.where(ensemble_valid)

        if self.score_mode == "raw":
            if len(raw_score_cols) == 1:
                strength = signals[raw_score_cols[0]]
            else:
                strength = signals[raw_score_cols].mean(axis=1)
        else:
            if len(z_score_cols) == 1:
                strength = signals[z_score_cols[0]]
            else:
                strength = signals[z_score_cols].mean(axis=1)
        signals["SignalStrength"] = strength.where(ensemble_valid)

        if self.long_only:
            position = np.where(signals["SignalStrength"] > 0, 1, 0)
        else:
            position = np.where(
                signals["SignalStrength"] > 0,
                1,
                np.where(signals["SignalStrength"] < 0, -1, 0),
            )

        signals["Position"] = np.where(ensemble_valid, position, 0).astype(int)
        sig = signals["Position"].diff().fillna(0).astype(int)
        signals["Signal"] = sig.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest"):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock, mode=mode)
        self.calculate_returns()
