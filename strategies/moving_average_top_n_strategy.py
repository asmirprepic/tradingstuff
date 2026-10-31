import pandas as pd

from agents.technical.moving_average_agent import MovingAverageAgent
from strategies.base_top_n_strategy import BaseTopNStrategy


class MovingAverageTopNStrategy(BaseTopNStrategy):
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
        evaluation_start=None,
        performance_benchmark=None,
        require_price_above_long=False,
        require_rising_long=False,
        long_slope_window=20,
        downside_shock_threshold=None,
        shock_lookback=5,
    ):
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.require_price_above_long = bool(require_price_above_long)
        self.require_rising_long = bool(require_rising_long)
        self.long_slope_window = int(long_slope_window)
        self.downside_shock_threshold = (
            None
            if downside_shock_threshold is None
            else float(downside_shock_threshold)
        )
        self.shock_lookback = int(shock_lookback)
        if self.short_window < 1 or self.long_window < 1:
            raise ValueError("short_window and long_window must be positive integers.")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window.")
        if self.long_slope_window < 1:
            raise ValueError("long_slope_window must be a positive integer.")
        if self.shock_lookback < 1:
            raise ValueError("shock_lookback must be a positive integer.")
        if (
            self.downside_shock_threshold is not None
            and not 0.0 < self.downside_shock_threshold < 1.0
        ):
            raise ValueError("downside_shock_threshold must be between 0 and 1.")
        super().__init__(
            data,
            "MovingAverageTopN",
            top_n,
            rebalance_frequency,
            price_type,
            transaction_cost,
            evaluation_start,
            performance_benchmark,
        )
        if auto_generate:
            self.run_all()

    def generate_target_weights(self):
        self.agent = MovingAverageAgent(
            self.data,
            short_window=self.short_window,
            long_window=self.long_window,
            price_type=self.price_type,
            auto_generate=True,
        )
        self.score_matrix = pd.DataFrame(index=self.data.index, columns=self.stocks_in_data, dtype=float)
        self.eligibility_matrix = pd.DataFrame(
            False, index=self.data.index, columns=self.stocks_in_data
        )
        for stock in self.stocks_in_data:
            signals = self.agent.signal_data[stock]
            signals["PriceAboveLong"] = signals["price"].gt(signals["SMA_long"])
            signals["LongSMARising"] = signals["SMA_long"].gt(
                signals["SMA_long"].shift(self.long_slope_window)
            )
            daily_return = signals["price"].pct_change(fill_method=None)
            signals["MaximumDailyLoss"] = (-daily_return).clip(lower=0.0).rolling(
                self.shock_lookback,
                min_periods=self.shock_lookback,
            ).max()
            signals["DownsideShockExcluded"] = False
            if self.downside_shock_threshold is not None:
                signals["DownsideShockExcluded"] = signals["MaximumDailyLoss"].ge(
                    self.downside_shock_threshold
                )
            eligible = signals["Position"].eq(1)
            if self.require_price_above_long:
                eligible &= signals["PriceAboveLong"]
            if self.require_rising_long:
                eligible &= signals["LongSMARising"]
            if self.downside_shock_threshold is not None:
                eligible &= ~signals["DownsideShockExcluded"]
            signals["StrategyEligible"] = eligible
            self.score_matrix[stock] = signals["SignalStrength"]
            self.eligibility_matrix[stock] = eligible

        minimum_history = self.long_window - 1
        if self.require_rising_long:
            minimum_history += self.long_slope_window
        if self.downside_shock_threshold is not None:
            minimum_history = max(minimum_history, self.shock_lookback)
        return self._build_target_weights(minimum_history=minimum_history)
