import pandas as pd

from agents.technical.moving_average_agent import MovingAverageAgent
from strategies.base_top_n_strategy import BaseTopNStrategy


class MovingAverageTopNStrategy(BaseTopNStrategy):
    """Equal-slot portfolio of the strongest moving-average trends."""

    def __init__(self, data, short_window=50, long_window=200, top_n=10,
                 rebalance_frequency="W-FRI", price_type="Close",
                 transaction_cost=0.0, auto_generate=True, evaluation_start=None,
                 performance_benchmark=None):
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        if self.short_window < 1 or self.long_window < 1:
            raise ValueError("short_window and long_window must be positive integers.")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window.")
        super().__init__(data, "MovingAverageTopN", top_n, rebalance_frequency,
                         price_type, transaction_cost, evaluation_start,
                         performance_benchmark)
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
            self.score_matrix[stock] = signals["SignalStrength"]
            self.eligibility_matrix[stock] = signals["Position"].eq(1)
        return self._build_target_weights(minimum_history=self.long_window - 1)
