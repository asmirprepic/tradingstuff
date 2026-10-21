import pandas as pd

from agents.technical.residual_momentum_agent import ResidualMomentumAgent
from strategies.base_top_n_strategy import BaseTopNStrategy


class ResidualMomentumTopNStrategy(BaseTopNStrategy):
    """Equal-slot portfolio of stocks with the strongest residual momentum."""

    def __init__(self, data, regression_window=126, momentum_window=20,
                 benchmark_returns=None, top_n=10, rebalance_frequency="W-FRI",
                 price_type="Close", transaction_cost=0.0, auto_generate=True,
                 evaluation_start=None):
        self.regression_window = int(regression_window)
        self.momentum_window = int(momentum_window)
        self.factor_benchmark_returns = benchmark_returns
        if self.regression_window < 2:
            raise ValueError("regression_window must be at least 2.")
        if self.momentum_window < 1:
            raise ValueError("momentum_window must be a positive integer.")
        super().__init__(data, "ResidualMomentumTopN", top_n, rebalance_frequency,
                         price_type, transaction_cost, evaluation_start)
        if auto_generate:
            self.run_all()

    def generate_target_weights(self):
        self.agent = ResidualMomentumAgent(
            self.data,
            regression_window=self.regression_window,
            momentum_window=self.momentum_window,
            benchmark_returns=self.factor_benchmark_returns,
            price_type=self.price_type,
            auto_generate=True,
        )
        self.score_matrix = pd.DataFrame(index=self.data.index, columns=self.stocks_in_data, dtype=float)
        self.eligibility_matrix = pd.DataFrame(
            False, index=self.data.index, columns=self.stocks_in_data
        )
        for stock in self.stocks_in_data:
            signals = self.agent.signal_data[stock]
            self.score_matrix[stock] = signals["ResidualMomentum"]
            self.eligibility_matrix[stock] = signals["Position"].eq(1)
        return self._build_target_weights(
            minimum_history=self.regression_window + self.momentum_window
        )
