import numpy as np
import pandas as pd

from agents.technical.stock_regime_agent import StockRegimeAgent
from strategies.base_top_n_strategy import BaseTopNStrategy


class StockRegimeTopNStrategy(BaseTopNStrategy):
    """Long-only equal-slot portfolio ranked by switching-AR forecasts.

    Forecasts are horizon-cumulative log returns conditional on zero market returns,
    not forecasts of executable next-open P&L. State uncertainty is integrated
    into the forecast; no hard continuation/reversal label is required.
    """

    def __init__(self, data, estimation_window=252, refit_interval=20,
                 market_returns=None, ar_threshold=.05, maxiter=200,
                 top_n=5, rebalance_frequency="B", minimum_forecast=0.,
                 price_type="Close", transaction_cost=0., auto_generate=True,
                 evaluation_start=None, performance_benchmark=None,
                 execution_mode="next_open", verbose=False, prediction_horizon=1):
        if not np.isfinite(minimum_forecast) or minimum_forecast < 0:
            raise ValueError("minimum_forecast must be finite and nonnegative")
        self.minimum_forecast = float(minimum_forecast)
        if (isinstance(prediction_horizon, bool) or not np.isfinite(prediction_horizon)
                or prediction_horizon < 1 or int(prediction_horizon) != prediction_horizon):
            raise ValueError("prediction_horizon must be a positive integer")
        self.prediction_horizon = int(prediction_horizon)
        self.agent_parameters = dict(
            estimation_window=estimation_window, refit_interval=refit_interval,
            market_returns=market_returns, ar_threshold=ar_threshold,
            maxiter=maxiter, price_type=price_type, verbose=verbose,
            prediction_horizon=self.prediction_horizon,
        )
        super().__init__(data, "StockRegimeTopN", top_n, rebalance_frequency,
                         price_type, transaction_cost, evaluation_start,
                         performance_benchmark, execution_mode)
        if auto_generate:
            self.run_all()

    def generate_target_weights(self):
        self.agent = StockRegimeAgent(self.data, **self.agent_parameters)
        self.score_matrix = pd.DataFrame(index=self.data.index,
                                         columns=self.stocks_in_data, dtype=float)
        self.eligibility_matrix = pd.DataFrame(False, index=self.data.index,
                                               columns=self.stocks_in_data)
        for stock in self.stocks_in_data:
            signals = self.agent.signal_data[stock]
            scores = signals["ExpectedReturn"].where(
                np.isfinite(signals["ExpectedReturn"]))
            self.score_matrix[stock] = scores
            self.eligibility_matrix[stock] = (
                signals["Valid"] & scores.gt(self.minimum_forecast))
        # Start after model warm-up, even if fits fail: unavailable slots stay
        # in cash rather than silently starting evaluation at the first winner.
        return self._build_target_weights(
            minimum_history=int(self.agent_parameters["estimation_window"]) + 2)
