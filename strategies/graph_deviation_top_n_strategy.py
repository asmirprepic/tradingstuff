import pandas as pd

from agents.technical.graph_relative_return_agent import GraphRelativeReturnAgent
from strategies.base_top_n_strategy import BaseTopNStrategy


class GraphDeviationTopNStrategy(BaseTopNStrategy):
    """Weekly top-N portfolio from dynamically oriented graph deviations."""

    def __init__(
        self,
        data,
        graph_window=126,
        top_k=8,
        minimum_correlation=0.25,
        recent_return_window=5,
        prediction_horizon=5,
        calibration_weeks=52,
        minimum_calibration_weeks=12,
        direction_shrinkage=0.10,
        shock_threshold=0.08,
        top_n=5,
        rebalance_frequency="W-FRI",
        remove_market_mode=True,
        price_type="Close",
        transaction_cost=0.0,
        auto_generate=True,
        evaluation_start=None,
        performance_benchmark=None,
        execution_mode="next_open",
    ):
        self.agent_parameters = {
            "graph_window": graph_window,
            "top_k": top_k,
            "minimum_correlation": minimum_correlation,
            "recent_return_window": recent_return_window,
            "prediction_horizon": prediction_horizon,
            "calibration_weeks": calibration_weeks,
            "minimum_calibration_weeks": minimum_calibration_weeks,
            "direction_shrinkage": direction_shrinkage,
            "shock_threshold": shock_threshold,
            "decision_frequency": rebalance_frequency,
            "remove_market_mode": remove_market_mode,
            "price_type": price_type,
        }
        super().__init__(
            data,
            "GraphDeviationTopN",
            top_n,
            rebalance_frequency,
            price_type,
            transaction_cost,
            evaluation_start,
            performance_benchmark,
            execution_mode,
        )
        if auto_generate:
            self.run_all()

    def generate_target_weights(self):
        self.agent = GraphRelativeReturnAgent(
            self.data,
            auto_generate=True,
            **self.agent_parameters,
        )
        self.score_matrix = pd.DataFrame(
            index=self.data.index, columns=self.stocks_in_data, dtype=float
        )
        self.eligibility_matrix = pd.DataFrame(
            False, index=self.data.index, columns=self.stocks_in_data
        )
        for stock in self.stocks_in_data:
            signals = self.agent.signal_data[stock]
            self.score_matrix[stock] = signals["GraphScore"]
            self.eligibility_matrix[stock] = signals["Position"].eq(1)

        minimum_history = (
            self.agent.first_score_position
            if self.agent.first_score_position is not None
            else len(self.data.index)
        )
        return self._build_target_weights(minimum_history=minimum_history)

    def plot_graph(self, *args, **kwargs):
        return self.agent.plot_graph(*args, **kwargs)

    def plot_direction_history(self, *args, **kwargs):
        return self.agent.plot_direction_history(*args, **kwargs)

    def plot_stock_signal(self, stock, *args, **kwargs):
        return self.agent.plot_stock_signal(stock, *args, **kwargs)
