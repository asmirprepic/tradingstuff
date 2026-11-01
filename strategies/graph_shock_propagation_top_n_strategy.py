import pandas as pd

from agents.technical.graph_shock_propagation_agent import GraphShockPropagationAgent
from strategies.base_top_n_strategy import BaseTopNStrategy


class GraphShockPropagationTopNStrategy(BaseTopNStrategy):
    """Long-only top-N portfolio from directed lag-network shock forecasts."""

    def __init__(
        self,
        data,
        estimation_window=252,
        max_lag=3,
        lasso_alpha=0.0005,
        shock_lookback=60,
        shock_z_threshold=2.0,
        minimum_observations=126,
        minimum_source_coverage=0.80,
        network_refit_frequency="monthly",
        sector_map=None,
        minimum_sector_size=3,
        top_n=5,
        rebalance_frequency="B",
        price_type="Close",
        transaction_cost=0.0,
        auto_generate=True,
        evaluation_start=None,
        performance_benchmark=None,
        execution_mode="next_open",
    ):
        self.agent_parameters = {
            "estimation_window": estimation_window,
            "max_lag": max_lag,
            "lasso_alpha": lasso_alpha,
            "shock_lookback": shock_lookback,
            "shock_z_threshold": shock_z_threshold,
            "minimum_observations": minimum_observations,
            "minimum_source_coverage": minimum_source_coverage,
            "network_refit_frequency": network_refit_frequency,
            "decision_frequency": rebalance_frequency,
            "sector_map": sector_map,
            "minimum_sector_size": minimum_sector_size,
            "price_type": price_type,
        }
        super().__init__(
            data,
            "GraphShockPropagationTopN",
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
        self.agent = GraphShockPropagationAgent(
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
            self.score_matrix[stock] = signals["PropagationScore"]
            self.eligibility_matrix[stock] = signals["Position"].eq(1)

        minimum_history = (
            self.agent.first_score_position
            if self.agent.first_score_position is not None
            else len(self.data.index)
        )
        return self._build_target_weights(minimum_history=minimum_history)

    def get_edge_table(self, *args, **kwargs):
        return self.agent.get_edge_table(*args, **kwargs)

    def propagation_contributions(self, *args, **kwargs):
        return self.agent.propagation_contributions(*args, **kwargs)

    def active_propagation_edges(self, *args, **kwargs):
        return self.agent.active_propagation_edges(*args, **kwargs)

    def plot_network(self, *args, **kwargs):
        return self.agent.plot_network(*args, **kwargs)

    def plot_propagation_contributions(self, *args, **kwargs):
        return self.agent.plot_propagation_contributions(*args, **kwargs)
