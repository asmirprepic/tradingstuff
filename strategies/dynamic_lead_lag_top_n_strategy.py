import pandas as pd

from agents.technical.dynamic_lead_lag_agent import DynamicLeadLagAgent
from strategies.base_top_n_strategy import BaseTopNStrategy


class DynamicLeadLagTopNStrategy(BaseTopNStrategy):
    """Daily long-only top-N portfolio from directed lead-lag forecasts."""

    def __init__(
        self,
        data,
        estimation_window=252,
        max_lag=3,
        maximum_leaders_per_target=15,
        lasso_alpha=0.0001,
        minimum_observations=126,
        minimum_source_coverage=0.80,
        network_refit_frequency="monthly",
        adjust_common_effects=True,
        sector_map=None,
        minimum_sector_size=3,
        top_n=5,
        rebalance_frequency="B",
        price_type="Close",
        transaction_cost=0.0,
        auto_generate=True,
        evaluation_start=None,
        performance_benchmark=None,
    ):
        self.agent_parameters = {
            "estimation_window": estimation_window,
            "max_lag": max_lag,
            "maximum_leaders_per_target": maximum_leaders_per_target,
            "lasso_alpha": lasso_alpha,
            "minimum_observations": minimum_observations,
            "minimum_source_coverage": minimum_source_coverage,
            "network_refit_frequency": network_refit_frequency,
            "decision_frequency": rebalance_frequency,
            "adjust_common_effects": adjust_common_effects,
            "sector_map": sector_map,
            "minimum_sector_size": minimum_sector_size,
            "price_type": price_type,
        }
        super().__init__(
            data,
            "DynamicLeadLagTopN",
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
        self.agent = DynamicLeadLagAgent(
            self.data,
            auto_generate=True,
            **self.agent_parameters,
        )
        self.score_matrix = self.agent.score_matrix.copy()
        self.eligibility_matrix = self.score_matrix.gt(0.0)
        minimum_history = (
            self.agent.first_score_position
            if self.agent.first_score_position is not None
            else len(self.data.index)
        )
        return self._build_target_weights(minimum_history=minimum_history)

    def get_edge_table(self, *args, **kwargs):
        return self.agent.get_edge_table(*args, **kwargs)

    def leaders_for_target(self, *args, **kwargs):
        return self.agent.leaders_for_target(*args, **kwargs)

    def plot_network(self, *args, **kwargs):
        return self.agent.plot_network(*args, **kwargs)
