from agents.technical.eigenportfolio_rotation_agent import EigenportfolioRotationAgent
from strategies.base_weighted_strategy import BaseWeightedStrategy


class EigenportfolioRotationStrategy(BaseWeightedStrategy):
    """Long-only rotation among momentum-ranked correlation eigen-sleeves."""

    def __init__(
        self,
        data,
        estimation_window=252,
        component_count=8,
        include_first_component=False,
        short_score_window=20,
        long_score_window=60,
        selected_sleeves=3,
        component_refit_frequency="monthly",
        rebalance_frequency="W-FRI",
        minimum_coverage=0.90,
        maximum_stock_weight=0.10,
        price_type="Close",
        transaction_cost=0.0,
        auto_generate=True,
        evaluation_start=None,
        performance_benchmark=None,
    ):
        self.agent_parameters = {
            "estimation_window": estimation_window,
            "component_count": component_count,
            "include_first_component": include_first_component,
            "short_score_window": short_score_window,
            "long_score_window": long_score_window,
            "selected_sleeves": selected_sleeves,
            "component_refit_frequency": component_refit_frequency,
            "decision_frequency": rebalance_frequency,
            "minimum_coverage": minimum_coverage,
            "maximum_stock_weight": maximum_stock_weight,
            "price_type": price_type,
        }
        super().__init__(
            data,
            "EigenportfolioRotation",
            rebalance_frequency,
            price_type,
            transaction_cost,
            evaluation_start,
            performance_benchmark,
        )
        if auto_generate:
            self.run_all()

    def generate_target_weights(self):
        self.agent = EigenportfolioRotationAgent(
            self.data,
            auto_generate=True,
            **self.agent_parameters,
        )
        valid_dates = self.agent.target_weight_matrix.index[
            self.agent.target_weight_matrix.notna().any(axis=1)
        ]
        if self.agent.first_weight_position is not None:
            first_weight_date = self.data.index[self.agent.first_weight_position]
            valid_dates = valid_dates[valid_dates >= first_weight_date]
        else:
            valid_dates = valid_dates[:0]
        return self._finalize_weight_targets(
            self.agent.target_weight_matrix,
            valid_dates,
        )

    def get_component_weights(self, *args, **kwargs):
        return self.agent.get_component_weights(*args, **kwargs)

    def component_summary(self, *args, **kwargs):
        return self.agent.component_summary(*args, **kwargs)

    def plot_explained_variance(self, *args, **kwargs):
        return self.agent.plot_explained_variance(*args, **kwargs)

    def plot_target_weights(self, *args, **kwargs):
        return self.agent.plot_target_weights(*args, **kwargs)
