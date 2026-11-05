from agents.technical.volume_matrix_profile_agent import VolumeMatrixProfileAgent
from strategies.base_top_n_strategy import BaseTopNStrategy


class VolumeMatrixProfileTopNStrategy(BaseTopNStrategy):
    """Weekly Top-N analogue strategy using return and relative-volume patterns."""

    def __init__(
        self,
        data,
        pattern_window=30,
        volume_lookback=20,
        volume_weight=0.25,
        return_candidate_pool=100,
        forecast_horizons=(5, 10, 20),
        primary_horizon=5,
        neighbours=30,
        maximum_matches_per_stock=3,
        minimum_neighbours=10,
        return_maximum_distance=1.25,
        maximum_distance=1.40,
        distance_temperature=0.25,
        history_window=500,
        candidate_stride=5,
        top_n=5,
        rebalance_frequency="W-FRI",
        price_type="Close",
        transaction_cost=0.0,
        auto_generate=True,
        evaluation_start=None,
        performance_benchmark=None,
        execution_mode="next_open",
    ):
        self.agent_parameters = {
            "pattern_window": pattern_window,
            "volume_lookback": volume_lookback,
            "volume_weight": volume_weight,
            "return_candidate_pool": return_candidate_pool,
            "forecast_horizons": forecast_horizons,
            "primary_horizon": primary_horizon,
            "neighbours": neighbours,
            "maximum_matches_per_stock": maximum_matches_per_stock,
            "minimum_neighbours": minimum_neighbours,
            "return_maximum_distance": return_maximum_distance,
            "maximum_distance": maximum_distance,
            "distance_temperature": distance_temperature,
            "history_window": history_window,
            "candidate_stride": candidate_stride,
            "decision_frequency": rebalance_frequency,
            "price_type": price_type,
        }
        super().__init__(
            data,
            "VolumeMatrixProfileTopN",
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
        self.agent = VolumeMatrixProfileAgent(
            self.data,
            auto_generate=True,
            **self.agent_parameters,
        )
        self.score_matrix = self.agent.forecast_matrix.copy()
        self.eligibility_matrix = self.score_matrix.gt(0.0)
        minimum_history = (
            self.agent.first_score_position
            if self.agent.first_score_position is not None
            else len(self.data.index)
        )
        return self._build_target_weights(minimum_history=minimum_history)

    def get_matches(self, stock):
        return self.agent.get_matches(stock)
