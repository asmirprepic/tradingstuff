import pandas as pd

from agents.technical.cross_sectional_entropy_agent import (
    CrossSectionalEntropyAgent,
)
from strategies.base_weighted_strategy import BaseWeightedStrategy
from utilities.eigenportfolio_utils import cap_and_normalize_long_weights


class EntropyLeaderContinuationStrategy(BaseWeightedStrategy):
    """Hold recent return leaders only when low entropy predicts continuation."""

    def __init__(
        self,
        data,
        leader_contribution_share=0.60,
        maximum_leaders=10,
        minimum_stocks=10,
        entropy_quantile=0.30,
        entropy_lookback=104,
        minimum_entropy_weeks=26,
        forward_horizon_weeks=1,
        calibration_weeks=104,
        minimum_calibration_events=8,
        require_positive_market_week=True,
        require_falling_entropy=True,
        maximum_stock_weight=0.25,
        rebalance_frequency="W-FRI",
        price_type="Close",
        transaction_cost=0.0,
        auto_generate=True,
        evaluation_start=None,
        performance_benchmark=None,
    ):
        self.maximum_stock_weight = float(maximum_stock_weight)
        if not 0.0 < self.maximum_stock_weight <= 1.0:
            raise ValueError("maximum_stock_weight must be in (0, 1].")
        self.agent_parameters = {
            "leader_contribution_share": leader_contribution_share,
            "maximum_leaders": maximum_leaders,
            "minimum_stocks": minimum_stocks,
            "entropy_quantile": entropy_quantile,
            "entropy_lookback": entropy_lookback,
            "minimum_entropy_weeks": minimum_entropy_weeks,
            "forward_horizon_weeks": forward_horizon_weeks,
            "calibration_weeks": calibration_weeks,
            "minimum_calibration_events": minimum_calibration_events,
            "require_positive_market_week": require_positive_market_week,
            "require_falling_entropy": require_falling_entropy,
            "decision_frequency": rebalance_frequency,
            "price_type": price_type,
        }
        super().__init__(
            data,
            "EntropyLeaderContinuation",
            rebalance_frequency,
            price_type,
            transaction_cost,
            evaluation_start,
            performance_benchmark,
        )
        if auto_generate:
            self.run_all()

    def generate_target_weights(self):
        self.agent = CrossSectionalEntropyAgent(
            self.data,
            auto_generate=True,
            **self.agent_parameters,
        )
        targets = pd.DataFrame(
            float("nan"),
            index=self.data.index,
            columns=self.stocks_in_data,
        )
        decision_dates = self.agent.entropy_history.index
        for date in decision_dates:
            targets.loc[date] = 0.0
            scores = self.agent.score_matrix.loc[date].clip(lower=0.0).fillna(0.0)
            if scores.sum() > 0.0:
                targets.loc[date] = cap_and_normalize_long_weights(
                    scores,
                    maximum_weight=self.maximum_stock_weight,
                )

        if self.agent.first_tradable_position is None:
            valid_dates = decision_dates[:0]
        else:
            first_date = self.data.index[self.agent.first_tradable_position]
            valid_dates = decision_dates[decision_dates >= first_date]
        return self._finalize_weight_targets(targets, valid_dates)

    def entropy_continuation_study(self, quantiles=5):
        from research.entropy_continuation_study import entropy_continuation_summary

        return entropy_continuation_summary(self.agent.event_history, quantiles=quantiles)

    def plot_entropy_history(self, *args, **kwargs):
        return self.agent.plot_entropy_history(*args, **kwargs)

    def latest_leader_contributions(self):
        return self.agent.latest_leader_contributions()

    def calibration_status(self):
        return self.agent.calibration_status()
