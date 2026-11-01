import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.technical.cross_sectional_entropy_agent import (
    CrossSectionalEntropyAgent,
)
from research.entropy_continuation_study import entropy_continuation_summary
from strategies import EntropyLeaderContinuationStrategy
from utilities.return_entropy import (
    calculate_return_entropy,
    select_contribution_leaders,
)


def make_entropy_data(weeks=80, stocks=12):
    index = pd.date_range("2024-01-01", periods=weeks * 5, freq="B")
    names = [f"S{number}" for number in range(stocks)]
    returns = pd.DataFrame(0.0, index=index, columns=names)
    friday_positions = [position for position, date in enumerate(index) if date.weekday() == 4]

    for week, position in enumerate(friday_positions):
        phase = week % 4
        leader = names[(week // 4) % stocks]
        if phase == 0:
            returns.iloc[position] = 0.01
        elif phase == 1:
            returns.iloc[position] = -0.002
            returns.iloc[position, returns.columns.get_loc(leader)] = 0.08
        elif phase == 2:
            returns.iloc[position] = 0.001
            returns.iloc[position, returns.columns.get_loc(leader)] = 0.04
        else:
            returns.iloc[position] = -0.003

    close = 100.0 * (1.0 + returns).cumprod()
    frames = []
    for stock in names:
        values = close[stock]
        frame = pd.DataFrame(
            {
                (stock, "Open"): values,
                (stock, "High"): values * 1.01,
                (stock, "Low"): values * 0.99,
                (stock, "Close"): values,
                (stock, "Volume"): 1000.0,
            },
            index=index,
        )
        frame.columns = pd.MultiIndex.from_tuples(frame.columns)
        frames.append(frame)
    return pd.concat(frames, axis=1)


def agent_parameters():
    return {
        "minimum_stocks": 8,
        "entropy_lookback": 12,
        "minimum_entropy_weeks": 6,
        "calibration_weeks": 30,
        "minimum_calibration_events": 3,
        "maximum_leaders": 6,
    }


class ReturnEntropyUtilityTests(unittest.TestCase):
    def test_normalized_entropy_distinguishes_broad_and_concentrated_returns(self):
        broad = calculate_return_entropy(pd.Series([0.01] * 10), minimum_stocks=2)
        concentrated = calculate_return_entropy(
            pd.Series([0.10] + [-0.01] * 9), minimum_stocks=2
        )

        self.assertAlmostEqual(broad["UpsideEntropy"], 1.0)
        self.assertAlmostEqual(concentrated["UpsideEntropy"], 0.0)
        self.assertEqual(concentrated["Leaders"], [0])

    def test_selects_smallest_set_reaching_contribution_threshold(self):
        shares = pd.Series({"A": 0.45, "B": 0.25, "C": 0.20, "D": 0.10})

        self.assertEqual(
            select_contribution_leaders(shares, cumulative_share=0.60),
            ["A", "B"],
        )


class CrossSectionalEntropyAgentTests(unittest.TestCase):
    def build_agent(self, data):
        return CrossSectionalEntropyAgent(data, **agent_parameters())

    def test_learns_positive_continuation_and_emits_signals(self):
        agent = self.build_agent(make_entropy_data())
        qualifying = agent.event_history.loc[
            agent.event_history["LowEntropy"]
            & agent.event_history["EntropyFalling"]
            & agent.event_history["PositiveMarketWeek"]
        ]

        self.assertFalse(agent.entropy_history.empty)
        self.assertFalse(qualifying.empty)
        self.assertGreater(qualifying["LeaderExcessReturn"].mean(), 0.0)
        self.assertTrue(agent.entropy_history["SignalActive"].any())
        self.assertIsNotNone(agent.first_tradable_position)

    def test_final_price_change_does_not_change_prior_scores(self):
        data = make_entropy_data()
        original = self.build_agent(data).score_matrix
        changed = data.copy()
        changed.loc[changed.index[-1], ("S0", "Close")] *= 1.5
        revised = self.build_agent(changed).score_matrix

        pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])

    def test_diagnostics_and_research_summary_are_available(self):
        agent = self.build_agent(make_entropy_data())
        summary = entropy_continuation_summary(agent.event_history, quantiles=3)
        contributions = agent.latest_leader_contributions()
        figure, axis = agent.plot_entropy_history()

        self.assertFalse(summary.empty)
        self.assertIn("HAC_t", summary.columns)
        self.assertFalse(contributions.empty)
        self.assertTrue(agent.calibration_status()["EntropyReady"])
        self.assertIs(axis.figure, figure)
        plt.close(figure)


class EntropyLeaderContinuationStrategyTests(unittest.TestCase):
    def build_strategy(self):
        return EntropyLeaderContinuationStrategy(
            make_entropy_data(),
            maximum_stock_weight=0.30,
            **agent_parameters(),
        )

    def test_uses_capped_long_only_weights_on_next_bar(self):
        strategy = self.build_strategy()
        active_dates = strategy.agent.entropy_history.index[
            strategy.agent.entropy_history["SignalActive"]
        ]
        decision_date = active_dates[0]
        next_date = strategy.data.index[
            strategy.data.index.get_loc(decision_date) + 1
        ]

        self.assertEqual(strategy.effective_weights.loc[decision_date].sum(), 0.0)
        self.assertGreater(strategy.effective_weights.loc[next_date].sum(), 0.0)
        self.assertTrue(strategy.target_weights.ge(0.0).all().all())
        self.assertLessEqual(strategy.target_weights.max().max(), 0.30 + 1e-12)

    def test_inactive_decision_exits_to_cash(self):
        strategy = self.build_strategy()
        history = strategy.agent.entropy_history
        active_positions = np.flatnonzero(history["SignalActive"].to_numpy())
        active_position = active_positions[0]
        later = history.iloc[active_position + 1:]
        inactive_date = later.index[~later["SignalActive"]][0]

        self.assertAlmostEqual(strategy.target_weights.loc[inactive_date].sum(), 0.0)

    def test_performance_exists_even_when_calibration_never_reaches_requirement(self):
        parameters = agent_parameters()
        parameters["minimum_calibration_events"] = 100
        strategy = EntropyLeaderContinuationStrategy(
            make_entropy_data(),
            maximum_stock_weight=0.30,
            **parameters,
        )

        self.assertFalse(strategy.evaluate_performance().empty)
        self.assertEqual(strategy.portfolio_returns.sum(), 0.0)
        self.assertFalse(strategy.calibration_status()["CalibrationReady"])


if __name__ == "__main__":
    unittest.main()
