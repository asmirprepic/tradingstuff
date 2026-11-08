import unittest

import numpy as np
import pandas as pd

from agents.technical.matrix_profile_agent import MatrixProfileAgent
from strategies import MatrixProfileTopNStrategy
from strategies import VolumeMatrixProfileTopNStrategy
from agents.technical.volume_matrix_profile_agent import VolumeMatrixProfileAgent
from utilities.matrix_profile_utils import (
    build_pattern_library,
    normalize_return_pattern,
)


def make_pattern_data(periods=220, stocks=6):
    rng = np.random.default_rng(17)
    index = pd.date_range("2023-01-02", periods=periods, freq="B")
    frames = []
    time = np.arange(periods)
    for number in range(stocks):
        returns = (
            0.001
            + 0.009 * np.sin(2.0 * np.pi * (time + number * 3) / 40.0)
            + rng.normal(0.0, 0.0015, periods)
        )
        close = 100.0 * np.exp(np.cumsum(returns))
        stock = f"S{number}"
        frame = pd.DataFrame(
            {
                (stock, "Open"): close * (1.0 - 0.001),
                (stock, "High"): close * 1.01,
                (stock, "Low"): close * 0.99,
                (stock, "Close"): close,
                (stock, "Volume"): 1000.0 * np.exp(
                    0.35 * np.sin(2.0 * np.pi * (time + number * 2) / 40.0)
                    + rng.normal(0.0, 0.04, periods)
                ),
            },
            index=index,
        )
        frame.columns = pd.MultiIndex.from_tuples(frame.columns)
        frames.append(frame)
    return pd.concat(frames, axis=1)


def agent_parameters():
    return {
        "pattern_window": 10,
        "forecast_horizons": (5, 10),
        "primary_horizon": 5,
        "neighbours": 8,
        "maximum_matches_per_stock": 2,
        "minimum_neighbours": 4,
        "maximum_distance": None,
        "distance_temperature": 0.30,
        "history_window": 150,
        "candidate_stride": 2,
    }


class MatrixProfileUtilityTests(unittest.TestCase):
    def test_normalizes_patterns_and_builds_forward_outcomes(self):
        normalized = normalize_return_pattern([1.0, 2.0, 3.0, 4.0])
        close = make_pattern_data(periods=80, stocks=3).xs(
            "Close", level=1, axis=1
        )
        patterns, metadata, returns = build_pattern_library(
            close,
            pattern_window=10,
            forecast_horizons=(5, 10),
            candidate_stride=2,
        )

        self.assertAlmostEqual(float(normalized.mean()), 0.0)
        self.assertAlmostEqual(float(normalized.std(ddof=0)), 1.0)
        self.assertEqual(patterns.shape[1], 10)
        self.assertIn("ForwardReturn_10", metadata.columns)
        self.assertEqual(returns.shape, close.shape)


class MatrixProfileAgentTests(unittest.TestCase):
    def build_agent(self, data):
        return MatrixProfileAgent(data, **agent_parameters())

    def test_generates_forecasts_and_causal_matches(self):
        data = make_pattern_data()
        agent = self.build_agent(data)
        forecasts = agent.forecast_matrix.stack().dropna()
        matches = agent.get_matches("S0")
        latest_date = agent.forecast_matrix.dropna(how="all").index[-1]
        query_position = data.index.get_loc(latest_date)
        query_start = query_position - agent.pattern_window + 1
        same_stock = matches.loc[matches["Stock"].eq("S0")]

        self.assertFalse(forecasts.empty)
        self.assertGreaterEqual(agent.neighbour_count_matrix.max().max(), 4)
        self.assertTrue(
            same_stock["EndPosition"].add(max(agent.forecast_horizons)).lt(
                query_start
            ).all()
        )

    def test_final_price_change_does_not_change_prior_forecasts(self):
        data = make_pattern_data()
        original = self.build_agent(data).forecast_matrix
        changed = data.copy()
        changed.loc[changed.index[-1], ("S0", "Close")] *= 1.25
        revised = self.build_agent(changed).forecast_matrix

        pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])


class MatrixProfileStrategyTests(unittest.TestCase):
    def test_builds_weekly_next_open_top_n_portfolio(self):
        strategy = MatrixProfileTopNStrategy(
            make_pattern_data(),
            top_n=3,
            execution_mode="next_open",
            **agent_parameters(),
        )
        selected = strategy.selection_log.index[
            strategy.selection_log["N_Held"].gt(0)
        ]
        first_decision = selected[0]
        execution_date = strategy.data.index[
            strategy.data.index.get_loc(first_decision) + 1
        ]

        self.assertLessEqual(strategy.target_weights.gt(0.0).sum(axis=1).max(), 3)
        self.assertEqual(strategy.effective_weights.loc[first_decision].sum(), 0.0)
        self.assertTrue(strategy.selection_log.loc[execution_date, "Execution"])
        self.assertGreater(strategy.effective_weights.loc[execution_date].sum(), 0.0)


class VolumeMatrixProfileTests(unittest.TestCase):
    def volume_parameters(self):
        return {
            **agent_parameters(),
            "volume_lookback": 10,
            "volume_weight": 0.25,
            "return_candidate_pool": 20,
            "return_maximum_distance": None,
            "maximum_distance": None,
        }

    def test_volume_agent_reranks_return_candidates(self):
        agent = VolumeMatrixProfileAgent(
            make_pattern_data(),
            **self.volume_parameters(),
        )
        matches = agent.get_matches("S0")

        self.assertFalse(agent.forecast_matrix.stack().dropna().empty)
        self.assertIn("ReturnDistance", matches.columns)
        self.assertIn("VolumeDistance", matches.columns)
        self.assertTrue(matches["Distance"].is_monotonic_increasing)
        self.assertTrue(
            agent.signal_data["S0"]["RelativeLogVolume"].notna().any()
        )

    def test_volume_strategy_builds_next_open_portfolio(self):
        strategy = VolumeMatrixProfileTopNStrategy(
            make_pattern_data(),
            top_n=3,
            execution_mode="next_open",
            **self.volume_parameters(),
        )
        selected = strategy.selection_log.index[
            strategy.selection_log["N_Held"].gt(0)
        ]
        execution_date = strategy.data.index[
            strategy.data.index.get_loc(selected[0]) + 1
        ]

        self.assertTrue(strategy.selection_log.loc[execution_date, "Execution"])
        self.assertLessEqual(strategy.target_weights.gt(0.0).sum(axis=1).max(), 3)

    def test_final_price_and_volume_change_do_not_change_prior_forecasts(self):
        data = make_pattern_data()
        original = VolumeMatrixProfileAgent(
            data,
            **self.volume_parameters(),
        ).forecast_matrix
        changed = data.copy()
        changed.loc[changed.index[-1], ("S0", "Close")] *= 1.25
        changed.loc[changed.index[-1], ("S0", "Volume")] *= 8.0
        revised = VolumeMatrixProfileAgent(
            changed,
            **self.volume_parameters(),
        ).forecast_matrix

        pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])

    def test_volume_agent_requires_volume_data(self):
        data = make_pattern_data().drop(columns="Volume", level=1)

        with self.assertRaisesRegex(ValueError, "requires Volume"):
            VolumeMatrixProfileAgent(
                data,
                auto_generate=False,
                **self.volume_parameters(),
            )


if __name__ == "__main__":
    unittest.main()
