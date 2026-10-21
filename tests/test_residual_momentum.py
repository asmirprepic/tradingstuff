import unittest

import numpy as np
import pandas as pd

from agents.technical.residual_momentum_agent import ResidualMomentumAgent
from strategies.residual_momentum_top_n_strategy import ResidualMomentumTopNStrategy


def make_residual_data(periods=40):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    market = pd.Series(
        0.003 * np.sin(np.arange(periods) * 1.7)
        + 0.002 * np.cos(np.arange(periods) * 0.6),
        index=index,
        name="Market",
    )
    alpha_residual = np.zeros(periods)
    alpha_residual[6:] = 0.002 + 0.001 * np.sin(np.arange(periods - 6) * 1.3)
    low_residual = np.zeros(periods)
    low_residual[6:] = -0.001
    returns = {
        "ALPHA": 0.0005 + 1.2 * market.to_numpy() + alpha_residual,
        "LOW": 0.0005 + 1.2 * market.to_numpy() + low_residual,
    }

    frames = []
    for ticker, log_returns in returns.items():
        close = 100.0 * np.exp(np.cumsum(log_returns))
        frame = pd.DataFrame(
            {
                (ticker, "Open"): close,
                (ticker, "High"): close + 1.0,
                (ticker, "Low"): close - 1.0,
                (ticker, "Close"): close,
                (ticker, "Volume"): 1000.0,
            },
            index=index,
        )
        frame.columns = pd.MultiIndex.from_tuples(frame.columns)
        frames.append(frame)
    return pd.concat(frames, axis=1), market


class ResidualMomentumAgentTests(unittest.TestCase):
    def test_regression_at_t_uses_only_returns_through_t_minus_one(self):
        index = pd.date_range("2024-01-01", periods=12, freq="B")
        market = pd.Series(
            [0.0, -0.004, 0.002, 0.005, -0.003, 0.001, 0.004, -0.002, 0.003, 0.001, -0.001, 0.002],
            index=index,
        )
        stock_returns = 0.001 + 1.5 * market.to_numpy()
        stock_returns[6] += 0.02
        close = 100.0 * np.exp(np.cumsum(stock_returns))
        data = pd.DataFrame({("AAA", "Close"): close}, index=index)
        data.columns = pd.MultiIndex.from_tuples(data.columns)

        agent = ResidualMomentumAgent(
            data,
            regression_window=5,
            momentum_window=1,
            benchmark_returns=market,
        )
        row = agent.signal_data["AAA"].iloc[6]

        self.assertAlmostEqual(row["Alpha"], 0.001, places=10)
        self.assertAlmostEqual(row["Beta"], 1.5, places=10)
        self.assertAlmostEqual(row["ResidualReturn"], 0.02, places=10)

    def test_changing_final_price_does_not_change_prior_signals(self):
        data, market = make_residual_data()
        original = ResidualMomentumAgent(
            data, regression_window=5, momentum_window=3, benchmark_returns=market
        ).signal_data["ALPHA"]
        changed = data.copy()
        changed.loc[changed.index[-1], ("ALPHA", "Close")] *= 2.0
        revised = ResidualMomentumAgent(
            changed, regression_window=5, momentum_window=3, benchmark_returns=market
        ).signal_data["ALPHA"]

        pd.testing.assert_series_equal(
            original["ResidualMomentum"].iloc[:-1],
            revised["ResidualMomentum"].iloc[:-1],
        )

    def test_default_benchmark_is_equal_weight_universe_return(self):
        data, _ = make_residual_data()
        agent = ResidualMomentumAgent(
            data, regression_window=5, momentum_window=3, auto_generate=False
        )
        close = data.xs("Close", level=1, axis=1)
        expected = np.log(close / close.shift(1)).mean(axis=1)

        pd.testing.assert_series_equal(
            agent.benchmark_returns,
            expected.rename("BenchmarkReturn"),
        )


class ResidualMomentumTopNStrategyTests(unittest.TestCase):
    def test_selects_highest_positive_residual_momentum(self):
        data, market = make_residual_data()
        strategy = ResidualMomentumTopNStrategy(
            data,
            regression_window=5,
            momentum_window=3,
            benchmark_returns=market,
            top_n=1,
        )
        selections = strategy.selection_log.loc[
            strategy.selection_log["Rebalance"] & strategy.selection_log["N_Held"].gt(0),
            "Selected_Stocks",
        ]

        self.assertFalse(selections.empty)
        self.assertTrue(all(selected == ["ALPHA"] for selected in selections))

    def test_strategy_delays_selected_weight_by_one_bar(self):
        data, market = make_residual_data()
        strategy = ResidualMomentumTopNStrategy(
            data,
            regression_window=5,
            momentum_window=3,
            benchmark_returns=market,
            top_n=1,
        )
        selected_date = strategy.selection_log.index[strategy.selection_log["N_Held"].gt(0)][0]
        next_date = strategy.data.index[strategy.data.index.get_loc(selected_date) + 1]

        self.assertEqual(strategy.effective_weights.loc[selected_date, "ALPHA"], 0.0)
        self.assertEqual(strategy.effective_weights.loc[next_date, "ALPHA"], 1.0)

    def test_fixed_evaluation_start_keeps_benchmark_constant_across_windows(self):
        data, market = make_residual_data()
        evaluation_start = data.index[20]
        short_window = ResidualMomentumTopNStrategy(
            data,
            regression_window=5,
            momentum_window=3,
            benchmark_returns=market,
            top_n=1,
            evaluation_start=evaluation_start,
        )
        long_window = ResidualMomentumTopNStrategy(
            data,
            regression_window=8,
            momentum_window=3,
            benchmark_returns=market,
            top_n=1,
            evaluation_start=evaluation_start,
        )

        self.assertEqual(short_window.evaluation_start, evaluation_start)
        self.assertEqual(long_window.evaluation_start, evaluation_start)
        pd.testing.assert_series_equal(
            short_window.benchmark_returns,
            long_window.benchmark_returns,
        )

    def test_fixed_evaluation_start_requires_enough_prior_history(self):
        data, market = make_residual_data()

        with self.assertRaisesRegex(ValueError, "first executable date"):
            ResidualMomentumTopNStrategy(
                data,
                regression_window=8,
                momentum_window=3,
                benchmark_returns=market,
                top_n=1,
                evaluation_start=data.index[10],
            )


if __name__ == "__main__":
    unittest.main()
