import unittest

import numpy as np
import pandas as pd

from strategies.base_top_n_strategy import BaseTopNStrategy


def make_execution_data(include_open=True):
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    values = {
        ("A", "High"): [101.0, 111.0, 243.0, 267.0],
        ("A", "Low"): [99.0, 109.0, 219.0, 265.0],
        ("A", "Close"): [100.0, 110.0, 242.0, 266.2],
        ("A", "Volume"): [1000.0] * 4,
    }
    if include_open:
        values[("A", "Open")] = [100.0, 100.0, 220.0, 242.0]
    frame = pd.DataFrame(values, index=index)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame.sort_index(axis=1)


class FixedTargetStrategy(BaseTopNStrategy):
    def __init__(self, data, execution_mode, transaction_cost=0.0):
        super().__init__(
            data,
            "FixedTarget",
            top_n=1,
            rebalance_frequency="B",
            transaction_cost=transaction_cost,
            execution_mode=execution_mode,
        )
        self.run_all()

    def generate_target_weights(self):
        index = self.data.index
        decision_date = index[1]
        weights = pd.DataFrame(np.nan, index=index, columns=self.stocks_in_data)
        weights.loc[decision_date] = 0.0
        weights.loc[decision_date, "A"] = 1.0
        self.target_weights = weights.ffill().fillna(0.0)
        self.effective_weights = self.target_weights.shift(1).fillna(0.0)
        self.weight_rebalance_dates = pd.DatetimeIndex([decision_date])
        self.evaluation_start = index[2]
        return self.target_weights


class ExecutionModeTests(unittest.TestCase):
    def test_next_open_excludes_pre_execution_overnight_return(self):
        strategy = FixedTargetStrategy(make_execution_data(), "next_open")
        execution_date = strategy.data.index[2]

        self.assertAlmostEqual(strategy.overnight_returns.loc[execution_date], 0.0)
        self.assertAlmostEqual(strategy.intraday_returns.loc[execution_date], 0.10)
        self.assertAlmostEqual(strategy.portfolio_returns.loc[execution_date], 0.10)
        self.assertTrue(strategy.selection_log.loc[execution_date, "Execution"])

    def test_next_close_waits_until_after_execution_day_return(self):
        strategy = FixedTargetStrategy(make_execution_data(), "next_close")
        execution_date = strategy.data.index[2]
        following_date = strategy.data.index[3]

        self.assertAlmostEqual(strategy.portfolio_returns.loc[execution_date], 0.0)
        self.assertAlmostEqual(strategy.portfolio_returns.loc[following_date], 0.10)
        self.assertEqual(strategy.effective_weights.loc[execution_date, "A"], 0.0)
        self.assertEqual(strategy.effective_weights.loc[following_date, "A"], 1.0)

    def test_legacy_mode_reproduces_same_close_style_return(self):
        strategy = FixedTargetStrategy(make_execution_data(), "legacy_same_close")
        execution_date = strategy.data.index[2]

        self.assertAlmostEqual(strategy.portfolio_returns.loc[execution_date], 1.20)

    def test_next_open_applies_transaction_cost_at_open(self):
        strategy = FixedTargetStrategy(
            make_execution_data(), "next_open", transaction_cost=0.01
        )
        execution_date = strategy.data.index[2]

        self.assertAlmostEqual(strategy.selection_log.loc[execution_date, "Turnover"], 1.0)
        self.assertAlmostEqual(
            strategy.portfolio_returns.loc[execution_date],
            0.99 * 1.10 - 1.0,
        )

    def test_next_open_requires_open_prices(self):
        with self.assertRaisesRegex(ValueError, "requires Open prices"):
            FixedTargetStrategy(make_execution_data(include_open=False), "next_open")

    def test_rejects_unknown_execution_mode(self):
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            FixedTargetStrategy(make_execution_data(), "unknown")


if __name__ == "__main__":
    unittest.main()
