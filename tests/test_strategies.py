import unittest

import numpy as np
import pandas as pd

from strategies.moving_average_top_n_strategy import MovingAverageTopNStrategy


def make_strategy_data(periods=15):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    prices = {
        "FAST": 100.0 * np.power(1.04, np.arange(periods)),
        "SLOW": 100.0 * np.power(1.01, np.arange(periods)),
        "DOWN": 100.0 * np.power(0.99, np.arange(periods)),
    }
    frames = []
    for ticker, close in prices.items():
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
    return pd.concat(frames, axis=1)


class MovingAverageTopNStrategyTests(unittest.TestCase):
    def test_selects_strongest_eligible_stock(self):
        strategy = MovingAverageTopNStrategy(
            make_strategy_data(), short_window=2, long_window=4, top_n=1
        )
        rebalance_rows = strategy.selection_log[
            strategy.selection_log["Rebalance"] & strategy.selection_log["N_Held"].gt(0)
        ]

        self.assertFalse(rebalance_rows.empty)
        self.assertEqual(rebalance_rows.iloc[0]["Selected_Stocks"], ["FAST"])
        self.assertEqual(rebalance_rows.iloc[0]["InvestedWeight"], 1.0)

    def test_keeps_empty_slots_in_cash(self):
        data = make_strategy_data().drop(columns=[("SLOW", field) for field in data_fields()])
        strategy = MovingAverageTopNStrategy(
            data, short_window=2, long_window=4, top_n=2
        )
        invested = strategy.selection_log.loc[
            strategy.selection_log["N_Held"].gt(0), "InvestedWeight"
        ]

        self.assertTrue(np.allclose(invested, 0.5))

    def test_applies_target_weights_on_next_bar(self):
        strategy = MovingAverageTopNStrategy(
            make_strategy_data(), short_window=2, long_window=4, top_n=1
        )
        first_selected = strategy.selection_log.index[strategy.selection_log["N_Held"].gt(0)][0]
        position = strategy.data.index.get_loc(first_selected)
        next_date = strategy.data.index[position + 1]

        self.assertEqual(strategy.effective_weights.loc[first_selected, "FAST"], 0.0)
        self.assertEqual(strategy.effective_weights.loc[next_date, "FAST"], 1.0)

    def test_returns_portfolio_and_benchmark_metrics(self):
        strategy = MovingAverageTopNStrategy(
            make_strategy_data(), short_window=2, long_window=4, top_n=1
        )
        performance = strategy.evaluate_performance()

        self.assertEqual(performance["Portfolio"].tolist(), [
            "MovingAverageTopN",
            "EqualWeightBuyHold",
        ])
        self.assertTrue(np.isfinite(performance.loc[0, "Total_Return_%"]))

    def test_validates_top_n(self):
        with self.assertRaises(ValueError):
            MovingAverageTopNStrategy(make_strategy_data(), top_n=4, auto_generate=False)


def data_fields():
    return ["Open", "High", "Low", "Close", "Volume"]


if __name__ == "__main__":
    unittest.main()
