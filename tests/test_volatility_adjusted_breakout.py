import unittest

import numpy as np
import pandas as pd

from agents.technical.volatility_adjusted_breakout_agent import (
    VolatilityAdjustedBreakoutAgent,
)
from strategies import VolatilityAdjustedBreakoutStrategy


def make_breakout_data(periods=80):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    base = np.full(periods, 100.0)
    prices = {
        "STRONG": base.copy(),
        "WEAK": base.copy(),
        "FLAT": base.copy(),
    }
    breakout_start = 60
    steps = np.arange(1, periods - breakout_start + 1)
    prices["STRONG"][breakout_start:] = 100.0 + 3.0 * steps
    prices["WEAK"][breakout_start:] = 100.0 + 1.0 * steps

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


class VolatilityAdjustedBreakoutAgentTests(unittest.TestCase):
    def test_prior_high_excludes_current_price(self):
        data = make_breakout_data()
        agent = VolatilityAdjustedBreakoutAgent(
            data, breakout_window=60, atr_window=20
        )
        signals = agent.signal_data["STRONG"]
        date = data.index[60]

        self.assertEqual(signals.loc[date, "PriorHigh"], 100.0)
        self.assertGreater(signals.loc[date, "BreakoutScore"], 0.0)
        self.assertEqual(signals.loc[date, "Position"], 1)

    def test_future_price_change_does_not_change_prior_scores(self):
        data = make_breakout_data()
        original = VolatilityAdjustedBreakoutAgent(
            data, breakout_window=20, atr_window=10
        ).signal_data["STRONG"]["BreakoutScore"]
        changed = data.copy()
        changed.loc[changed.index[-1], ("STRONG", "Close")] *= 2.0
        revised = VolatilityAdjustedBreakoutAgent(
            changed, breakout_window=20, atr_window=10
        ).signal_data["STRONG"]["BreakoutScore"]

        pd.testing.assert_series_equal(original.iloc[:-1], revised.iloc[:-1])


class VolatilityAdjustedBreakoutStrategyTests(unittest.TestCase):
    def test_selects_strongest_normalized_breakout(self):
        strategy = VolatilityAdjustedBreakoutStrategy(
            make_breakout_data(),
            breakout_window=60,
            atr_window=20,
            top_n=1,
        )
        selections = strategy.selection_log.loc[
            strategy.selection_log["Rebalance"]
            & strategy.selection_log["N_Held"].gt(0),
            "Selected_Stocks",
        ]

        self.assertFalse(selections.empty)
        self.assertTrue(all(selected == ["STRONG"] for selected in selections))

    def test_applies_weekly_selection_on_next_bar(self):
        strategy = VolatilityAdjustedBreakoutStrategy(
            make_breakout_data(),
            breakout_window=60,
            atr_window=20,
            top_n=1,
        )
        selected_date = strategy.selection_log.index[
            strategy.selection_log["N_Held"].gt(0)
        ][0]
        next_date = strategy.data.index[strategy.data.index.get_loc(selected_date) + 1]

        self.assertEqual(strategy.effective_weights.loc[selected_date, "STRONG"], 0.0)
        self.assertEqual(strategy.effective_weights.loc[next_date, "STRONG"], 1.0)

    def test_holds_after_breakout_until_trailing_stop_is_crossed(self):
        data = make_breakout_data()
        close_column = ("STRONG", "Close")
        data.loc[data.index[60], close_column] = 110.0
        data.loc[data.index[61:66], close_column] = 109.0
        data.loc[data.index[66], close_column] = 95.0
        for position in range(60, 67):
            close = data.iloc[position][close_column]
            data.loc[data.index[position], ("STRONG", "Open")] = close
            data.loc[data.index[position], ("STRONG", "High")] = close + 1.0
            data.loc[data.index[position], ("STRONG", "Low")] = close - 1.0

        strategy = VolatilityAdjustedBreakoutStrategy(
            data,
            breakout_window=60,
            atr_window=20,
            stop_atr_multiple=3.0,
            top_n=1,
        )

        self.assertTrue(strategy.entry_matrix.loc[data.index[60], "STRONG"])
        self.assertLessEqual(strategy.score_matrix.loc[data.index[61], "STRONG"], 0.0)
        self.assertEqual(strategy.target_weights.loc[data.index[65], "STRONG"], 1.0)
        self.assertTrue(strategy.exit_matrix.loc[data.index[66], "STRONG"])
        self.assertEqual(strategy.target_weights.loc[data.index[66], "STRONG"], 0.0)
        self.assertEqual(strategy.effective_weights.loc[data.index[66], "STRONG"], 1.0)
        self.assertEqual(strategy.effective_weights.loc[data.index[67], "STRONG"], 0.0)

    def test_validates_windows(self):
        with self.assertRaises(ValueError):
            VolatilityAdjustedBreakoutStrategy(
                make_breakout_data(), breakout_window=0, auto_generate=False
            )

        with self.assertRaises(ValueError):
            VolatilityAdjustedBreakoutStrategy(
                make_breakout_data(), stop_atr_multiple=0, auto_generate=False
            )


if __name__ == "__main__":
    unittest.main()
