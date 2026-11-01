import unittest
import warnings

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

    def test_timezone_aware_data_preserves_timezone_without_period_warning(self):
        data = make_strategy_data()
        data.index = data.index.tz_localize("Europe/Stockholm")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            strategy = MovingAverageTopNStrategy(
                data, short_window=2, long_window=4, top_n=1
            )

        timezone_warnings = [
            warning for warning in captured if "drop timezone" in str(warning.message)
        ]
        self.assertEqual(timezone_warnings, [])
        self.assertEqual(strategy._rebalance_dates().tz, data.index.tz)

    def test_incomplete_final_week_is_not_a_rebalance_date(self):
        data = make_strategy_data(periods=16)
        strategy = MovingAverageTopNStrategy(
            data, short_window=2, long_window=4, top_n=1
        )

        self.assertEqual(data.index[-1].weekday(), 0)
        self.assertNotIn(data.index[-1], strategy._rebalance_dates())
        self.assertEqual(strategy._rebalance_dates()[-1].weekday(), 4)

    def test_price_confirmation_excludes_stock_below_long_average(self):
        data = make_strategy_data()
        data.loc[data.index[:5], ("FAST", "Close")] = [100, 100, 110, 110, 105]
        decision_date = data.index[4]
        strategy = MovingAverageTopNStrategy(
            data,
            short_window=2,
            long_window=4,
            top_n=1,
            require_price_above_long=True,
        )
        signals = strategy.agent.signal_data["FAST"].loc[decision_date]

        self.assertEqual(signals["Position"], 1)
        self.assertFalse(signals["PriceAboveLong"])
        self.assertFalse(signals["StrategyEligible"])
        self.assertEqual(strategy.target_weights.loc[decision_date, "FAST"], 0.0)

    def test_rising_long_confirmation_uses_requested_lookback(self):
        strategy = MovingAverageTopNStrategy(
            make_strategy_data(),
            short_window=2,
            long_window=4,
            top_n=1,
            require_rising_long=True,
            long_slope_window=3,
        )
        signals = strategy.agent.signal_data["FAST"]
        first_eligible = signals.index[signals["StrategyEligible"]][0]

        self.assertTrue(signals.loc[first_eligible, "LongSMARising"])
        self.assertGreaterEqual(strategy.data.index.get_loc(first_eligible), 6)

    def test_validates_long_slope_window(self):
        with self.assertRaises(ValueError):
            MovingAverageTopNStrategy(
                make_strategy_data(), long_slope_window=0, auto_generate=False
            )

    def test_optional_downside_shock_filter_excludes_large_recent_loss(self):
        data = make_strategy_data()
        decision_date = data.index[-1]
        previous_date = data.index[-2]
        data.loc[decision_date, ("FAST", "Close")] = (
            data.loc[previous_date, ("FAST", "Close")] * 0.90
        )
        strategy = MovingAverageTopNStrategy(
            data,
            short_window=2,
            long_window=4,
            top_n=1,
            downside_shock_threshold=0.08,
            shock_lookback=5,
        )
        signals = strategy.agent.signal_data["FAST"].loc[decision_date]

        self.assertEqual(signals["Position"], 1)
        self.assertAlmostEqual(signals["MaximumDailyLoss"], 0.10)
        self.assertTrue(signals["DownsideShockExcluded"])
        self.assertFalse(signals["StrategyEligible"])
        self.assertEqual(strategy.target_weights.loc[decision_date, "FAST"], 0.0)

    def test_validates_downside_shock_parameters(self):
        with self.assertRaises(ValueError):
            MovingAverageTopNStrategy(
                make_strategy_data(),
                downside_shock_threshold=0.0,
                auto_generate=False,
            )
        with self.assertRaises(ValueError):
            MovingAverageTopNStrategy(
                make_strategy_data(), shock_lookback=0, auto_generate=False
            )


def data_fields():
    return ["Open", "High", "Low", "Close", "Volume"]


if __name__ == "__main__":
    unittest.main()
