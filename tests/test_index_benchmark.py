import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmarks import IndexBenchmark
from strategies import MovingAverageTopNStrategy


def make_prices(ticker, daily_return, periods=12, timezone=None):
    index = pd.date_range("2024-01-01", periods=periods, freq="B", tz=timezone)
    return pd.Series(
        100.0 * np.power(1.0 + daily_return, np.arange(periods)),
        index=index,
        name=ticker,
    )


def make_stock_data(periods=12):
    prices = {
        "FAST": make_prices("FAST", 0.02, periods, "Europe/Stockholm"),
        "SLOW": make_prices("SLOW", 0.01, periods, "Europe/Stockholm"),
    }
    data = pd.DataFrame(
        {(ticker, "Close"): values.to_numpy() for ticker, values in prices.items()},
        index=prices["FAST"].index,
    )
    data.columns = pd.MultiIndex.from_tuples(data.columns)
    return data


class IndexBenchmarkTests(unittest.TestCase):
    def test_returns_and_equity_are_derived_from_index_prices(self):
        prices = make_prices("^OMX", 0.01)
        benchmark = IndexBenchmark(prices=prices)

        returns = benchmark.return_series()
        equity = benchmark.equity_curve()

        self.assertEqual(returns.iloc[0], 0.0)
        self.assertTrue(np.allclose(returns.iloc[1:], 0.01))
        self.assertAlmostEqual(equity.iloc[-1], prices.iloc[-1] / prices.iloc[0])

    def test_evaluation_bounds_apply_to_index_returns(self):
        prices = make_prices("^OMX", 0.01)
        benchmark = IndexBenchmark(prices=prices)

        returns = benchmark.return_series(prices.index[2], prices.index[6])

        self.assertEqual(returns.index[0], prices.index[2])
        self.assertEqual(returns.index[-1], prices.index[6])
        self.assertEqual(len(returns), 5)

    @patch("benchmarks.index_benchmark.GetStockDataTest")
    def test_fetch_uses_omx_ticker_and_inclusive_requested_end(self, loader_class):
        index_prices = make_prices("^OMX", 0.01)
        fetched = pd.DataFrame({("^OMX", "Close"): index_prices})
        fetched.columns = pd.MultiIndex.from_tuples(fetched.columns)
        loader_class.return_value.getData.return_value = fetched
        benchmark = IndexBenchmark()

        prices = benchmark.fetch("2024-01-01", "2024-01-10")

        self.assertFalse(prices.empty)
        kwargs = loader_class.call_args.kwargs
        self.assertEqual(kwargs["stocks"], ["^OMX"])
        self.assertEqual(str(kwargs["enddate"]), "2024-01-11")

    def test_strategy_uses_index_as_its_performance_benchmark(self):
        data = make_stock_data()
        index_prices = make_prices("^OMX", 0.005)
        benchmark = IndexBenchmark(prices=index_prices)
        strategy = MovingAverageTopNStrategy(
            data,
            short_window=2,
            long_window=4,
            top_n=1,
            evaluation_start=data.index[6],
            performance_benchmark=benchmark,
        )

        result = strategy.evaluate_performance()
        expected = benchmark.evaluate_performance(
            strategy.evaluation_start,
            data.index[-1],
        )

        self.assertEqual(result.loc[1, "Portfolio"], "OMX Stockholm 30")
        self.assertAlmostEqual(
            result.loc[1, "Total_Return_%"],
            expected.loc[0, "Total_Return_%"],
            places=4,
        )

    def test_strategy_rejects_an_empty_index_benchmark(self):
        data = make_stock_data()

        with self.assertRaisesRegex(ValueError, "non-empty price Series"):
            MovingAverageTopNStrategy(
                data,
                short_window=2,
                long_window=4,
                top_n=1,
                evaluation_start=data.index[6],
                performance_benchmark=IndexBenchmark(),
            )


if __name__ == "__main__":
    unittest.main()
