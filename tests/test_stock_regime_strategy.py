import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from strategies import StockRegimeTopNStrategy


class StockRegimeStrategyTests(unittest.TestCase):
    def data(self):
        index = pd.bdate_range("2020-01-01", periods=65)
        return pd.DataFrame({(s, f): 100*np.exp(np.arange(65)*.001)
                             for s in ['A', 'B', 'C'] for f in ['Open', 'Close']},
                            index=index)

    def make(self, data, scores=(.03, .01, -.01), **kwargs):
        with patch('strategies.stock_regime_top_n_strategy.StockRegimeAgent') as cls:
            cls.return_value.signal_data = {
                s: pd.DataFrame({'ExpectedReturn': score, 'Valid': np.isfinite(score)},
                                index=data.index)
                for s, score in zip(['A', 'B', 'C'], scores)}
            strategy = StockRegimeTopNStrategy(data, estimation_window=40, top_n=2, **kwargs)
            self.assertEqual(cls.call_args.kwargs['prediction_horizon'],
                             kwargs.get('prediction_horizon', 1))
            return strategy

    def test_slots_cash_and_execution_lag(self):
        data = self.data()
        strategy = self.make(data, minimum_forecast=.02)
        self.assertEqual(strategy.target_weights.iloc[42]['A'], .5)
        self.assertEqual(strategy.target_weights.iloc[42].sum(), .5)
        self.assertEqual(strategy.effective_weights.iloc[42].sum(), 0)
        self.assertEqual(strategy.intraday_weights.iloc[43]['A'], .5)
        self.assertFalse(strategy.evaluate_performance().empty)

    def test_invalid_negative_scores_leave_cash(self):
        strategy = self.make(self.data(), scores=(np.nan, -.01, 0))
        self.assertEqual(strategy.target_weights.to_numpy().sum(), 0)
        self.assertTrue(strategy.portfolio_returns.eq(0).all())

    def test_prefix_and_weekly_schedule(self):
        data = self.data()
        a = self.make(data, rebalance_frequency='W-FRI', prediction_horizon=5)
        b = self.make(data.iloc[:60], rebalance_frequency='W-FRI', prediction_horizon=5)
        pd.testing.assert_frame_equal(a.target_weights.iloc[:60], b.target_weights)
        self.assertTrue(all(d.weekday() == 4 for d in a.weight_rebalance_dates))

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            self.make(self.data(), minimum_forecast=-1)
