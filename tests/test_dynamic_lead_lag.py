import unittest

import numpy as np
import pandas as pd

from agents.technical.dynamic_lead_lag_agent import DynamicLeadLagAgent
from strategies import DynamicLeadLagTopNStrategy
from utilities.lead_lag_network import (
    estimate_lead_lag_network,
    lead_lag_edge_table,
    screen_lagged_leaders,
)


def make_lead_lag_returns(periods=340):
    rng = np.random.default_rng(44)
    index = pd.date_range("2023-01-02", periods=periods, freq="B")
    leader = rng.normal(0.0002, 0.012, periods)
    target = np.zeros(periods)
    for position in range(1, periods):
        target[position] = (
            0.70 * leader[position - 1]
            + 0.10 * target[position - 1]
            + rng.normal(0.0, 0.002)
        )
    return pd.DataFrame(
        {
            "LEADER": leader,
            "TARGET": target,
            "NOISE1": rng.normal(0.0, 0.010, periods),
            "NOISE2": rng.normal(0.0, 0.010, periods),
            "NOISE3": rng.normal(0.0, 0.010, periods),
            "NOISE4": rng.normal(0.0, 0.010, periods),
        },
        index=index,
    )


def returns_to_ohlcv(returns):
    frames = []
    for stock in returns.columns:
        close = 100.0 * np.exp(returns[stock].cumsum())
        frame = pd.DataFrame(
            {
                (stock, "Open"): close,
                (stock, "High"): close * 1.01,
                (stock, "Low"): close * 0.99,
                (stock, "Close"): close,
                (stock, "Volume"): 1000.0,
            },
            index=returns.index,
        )
        frame.columns = pd.MultiIndex.from_tuples(frame.columns)
        frames.append(frame)
    return pd.concat(frames, axis=1)


class LeadLagNetworkUtilityTests(unittest.TestCase):
    def test_screening_and_sparse_regression_recover_directed_leader(self):
        returns = make_lead_lag_returns()
        candidates, _ = screen_lagged_leaders(
            returns, lags=(1, 2, 3), maximum_leaders=3
        )
        coefficients, _, _ = estimate_lead_lag_network(
            returns,
            lags=(1, 2, 3),
            maximum_leaders=3,
            alpha=0.00005,
            minimum_observations=200,
        )
        edges = lead_lag_edge_table(coefficients)

        self.assertIn("LEADER", candidates["TARGET"])
        self.assertGreater(coefficients[1].loc["LEADER", "TARGET"], 0.40)
        self.assertTrue(
            edges.eval("Source == 'LEADER' and Target == 'TARGET' and Lag == 1").any()
        )


class DynamicLeadLagAgentTests(unittest.TestCase):
    def build_agent(self, data):
        return DynamicLeadLagAgent(
            data,
            estimation_window=160,
            max_lag=3,
            maximum_leaders_per_target=3,
            lasso_alpha=0.00005,
            minimum_observations=100,
            network_refit_frequency="monthly",
            adjust_common_effects=False,
        )

    def test_generates_unconditional_daily_forecasts_and_edges(self):
        agent = self.build_agent(returns_to_ohlcv(make_lead_lag_returns()))
        edges = agent.leaders_for_target("TARGET")
        forecasts = agent.signal_data["TARGET"]["LeadLagForecast"].dropna()

        self.assertFalse(forecasts.empty)
        self.assertTrue(
            (edges["Source"].eq("LEADER") & edges["Lag"].eq(1)).any()
        )
        self.assertLessEqual(edges["Source"].nunique(), 3)

    def test_final_return_change_does_not_change_prior_forecasts(self):
        returns = make_lead_lag_returns()
        original = self.build_agent(returns_to_ohlcv(returns)).score_matrix
        changed = returns.copy()
        changed.loc[changed.index[-1], "LEADER"] += 0.10
        revised = self.build_agent(returns_to_ohlcv(changed)).score_matrix

        pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])


class DynamicLeadLagStrategyTests(unittest.TestCase):
    def test_builds_long_only_next_bar_top_n_portfolio(self):
        strategy = DynamicLeadLagTopNStrategy(
            returns_to_ohlcv(make_lead_lag_returns()),
            estimation_window=160,
            max_lag=3,
            maximum_leaders_per_target=3,
            lasso_alpha=0.00005,
            minimum_observations=100,
            network_refit_frequency="monthly",
            adjust_common_effects=False,
            top_n=2,
        )
        decision_dates = strategy.target_weights.index[
            strategy.target_weights.diff().abs().sum(axis=1).gt(0.0)
        ]
        first_date = decision_dates[0]
        next_date = strategy.data.index[strategy.data.index.get_loc(first_date) + 1]

        self.assertTrue(strategy.target_weights.ge(0.0).all().all())
        self.assertLessEqual(strategy.target_weights.gt(0.0).sum(axis=1).max(), 2)
        self.assertNotEqual(
            strategy.target_weights.loc[first_date].sum(),
            strategy.effective_weights.loc[first_date].sum(),
        )
        self.assertGreaterEqual(strategy.effective_weights.loc[next_date].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
