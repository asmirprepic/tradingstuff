import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.technical.graph_shock_propagation_agent import GraphShockPropagationAgent
from strategies import GraphShockPropagationTopNStrategy
from utilities.shock_propagation_utils import (
    estimate_sparse_lag_network,
    lag_network_edge_table,
    residualize_common_effects,
)


def make_causal_returns(periods=340, final_shock=False):
    rng = np.random.default_rng(123)
    index = pd.date_range("2023-01-02", periods=periods, freq="B")
    source = rng.normal(0.0, 0.012, periods)
    target = np.zeros(periods)
    for position in range(1, periods):
        target[position] = (
            0.65 * source[position - 1]
            + 0.10 * target[position - 1]
            + rng.normal(0.0, 0.002)
        )
    returns = pd.DataFrame(
        {
            "SOURCE": source,
            "TARGET": target,
            "NOISE1": rng.normal(0.0, 0.009, periods),
            "NOISE2": rng.normal(0.0, 0.009, periods),
            "NOISE3": rng.normal(0.0, 0.009, periods),
        },
        index=index,
    )
    if final_shock:
        returns.loc[index[-1], "SOURCE"] = 0.08
    return returns


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


class ShockPropagationUtilityTests(unittest.TestCase):
    def test_sector_adjustment_only_uses_reliable_groups(self):
        returns = make_causal_returns(periods=30)
        adjusted, mode = residualize_common_effects(
            returns,
            sector_map={
                "SOURCE": "TECH",
                "TARGET": "TECH",
                "NOISE1": "TECH",
                "NOISE2": "SMALL_GROUP",
            },
            minimum_sector_size=3,
        )

        self.assertEqual(adjusted.shape, returns.shape)
        self.assertEqual(mode["SOURCE"], "MARKET_AND_SECTOR")
        self.assertEqual(mode["NOISE2"], "MARKET")
        self.assertEqual(mode["NOISE3"], "MARKET")

    def test_sparse_network_recovers_incremental_source_to_target_edge(self):
        returns = make_causal_returns()
        coefficients = estimate_sparse_lag_network(
            returns,
            lags=(1, 2, 3),
            alpha=0.00005,
            minimum_observations=200,
        )
        edges = lag_network_edge_table(coefficients)

        self.assertGreater(coefficients[1].loc["SOURCE", "TARGET"], 0.40)
        self.assertTrue(
            (
                edges["Source"].eq("SOURCE")
                & edges["Target"].eq("TARGET")
                & edges["Lag"].eq(1)
            ).any()
        )
        self.assertEqual(coefficients[1].loc["TARGET", "TARGET"], 0.0)


class GraphShockPropagationAgentTests(unittest.TestCase):
    def build_agent(self, data):
        return GraphShockPropagationAgent(
            data,
            estimation_window=160,
            max_lag=3,
            lasso_alpha=0.00005,
            shock_lookback=40,
            shock_z_threshold=2.0,
            minimum_observations=100,
            network_refit_frequency="monthly",
        )

    def test_positive_source_shock_produces_positive_target_forecast(self):
        data = returns_to_ohlcv(make_causal_returns(final_shock=True))
        agent = self.build_agent(data)
        date = data.index[-1]

        self.assertTrue(agent.signal_data["SOURCE"].loc[date, "IsShock"])
        self.assertGreater(
            agent.signal_data["TARGET"].loc[date, "IncomingShockCount"], 0
        )
        self.assertGreater(
            agent.signal_data["TARGET"].loc[date, "PropagationScore"], 0.0
        )
        self.assertGreater(agent.latest_coefficients[1].loc["SOURCE", "TARGET"], 0.0)
        contributions = agent.propagation_contributions("TARGET", date)
        self.assertIn("SOURCE", contributions["Source"].tolist())
        self.assertAlmostEqual(
            contributions["Contribution"].sum(),
            agent.signal_data["TARGET"].loc[date, "PropagationScore"],
        )

    def test_final_shock_does_not_change_prior_forecasts(self):
        returns = make_causal_returns()
        original = self.build_agent(returns_to_ohlcv(returns))
        changed_returns = returns.copy()
        changed_returns.loc[changed_returns.index[-1], "SOURCE"] = 0.08
        revised = self.build_agent(returns_to_ohlcv(changed_returns))

        original_score = original.signal_data["TARGET"]["PropagationScore"]
        revised_score = revised.signal_data["TARGET"]["PropagationScore"]
        pd.testing.assert_series_equal(original_score.iloc[:-1], revised_score.iloc[:-1])

    def test_plotting_interfaces_return_matplotlib_objects(self):
        data = returns_to_ohlcv(make_causal_returns(final_shock=True))
        agent = self.build_agent(data)

        network_figure, network_axis = agent.plot_network(
            max_nodes=5, max_edges=20
        )
        active_figure, active_axis = agent.plot_network(
            active_only=True, max_nodes=5, max_edges=20
        )
        contribution_figure, contribution_axis = (
            agent.plot_propagation_contributions("TARGET")
        )

        self.assertIs(network_axis.figure, network_figure)
        self.assertIs(active_axis.figure, active_figure)
        self.assertIs(contribution_axis.figure, contribution_figure)
        plt.close(network_figure)
        plt.close(active_figure)
        plt.close(contribution_figure)


class GraphShockPropagationStrategyTests(unittest.TestCase):
    def test_builds_distinct_next_bar_top_n_portfolio(self):
        data = returns_to_ohlcv(make_causal_returns(final_shock=True))
        strategy = GraphShockPropagationTopNStrategy(
            data,
            estimation_window=160,
            max_lag=3,
            lasso_alpha=0.00005,
            shock_lookback=40,
            shock_z_threshold=2.0,
            minimum_observations=100,
            network_refit_frequency="monthly",
            top_n=2,
        )
        selected = strategy.selection_log.index[strategy.selection_log["N_Held"].gt(0)]

        self.assertGreater(len(selected), 0)
        first_date = selected[0]
        next_date = strategy.data.index[strategy.data.index.get_loc(first_date) + 1]
        self.assertEqual(strategy.effective_weights.loc[first_date].sum(), 0.0)
        self.assertGreater(strategy.effective_weights.loc[next_date].sum(), 0.0)
        self.assertEqual(strategy.strategy_name, "GraphShockPropagationTopN")


if __name__ == "__main__":
    unittest.main()
