import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.technical.graph_relative_return_agent import GraphRelativeReturnAgent
from strategies import GraphDeviationTopNStrategy
from utilities.graph_utils import (
    build_correlation_graph,
    graph_deviation,
    neighbour_weighted_values,
    random_walk_laplacian,
)


def make_graph_data(periods=180, stocks=8):
    rng = np.random.default_rng(42)
    index = pd.date_range("2023-01-02", periods=periods, freq="B")
    market = rng.normal(0.0003, 0.007, periods)
    sectors = [rng.normal(0.0, 0.006, periods), rng.normal(0.0, 0.006, periods)]
    frames = []
    for number in range(stocks):
        group = number % 2
        innovation = rng.normal(0.0, 0.003, periods)
        persistent = np.zeros(periods)
        for position in range(1, periods):
            persistent[position] = 0.65 * persistent[position - 1] + innovation[position]
        log_returns = market + 0.8 * sectors[group] + persistent
        close = 100.0 * np.exp(np.cumsum(log_returns))
        ticker = f"S{number}"
        frame = pd.DataFrame(
            {
                (ticker, "Open"): close,
                (ticker, "High"): close * 1.01,
                (ticker, "Low"): close * 0.99,
                (ticker, "Close"): close,
                (ticker, "Volume"): 1000.0,
            },
            index=index,
        )
        frame.columns = pd.MultiIndex.from_tuples(frame.columns)
        frames.append(frame)
    return pd.concat(frames, axis=1)


class GraphUtilityTests(unittest.TestCase):
    def test_builds_symmetric_sparse_positive_graph(self):
        rng = np.random.default_rng(1)
        factor = rng.normal(size=80)
        returns = pd.DataFrame({
            "A": factor + rng.normal(scale=0.1, size=80),
            "B": factor + rng.normal(scale=0.1, size=80),
            "C": -factor + rng.normal(scale=0.1, size=80),
        })

        adjacency = build_correlation_graph(
            returns,
            top_k=1,
            minimum_correlation=0.2,
            remove_market_mode=False,
        )

        pd.testing.assert_frame_equal(adjacency, adjacency.T)
        self.assertTrue((adjacency.to_numpy() >= 0.0).all())
        self.assertTrue(np.allclose(np.diag(adjacency), 0.0))
        self.assertGreater(adjacency.loc["A", "B"], 0.0)
        self.assertEqual(adjacency.loc["A", "C"], 0.0)

    def test_laplacian_deviation_matches_neighbour_difference(self):
        adjacency = pd.DataFrame(
            [[0.0, 2.0, 1.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        values = pd.Series({"A": 0.06, "B": 0.03, "C": 0.00})

        neighbours = neighbour_weighted_values(adjacency, values)
        deviation = graph_deviation(adjacency, values)
        matrix_deviation = random_walk_laplacian(adjacency).dot(values)

        self.assertAlmostEqual(neighbours["A"], 0.02)
        pd.testing.assert_series_equal(deviation, matrix_deviation.rename("GraphDeviation"))


class GraphRelativeReturnAgentTests(unittest.TestCase):
    def build_agent(self, data):
        return GraphRelativeReturnAgent(
            data,
            graph_window=30,
            top_k=2,
            minimum_correlation=0.05,
            recent_return_window=5,
            prediction_horizon=5,
            calibration_weeks=8,
            minimum_calibration_weeks=3,
            direction_shrinkage=0.1,
        )

    def test_builds_graph_and_learns_a_direction(self):
        agent = self.build_agent(make_graph_data())

        self.assertFalse(agent.latest_adjacency.empty)
        self.assertFalse(agent.graph_history.empty)
        self.assertIsNotNone(agent.first_score_position)
        self.assertTrue(agent.direction_history["DirectionBeta"].notna().any())
        self.assertTrue(agent.signal_data["S0"]["GraphScore"].notna().any())

    def test_final_price_change_does_not_change_prior_graph_scores(self):
        data = make_graph_data()
        original = self.build_agent(data).signal_data["S0"]["GraphScore"]
        changed = data.copy()
        changed.loc[changed.index[-1], ("S0", "Close")] *= 1.5
        revised = self.build_agent(changed).signal_data["S0"]["GraphScore"]

        pd.testing.assert_series_equal(original.iloc[:-1], revised.iloc[:-1])

    def test_incomplete_final_week_is_not_a_decision_date(self):
        data = make_graph_data(periods=181)
        agent = GraphRelativeReturnAgent(
            data,
            graph_window=30,
            top_k=2,
            minimum_correlation=0.05,
            recent_return_window=5,
            prediction_horizon=5,
            calibration_weeks=8,
            minimum_calibration_weeks=3,
            auto_generate=False,
        )

        self.assertNotEqual(data.index[-1].weekday(), 4)
        self.assertNotIn(data.index[-1], agent._decision_dates())
        self.assertEqual(agent._decision_dates()[-1].weekday(), 4)


class GraphDeviationTopNStrategyTests(unittest.TestCase):
    def test_creates_weekly_next_bar_portfolio(self):
        strategy = GraphDeviationTopNStrategy(
            make_graph_data(),
            graph_window=30,
            top_k=2,
            minimum_correlation=0.05,
            recent_return_window=5,
            prediction_horizon=5,
            calibration_weeks=8,
            minimum_calibration_weeks=3,
            top_n=2,
        )
        selected_dates = strategy.selection_log.index[
            strategy.selection_log["N_Held"].gt(0)
        ]

        self.assertGreater(len(selected_dates), 0)
        first_date = selected_dates[0]
        next_date = strategy.data.index[strategy.data.index.get_loc(first_date) + 1]
        self.assertEqual(strategy.effective_weights.loc[first_date].sum(), 0.0)
        self.assertEqual(strategy.effective_weights.loc[next_date].sum(), 1.0)
        self.assertEqual(strategy.selection_log.loc[first_date, "N_Held"], 2)

    def test_rejects_too_many_neighbours(self):
        with self.assertRaises(ValueError):
            GraphDeviationTopNStrategy(
                make_graph_data(stocks=3),
                graph_window=30,
                top_k=3,
            )

    def test_plotting_interfaces_return_matplotlib_objects(self):
        strategy = GraphDeviationTopNStrategy(
            make_graph_data(),
            graph_window=30,
            top_k=2,
            minimum_correlation=0.05,
            recent_return_window=5,
            prediction_horizon=5,
            calibration_weeks=8,
            minimum_calibration_weeks=3,
            top_n=2,
        )

        graph_figure, graph_axis = strategy.plot_graph(max_nodes=6, max_edges=20)
        direction_figure, direction_axis = strategy.plot_direction_history()
        stock_figure, stock_axes = strategy.plot_stock_signal("S0")

        self.assertGreater(len(strategy.agent.graph_snapshots), 0)
        self.assertIs(graph_axis.figure, graph_figure)
        self.assertIs(direction_axis.figure, direction_figure)
        self.assertIs(stock_axes[0].figure, stock_figure)
        plt.close(graph_figure)
        plt.close(direction_figure)
        plt.close(stock_figure)


if __name__ == "__main__":
    unittest.main()
