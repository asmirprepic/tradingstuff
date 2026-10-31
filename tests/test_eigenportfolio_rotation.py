import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.technical.eigenportfolio_rotation_agent import EigenportfolioRotationAgent
from strategies import EigenportfolioRotationStrategy
from utilities.eigenportfolio_utils import (
    align_eigenvectors,
    cap_and_normalize_long_weights,
    estimate_shrunk_eigenportfolios,
    long_only_component_sleeves,
)


def make_factor_data(periods=360, stocks=12):
    rng = np.random.default_rng(91)
    index = pd.date_range("2023-01-02", periods=periods, freq="B")
    market = rng.normal(0.0012, 0.006, periods)
    factors = [
        rng.normal(0.0004, 0.007, periods),
        rng.normal(0.0002, 0.006, periods),
        rng.normal(0.0003, 0.005, periods),
    ]
    frames = []
    for number in range(stocks):
        group = number % len(factors)
        returns = (
            market
            + 0.75 * factors[group]
            + rng.normal(0.0, 0.0025, periods)
        )
        close = 100.0 * np.exp(np.cumsum(returns))
        stock = f"S{number}"
        frame = pd.DataFrame(
            {
                (stock, "Open"): close,
                (stock, "High"): close * 1.01,
                (stock, "Low"): close * 0.99,
                (stock, "Close"): close,
                (stock, "Volume"): 1000.0,
            },
            index=index,
        )
        frame.columns = pd.MultiIndex.from_tuples(frame.columns)
        frames.append(frame)
    return pd.concat(frames, axis=1)


class EigenportfolioUtilityTests(unittest.TestCase):
    def test_aligns_component_permutation_and_sign(self):
        previous = pd.DataFrame(
            {"PC1": [1.0, 0.0, 0.0], "PC2": [0.0, 1.0, 0.0]},
            index=["A", "B", "C"],
        )
        current = pd.DataFrame(
            {"X": [0.0, -1.0, 0.0], "Y": [-1.0, 0.0, 0.0]},
            index=previous.index,
        )

        aligned, order = align_eigenvectors(current, previous)

        self.assertEqual(order, [1, 0])
        self.assertGreater(aligned.loc["A", "PC1"], 0.0)
        self.assertGreater(aligned.loc["B", "PC2"], 0.0)

    def test_builds_long_only_volatility_adjusted_sleeves(self):
        data = make_factor_data(periods=180, stocks=8)
        close = data.xs("Close", level=1, axis=1)
        estimate = estimate_shrunk_eigenportfolios(
            close.pct_change(fill_method=None),
            component_count=4,
        )
        sleeves = long_only_component_sleeves(estimate["tradable_loadings"])

        self.assertTrue(sleeves.ge(0.0).all().all())
        self.assertTrue(np.allclose(sleeves.sum(axis=0), 1.0))
        self.assertIn("PC1_POS", sleeves.columns)
        self.assertTrue(any(column.endswith("_NEG") for column in sleeves.columns))

    def test_caps_and_normalizes_long_weights(self):
        weights = pd.Series({"A": 0.80, "B": 0.10, "C": 0.05, "D": 0.05})
        capped = cap_and_normalize_long_weights(weights, maximum_weight=0.30)

        self.assertLessEqual(capped.max(), 0.30)
        self.assertAlmostEqual(capped.sum(), 1.0)
        self.assertTrue(capped.ge(0.0).all())


class EigenportfolioRotationAgentTests(unittest.TestCase):
    def build_agent(self, data, include_first_component=True):
        return EigenportfolioRotationAgent(
            data,
            estimation_window=120,
            component_count=5,
            include_first_component=include_first_component,
            short_score_window=10,
            long_score_window=30,
            selected_sleeves=2,
            component_refit_frequency="monthly",
            decision_frequency="W-FRI",
            maximum_stock_weight=0.20,
        )

    def test_generates_long_only_capped_target_weights(self):
        agent = self.build_agent(make_factor_data())
        targets = agent.target_weight_matrix.dropna(how="all")

        self.assertFalse(targets.empty)
        self.assertTrue(targets.fillna(0.0).ge(0.0).all().all())
        self.assertLessEqual(targets.max().max(), 0.20 + 1e-12)
        self.assertTrue(targets.sum(axis=1).le(1.0 + 1e-12).all())
        self.assertFalse(agent.component_history.empty)
        self.assertFalse(agent.sleeve_returns.empty)

    def test_can_exclude_first_component_from_rotation(self):
        agent = self.build_agent(make_factor_data(), include_first_component=False)

        self.assertTrue(
            all(
                not sleeve.startswith("PC1_")
                for sleeve in agent.sleeve_score_history.columns
            )
        )

    def test_selects_only_one_side_of_each_component(self):
        agent = self.build_agent(make_factor_data())

        for selected in agent.selected_sleeves_history["SelectedSleeves"]:
            components = [sleeve.rsplit("_", 1)[0] for sleeve in selected]
            self.assertEqual(len(components), len(set(components)))

    def test_final_price_change_does_not_change_prior_targets(self):
        data = make_factor_data()
        original = self.build_agent(data).target_weight_matrix
        changed = data.copy()
        changed.loc[changed.index[-1], ("S0", "Close")] *= 1.5
        revised = self.build_agent(changed).target_weight_matrix

        pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])

    def test_plotting_and_component_diagnostics(self):
        agent = self.build_agent(make_factor_data())

        summary = agent.component_summary()
        weights = agent.get_component_weights()
        variance_figure, variance_axis = agent.plot_explained_variance()
        weights_figure, weights_axis = agent.plot_target_weights()

        self.assertFalse(summary.empty)
        self.assertFalse(weights.empty)
        self.assertIs(variance_axis.figure, variance_figure)
        self.assertIs(weights_axis.figure, weights_figure)
        plt.close(variance_figure)
        plt.close(weights_figure)


class EigenportfolioRotationStrategyTests(unittest.TestCase):
    def test_applies_weighted_targets_on_next_bar(self):
        strategy = EigenportfolioRotationStrategy(
            make_factor_data(),
            estimation_window=120,
            component_count=5,
            include_first_component=True,
            short_score_window=10,
            long_score_window=30,
            selected_sleeves=2,
            maximum_stock_weight=0.20,
        )
        target_dates = strategy.agent.target_weight_matrix.index[
            strategy.agent.target_weight_matrix.notna().any(axis=1)
        ]
        first_date = target_dates[0]
        next_date = strategy.data.index[strategy.data.index.get_loc(first_date) + 1]

        self.assertEqual(strategy.effective_weights.loc[first_date].sum(), 0.0)
        self.assertAlmostEqual(
            strategy.effective_weights.loc[next_date].sum(),
            strategy.target_weights.loc[first_date].sum(),
        )
        self.assertTrue(strategy.target_weights.ge(0.0).all().all())
        self.assertLessEqual(strategy.target_weights.max().max(), 0.20 + 1e-12)

    def test_weights_drift_between_weekly_rebalances(self):
        strategy = EigenportfolioRotationStrategy(
            make_factor_data(),
            estimation_window=120,
            component_count=5,
            include_first_component=True,
            short_score_window=10,
            long_score_window=30,
            selected_sleeves=2,
            maximum_stock_weight=0.20,
        )
        invested = strategy.effective_weights.sum(axis=1).gt(0.0)
        non_execution = strategy.selection_log["Turnover"].eq(0.0)
        candidates = strategy.data.index[invested & non_execution]
        date = candidates[5]
        previous_date = strategy.data.index[strategy.data.index.get_loc(date) - 1]

        self.assertFalse(
            np.allclose(
                strategy.effective_weights.loc[date],
                strategy.target_weights.loc[previous_date],
            )
        )


if __name__ == "__main__":
    unittest.main()
