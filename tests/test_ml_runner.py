import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.run_ml_agents import (
    AGENT_GROUPS,
    build_agent,
    parse_agent_names,
    safe_component,
    tickers_from_shortlist,
    write_manifest,
)
from scripts.run_technical_agents import make_synthetic_ohlcv


class MLRunnerTests(unittest.TestCase):
    def test_parse_agent_names_expands_groups_without_duplicates(self):
        selected = parse_agent_names("lightweight,gaussian_process,logistic_reg")

        self.assertEqual(selected[: len(AGENT_GROUPS["lightweight"])], AGENT_GROUPS["lightweight"])
        self.assertEqual(selected.count("logistic_reg"), 1)
        self.assertEqual(selected[-1], "gaussian_process")

        with self.assertRaises(SystemExit):
            parse_agent_names("not_a_model")

    def test_tickers_from_shortlist_filters_tiers_and_deduplicates(self):
        shortlist = pd.DataFrame(
            [
                {"Stock": "AAPL", "ShortlistTier": "TierA"},
                {"Stock": "MSFT", "ShortlistTier": "Watch"},
                {"Stock": "AAPL", "ShortlistTier": "TierB"},
            ]
        )
        with patch("scripts.run_ml_agents.Path.exists", return_value=True), patch(
            "scripts.run_ml_agents.pd.read_csv",
            return_value=shortlist,
        ):
            tickers = tickers_from_shortlist("shortlist.csv", ["TierA", "TierB"])

        self.assertEqual(tickers, ["AAPL"])

    def test_build_agent_supports_gaussian_process_and_safe_artifact_names(self):
        data = make_synthetic_ohlcv(["BRK/B"], periods=40)
        args = Namespace(proba_threshold=0.6, gaussian_max_samples=30, epochs=1, verbose=0)

        agent = build_agent("gaussian_process", data, args)

        self.assertEqual(agent.algorithm_name, "GaussianProcess")
        self.assertEqual(agent.max_samples, 30)
        self.assertEqual(safe_component("BRK/B"), "BRK_B")

    def test_build_agent_supports_quantile_regression(self):
        data = make_synthetic_ohlcv(["AAA"], periods=100)
        agent = build_agent("quantile_regression", data, Namespace())

        self.assertEqual(agent.algorithm_name, "QuantileRegression")
        self.assertIn("quantile_regression", parse_agent_names("classical"))

    def test_build_agent_supports_perceptron(self):
        data = make_synthetic_ohlcv(["AAA"], periods=100)
        agent = build_agent("perceptron", data, Namespace())

        self.assertEqual(agent.algorithm_name, "Perceptron")
        self.assertEqual(agent.score_column, "DecisionScore")
        self.assertIn("perceptron", parse_agent_names("classical"))

    def test_build_agent_supports_qda(self):
        data = make_synthetic_ohlcv(["AAA"], periods=100)
        agent = build_agent("qda", data, Namespace(proba_threshold=0.65))

        self.assertEqual(agent.algorithm_name, "QDA")
        self.assertEqual(agent.proba_threshold, 0.65)
        self.assertIn("qda", parse_agent_names("classical"))

    def test_build_agent_supports_spline_logistic(self):
        data = make_synthetic_ohlcv(["AAA"], periods=100)
        agent = build_agent("spline_logistic", data, Namespace(proba_threshold=0.65))

        self.assertEqual(agent.algorithm_name, "SplineLogistic")
        self.assertEqual(agent.proba_threshold, 0.65)
        self.assertIn("spline_logistic", parse_agent_names("classical"))

    def test_build_agent_supports_one_class_svm(self):
        data = make_synthetic_ohlcv(["AAA"], periods=100)
        agent = build_agent("one_class_svm", data, Namespace())

        self.assertEqual(agent.algorithm_name, "OneClassSVM")
        self.assertIn("one_class_svm", parse_agent_names("specialized"))

    def test_manifest_paths_are_relative_and_portable(self):
        args = Namespace(
            use_synthetic=True, technical_shortlist=None, shortlist_tiers=[], start=None,
            end=None, lookback_days=20, interval="1d", mode="backtest", persistence=1,
            proba_threshold=0.55, epochs=1, reuse_artifacts=False, save_artifacts=False,
            artifact_dir="outputs/models",
        )
        manifest_path = Path("outputs/test_ml_manifest.json")
        try:
            write_manifest(
                manifest_path, "test", args, ["AAA"], ["logistic_reg"],
                {"summary": Path("outputs/test_summary.csv").resolve()}, {}, [],
            )
            manifest_text = manifest_path.read_text(encoding="utf-8")

            self.assertIn('"summary": "test_summary.csv"', manifest_text)
            self.assertIn('"artifact_dir": "models"', manifest_text)
            self.assertNotIn(str(Path.home()), manifest_text)
        finally:
            if manifest_path.exists():
                manifest_path.unlink()


if __name__ == "__main__":
    unittest.main()
