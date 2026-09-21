import sys
import unittest

from scripts.trading_gui import build_cli_args


class TradingGUITests(unittest.TestCase):
    def test_builds_ml_command(self):
        command = build_cli_args(
            "ml",
            tickers="AAPL,MSFT",
            agents="qda,spline_logistic",
            use_synthetic=True,
        )

        self.assertEqual(command[:4], [sys.executable, "-m", "scripts.trading_cli", "ml"])
        self.assertIn("--use-synthetic", command)
        self.assertEqual(command[command.index("--agents") + 1], "qda,spline_logistic")

    def test_ticker_file_takes_precedence_and_extra_options_are_split(self):
        command = build_cli_args(
            "technical",
            tickers="AAPL",
            tickers_file="approved tickers.csv",
            lookback_days=300,
            extra_args='--top-n-per-agent 5 --summary-output "my summary.csv"',
        )

        self.assertNotIn("--tickers", command)
        self.assertEqual(command[command.index("--tickers-file") + 1], "approved tickers.csv")
        self.assertEqual(command[command.index("--lookback-days") + 1], "300")
        self.assertEqual(command[command.index("--summary-output") + 1], "my summary.csv")

    def test_dashboard_uses_latest_manifest_by_default(self):
        command = build_cli_args("dashboard")

        self.assertEqual(command[-2:], ["--latest-manifest-dir", "."])

    def test_rejects_unknown_command(self):
        with self.assertRaises(ValueError):
            build_cli_args("unknown")


if __name__ == "__main__":
    unittest.main()
