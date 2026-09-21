import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from agents.data_quality_agent import DataQualityAgent
from scripts.run_data_quality import main
from scripts.run_technical_agents import make_synthetic_ohlcv


class DataQualityTests(unittest.TestCase):
    def test_clean_data_passes(self):
        data = make_synthetic_ohlcv(["AAA"], periods=120)
        row = DataQualityAgent(data, min_history=100).run().iloc[0]

        self.assertEqual(row["Status"], "Pass")
        self.assertEqual(row["QualityScore"], 100)
        self.assertEqual(row["Issues"], "")

    def test_invalid_recent_data_is_rejected_and_issues_are_counted(self):
        data = make_synthetic_ohlcv(["AAA"], periods=120)
        last = data.index[-1]
        data.loc[last, ("AAA", "High")] = data.loc[last, ("AAA", "Low")] - 1
        data.loc[last, ("AAA", "Volume")] = -10
        data.loc[data.index[-2], ("AAA", "Close")] = np.nan

        row = DataQualityAgent(data).run().iloc[0]

        self.assertEqual(row["Status"], "Reject")
        self.assertEqual(row["InvalidOHLC"], 1)
        self.assertEqual(row["NegativeVolume"], 1)
        self.assertEqual(row["RecentMissingValues"], 1)

    def test_stale_and_extreme_prices_warn_without_forcing_rejection(self):
        data = make_synthetic_ohlcv(["AAA"], periods=120)
        stale_dates = data.index[-7:-1]
        data.loc[stale_dates, ("AAA", "Open")] = 100.0
        data.loc[stale_dates, ("AAA", "Close")] = 100.0
        data.loc[stale_dates, ("AAA", "High")] = 101.0
        data.loc[stale_dates, ("AAA", "Low")] = 99.0
        data.loc[data.index[-1], ("AAA", "Open")] = 144.0
        data.loc[data.index[-1], ("AAA", "Close")] = 145.0
        data.loc[data.index[-1], ("AAA", "High")] = 146.0
        data.loc[data.index[-1], ("AAA", "Low")] = 143.0

        row = DataQualityAgent(data, stale_run_limit=5, extreme_return=0.35).run().iloc[0]

        self.assertEqual(row["Status"], "Warning")
        self.assertGreaterEqual(row["LongestStaleCloseRun"], 5)
        self.assertGreaterEqual(row["ExtremeReturns"], 1)

    def test_short_history_is_rejected(self):
        data = make_synthetic_ohlcv(["AAA"], periods=40)
        row = DataQualityAgent(data, min_history=100).run().iloc[0]

        self.assertEqual(row["Status"], "Reject")
        self.assertIn("insufficient_history", row["Issues"])

    def test_runner_outputs_approved_tickers(self):
        paths = [
            Path("outputs/test_quality_report.csv"),
            Path("outputs/test_quality_approved.csv"),
            Path("outputs/test_quality_manifest.json"),
        ]
        try:
            result = main(
                [
                    "--use-synthetic",
                    "--tickers", "AAA,BBB",
                    "--synthetic-periods", "120",
                    "--output", str(paths[0]),
                    "--approved-output", str(paths[1]),
                    "--manifest-output", str(paths[2]),
                ]
            )

            self.assertEqual(result["report"]["Status"].tolist(), ["Pass", "Pass"])
            self.assertEqual(result["approved"]["Stock"].tolist(), ["AAA", "BBB"])
        finally:
            for path in paths:
                if path.exists():
                    path.unlink()

    def test_validates_configuration_and_column_shape(self):
        data = make_synthetic_ohlcv(["AAA"], periods=120)
        with self.assertRaises(ValueError):
            DataQualityAgent(data, stale_run_limit=1)
        with self.assertRaises(ValueError):
            DataQualityAgent(data, extreme_return=0)
        with self.assertRaises(ValueError):
            DataQualityAgent(pd.DataFrame({"Close": [1.0]}))


if __name__ == "__main__":
    unittest.main()
