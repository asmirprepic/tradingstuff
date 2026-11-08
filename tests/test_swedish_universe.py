import unittest

import numpy as np
import pandas as pd

from data_handling.swedish_universe import (
    filter_swedish_ohlcv,
    filter_swedish_tickers,
    swedish_ticker_filter_report,
)


class SwedishUniverseTests(unittest.TestCase):
    def setUp(self):
        self.tickers = [
            "EPI-A.ST",
            "EPI-B.ST",
            "VOLV-A.ST",
            "VOLV-B.ST",
            "ONLY-A.ST",
            "SAGAX-PREF.ST",
            "SAGAX-D.ST",
            "ALVO-SDB.ST",
        ]

    def test_default_keeps_one_common_class_and_excludes_preference(self):
        result = filter_swedish_tickers(self.tickers)

        self.assertEqual(
            result,
            [
                "EPI-B.ST",
                "VOLV-B.ST",
                "ONLY-A.ST",
                "SAGAX-D.ST",
                "ALVO-SDB.ST",
            ],
        )

    def test_all_a_shares_can_be_excluded_explicitly(self):
        result = filter_swedish_tickers(self.tickers, exclude_all_a=True)

        self.assertNotIn("ONLY-A.ST", result)
        self.assertNotIn("EPI-A.ST", result)

    def test_options_can_retain_preference_and_both_common_classes(self):
        result = filter_swedish_tickers(
            self.tickers,
            exclude_preference=False,
            prefer_b_over_a=False,
        )

        self.assertEqual(result, self.tickers)

    def test_report_explains_exclusions(self):
        report = swedish_ticker_filter_report(self.tickers).set_index("Ticker")

        self.assertEqual(
            report.loc["EPI-A.ST", "Reason"],
            "corresponding_b_share_available",
        )
        self.assertEqual(
            report.loc["SAGAX-PREF.ST", "Reason"],
            "preference_share",
        )
        self.assertTrue(report.loc["ONLY-A.ST", "Included"])

    def test_ohlcv_filter_preserves_multiindex_structure(self):
        columns = pd.MultiIndex.from_product(
            [self.tickers, ["Close", "Volume"]],
            names=["Stock", "Field"],
        )
        data = pd.DataFrame(
            np.arange(3 * len(columns)).reshape(3, len(columns)),
            index=pd.date_range("2026-01-01", periods=3),
            columns=columns,
        )

        result = filter_swedish_ohlcv(data)

        self.assertIsInstance(result.columns, pd.MultiIndex)
        self.assertEqual(
            result.columns.get_level_values("Stock").unique().tolist(),
            ["EPI-B.ST", "VOLV-B.ST", "ONLY-A.ST", "SAGAX-D.ST", "ALVO-SDB.ST"],
        )
        self.assertEqual(result.columns.names, ["Stock", "Field"])

    def test_invalid_flat_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            filter_swedish_ohlcv(pd.DataFrame({"Close": [1.0]}))


if __name__ == "__main__":
    unittest.main()
