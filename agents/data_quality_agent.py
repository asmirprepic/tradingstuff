import numpy as np
import pandas as pd


class DataQualityAgent:
    """Audit per-ticker OHLCV histories before strategy execution."""

    REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

    def __init__(
        self,
        data,
        min_history=100,
        recent_window=20,
        stale_run_limit=5,
        extreme_return=0.35,
    ):
        if min_history < 1 or recent_window < 1 or stale_run_limit < 2:
            raise ValueError("history/window values must be positive and stale_run_limit must be at least 2.")
        if extreme_return <= 0:
            raise ValueError("extreme_return must be positive.")
        if not isinstance(data.columns, pd.MultiIndex) or data.columns.nlevels < 2:
            raise ValueError("data must have MultiIndex columns in (ticker, field) form.")

        self.data = data
        self.min_history = int(min_history)
        self.recent_window = int(recent_window)
        self.stale_run_limit = int(stale_run_limit)
        self.extreme_return = float(extreme_return)

    @staticmethod
    def _longest_unchanged_run(series):
        values = series.dropna()
        if values.empty:
            return 0
        groups = values.ne(values.shift()).cumsum()
        return int(values.groupby(groups).size().max())

    def inspect_stock(self, stock):
        available = set(self.data[stock].columns)
        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in available]
        if missing_columns:
            return {
                "Stock": stock,
                "Status": "Reject",
                "QualityScore": 0,
                "Rows": len(self.data),
                "Issues": f"missing_columns:{','.join(missing_columns)}",
            }

        bars = self.data[stock].loc[:, self.REQUIRED_COLUMNS].copy()
        duplicate_timestamps = int(bars.index.duplicated(keep=False).sum())
        bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
        rows = len(bars)
        missing_values = int(bars.isna().sum().sum())
        recent = bars.tail(self.recent_window)
        recent_missing_values = int(recent.isna().sum().sum())

        prices = bars[["Open", "High", "Low", "Close"]]
        nonpositive_prices = int((prices <= 0).any(axis=1).sum())
        invalid_ohlc = int(
            (
                (bars["High"] < prices.max(axis=1))
                | (bars["Low"] > prices.min(axis=1))
                | (bars["High"] < bars["Low"])
            ).sum()
        )
        negative_volume = int((bars["Volume"] < 0).sum())
        zero_volume_recent = int((recent["Volume"] == 0).sum())
        longest_stale_run = self._longest_unchanged_run(bars["Close"])
        returns = bars["Close"].pct_change(fill_method=None)
        extreme_returns = int((returns.abs() > self.extreme_return).sum())

        issues = []
        hard_failure = False
        score = 100
        if rows < self.min_history:
            issues.append(f"insufficient_history:{rows}<{self.min_history}")
            score -= 50
            hard_failure = True
        if duplicate_timestamps:
            issues.append(f"duplicate_timestamps:{duplicate_timestamps}")
            score -= min(30, duplicate_timestamps * 5)
        if missing_values:
            issues.append(f"missing_values:{missing_values}")
            score -= min(30, missing_values)
        if recent_missing_values:
            issues.append(f"recent_missing_values:{recent_missing_values}")
            score -= 20
            hard_failure = True
        if nonpositive_prices:
            issues.append(f"nonpositive_prices:{nonpositive_prices}")
            score -= 40
            hard_failure = True
        if invalid_ohlc:
            issues.append(f"invalid_ohlc:{invalid_ohlc}")
            score -= 40
            hard_failure = True
        if negative_volume:
            issues.append(f"negative_volume:{negative_volume}")
            score -= 30
            hard_failure = True
        if zero_volume_recent:
            issues.append(f"recent_zero_volume:{zero_volume_recent}")
            score -= min(20, zero_volume_recent * 2)
        if longest_stale_run >= self.stale_run_limit:
            issues.append(f"stale_close_run:{longest_stale_run}")
            score -= min(25, longest_stale_run)
        if extreme_returns:
            issues.append(f"extreme_returns:{extreme_returns}")
            score -= min(20, extreme_returns * 2)

        score = max(0, int(score))
        status = "Reject" if hard_failure or score < 50 else ("Warning" if issues else "Pass")
        return {
            "Stock": stock,
            "Status": status,
            "QualityScore": score,
            "Rows": rows,
            "Start": bars.index.min() if rows else pd.NaT,
            "End": bars.index.max() if rows else pd.NaT,
            "DuplicateTimestamps": duplicate_timestamps,
            "MissingValues": missing_values,
            "RecentMissingValues": recent_missing_values,
            "InvalidOHLC": invalid_ohlc,
            "NonpositivePrices": nonpositive_prices,
            "NegativeVolume": negative_volume,
            "RecentZeroVolume": zero_volume_recent,
            "LongestStaleCloseRun": longest_stale_run,
            "ExtremeReturns": extreme_returns,
            "Issues": "; ".join(issues),
        }

    def run(self):
        stocks = self.data.columns.get_level_values(0).unique()
        rows = [self.inspect_stock(stock) for stock in stocks]
        status_order = {"Reject": 0, "Warning": 1, "Pass": 2}
        report = pd.DataFrame(rows)
        report["_status_order"] = report["Status"].map(status_order)
        return report.sort_values(
            ["_status_order", "QualityScore", "Stock"],
            ascending=[True, True, True],
        ).drop(columns="_status_order").reset_index(drop=True)
