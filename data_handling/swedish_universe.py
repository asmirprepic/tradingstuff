import re
from collections.abc import Iterable

import pandas as pd


_SHARE_CLASS_PATTERN = re.compile(
    r"^(?P<issuer>.+)-(?P<share_class>A|B|PREF)\.ST$",
    flags=re.IGNORECASE,
)


def _unique_tickers(tickers: Iterable[str]) -> list[str]:
    if isinstance(tickers, (str, bytes)):
        raise TypeError("tickers must be an iterable of ticker strings, not one string.")

    unique = []
    seen = set()
    for ticker in tickers:
        if not isinstance(ticker, str):
            raise TypeError("Every ticker must be a string.")
        ticker = ticker.strip()
        if ticker and ticker.upper() not in seen:
            unique.append(ticker)
            seen.add(ticker.upper())
    return unique


def _filter_decisions(
    tickers: Iterable[str],
    exclude_preference: bool,
    prefer_b_over_a: bool,
    exclude_all_a: bool,
) -> list[dict]:
    unique = _unique_tickers(tickers)
    available = {ticker.upper() for ticker in unique}
    decisions = []

    for ticker in unique:
        match = _SHARE_CLASS_PATTERN.match(ticker)
        share_class = match.group("share_class").upper() if match else None
        include = True
        reason = "included"

        if share_class == "PREF" and exclude_preference:
            include = False
            reason = "preference_share"
        elif share_class == "A" and exclude_all_a:
            include = False
            reason = "a_share_excluded"
        elif share_class == "A" and prefer_b_over_a:
            b_ticker = f"{match.group('issuer')}-B.ST".upper()
            if b_ticker in available:
                include = False
                reason = "corresponding_b_share_available"

        decisions.append(
            {
                "Ticker": ticker,
                "Included": include,
                "Reason": reason,
            }
        )

    return decisions


def filter_swedish_tickers(
    tickers: Iterable[str],
    *,
    exclude_preference: bool = True,
    prefer_b_over_a: bool = True,
    exclude_all_a: bool = False,
) -> list[str]:
    """Filter Swedish share classes while preserving input order.

    By default, preference shares are removed and an A share is removed only
    when the corresponding B share is present. This avoids dropping issuers
    whose only available common share happens to be an A share.
    """
    decisions = _filter_decisions(
        tickers,
        exclude_preference=exclude_preference,
        prefer_b_over_a=prefer_b_over_a,
        exclude_all_a=exclude_all_a,
    )
    return [row["Ticker"] for row in decisions if row["Included"]]


def swedish_ticker_filter_report(
    tickers: Iterable[str],
    *,
    exclude_preference: bool = True,
    prefer_b_over_a: bool = True,
    exclude_all_a: bool = False,
) -> pd.DataFrame:
    """Return an audit table for the Swedish-universe filtering decision."""
    decisions = _filter_decisions(
        tickers,
        exclude_preference=exclude_preference,
        prefer_b_over_a=prefer_b_over_a,
        exclude_all_a=exclude_all_a,
    )
    return pd.DataFrame(decisions, columns=["Ticker", "Included", "Reason"])


def filter_swedish_ohlcv(
    data: pd.DataFrame,
    *,
    exclude_preference: bool = True,
    prefer_b_over_a: bool = True,
    exclude_all_a: bool = False,
) -> pd.DataFrame:
    """Apply the Swedish-universe filter to a (Stock, Field) OHLCV frame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(data.columns, pd.MultiIndex) or data.columns.nlevels < 2:
        raise ValueError("data must have MultiIndex columns in (Stock, Field) form.")

    stocks = data.columns.get_level_values(0).unique().tolist()
    included = set(
        filter_swedish_tickers(
            stocks,
            exclude_preference=exclude_preference,
            prefer_b_over_a=prefer_b_over_a,
            exclude_all_a=exclude_all_a,
        )
    )
    column_mask = data.columns.get_level_values(0).isin(included)
    return data.loc[:, column_mask].copy()
