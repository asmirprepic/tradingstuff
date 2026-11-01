import numpy as np
import pandas as pd


def _entropy_from_contributions(contributions, eligible_count):
    positive = pd.Series(contributions, dtype=float).clip(lower=0.0).dropna()
    positive = positive.loc[positive.gt(0.0)]
    total = positive.sum()
    if total <= 0.0 or eligible_count < 2:
        return np.nan, np.nan, pd.Series(dtype=float)
    shares = positive / total
    raw_entropy = float(-(shares * np.log(shares)).sum())
    normalized = raw_entropy / np.log(eligible_count)
    return raw_entropy, normalized, shares.sort_values(ascending=False)


def select_contribution_leaders(shares, cumulative_share=0.60, maximum_leaders=10):
    """Select the smallest leading set reaching a contribution-share threshold."""
    if not 0.0 < cumulative_share <= 1.0:
        raise ValueError("cumulative_share must be in (0, 1].")
    maximum_leaders = int(maximum_leaders)
    if maximum_leaders < 1:
        raise ValueError("maximum_leaders must be positive.")
    ordered = pd.Series(shares, dtype=float).dropna().sort_values(ascending=False)
    if ordered.empty:
        return []
    count = int(np.searchsorted(ordered.cumsum().to_numpy(), cumulative_share) + 1)
    return ordered.index[:min(count, maximum_leaders)].tolist()


def calculate_return_entropy(
    weekly_returns,
    stock_weights=None,
    leader_contribution_share=0.60,
    maximum_leaders=10,
    minimum_stocks=10,
):
    """Calculate separate upside/downside contribution entropy for one period."""
    returns = pd.to_numeric(pd.Series(weekly_returns), errors="coerce").dropna()
    if stock_weights is None:
        weights = pd.Series(1.0, index=returns.index)
    else:
        weights = pd.to_numeric(pd.Series(stock_weights), errors="coerce").reindex(
            returns.index
        )
        valid = weights.notna() & weights.gt(0.0)
        returns = returns.loc[valid]
        weights = weights.loc[valid]
    eligible_count = len(returns)
    if eligible_count < int(minimum_stocks):
        return {
            "EligibleStocks": eligible_count,
            "MarketReturn": np.nan,
            "PositiveBreadth": np.nan,
            "UpsideEntropy": np.nan,
            "DownsideEntropy": np.nan,
            "EffectiveContributors": np.nan,
            "Top5Share": np.nan,
            "Leaders": [],
            "UpsideShares": pd.Series(dtype=float),
            "DownsideShares": pd.Series(dtype=float),
        }

    weighted_returns = returns * weights
    upside_raw, upside_entropy, upside_shares = _entropy_from_contributions(
        weighted_returns.clip(lower=0.0), eligible_count
    )
    _, downside_entropy, downside_shares = _entropy_from_contributions(
        (-weighted_returns).clip(lower=0.0), eligible_count
    )
    leaders = select_contribution_leaders(
        upside_shares,
        cumulative_share=leader_contribution_share,
        maximum_leaders=maximum_leaders,
    )
    return {
        "EligibleStocks": eligible_count,
        "MarketReturn": float(returns.mean()),
        "PositiveBreadth": float(returns.gt(0.0).mean()),
        "UpsideEntropy": upside_entropy,
        "DownsideEntropy": downside_entropy,
        "EffectiveContributors": (
            float(np.exp(upside_raw)) if np.isfinite(upside_raw) else np.nan
        ),
        "Top5Share": float(upside_shares.head(5).sum()) if len(upside_shares) else np.nan,
        "Leaders": leaders,
        "UpsideShares": upside_shares,
        "DownsideShares": downside_shares,
    }
