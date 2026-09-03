import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path so imports work when invoked from repo root.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from data_handling.get_stock_data import GetStockDataTest
from data_handling.get_stock_tickers_upd import GetTickers
from agents.technical.bollinger_bands_agent import BollingerBandsAgent
from agents.technical.adx_dmi_agent import ADXDMIAgent
from agents.technical.high_low import HighLowAgent
from agents.technical.macd_agent import MACDAgent
from agents.technical.mean_reversion_agent import MeanReversionAgent
from agents.technical.momentum_agent import MomentumAgent
from agents.technical.moving_average_agent import MovingAverageAgent
from agents.technical.moving_average_crossover_agent import MovingAverageCrossoverAgent
from agents.technical.nr7_agent import NR7BreakoutAgent
from agents.technical.on_balance_volume_agent import OBVAgent
from agents.technical.performance_agent import PerformanceBasedAgent
from agents.technical.price_volume_trend_agent import PVTAgent
from agents.technical.rsi_agent import RSIAgent
from agents.technical.supertrend_agent import SupertrendAgent
from agents.technical.volume_price_divergence_agent import VolumePriceDivergenceAgent
from agents.technical.volume_weighted_average_price_agent import VWAPAgent
from agents.utils.evaluate import evaluations_from_agent


AGENT_ORDER = [
    "momentum",
    "moving_average",
    "moving_average_crossover",
    "macd",
    "bollinger",
    "adx_dmi",
    "mean_reversion",
    "rsi",
    "supertrend",
    "vwap",
    "obv",
    "pvt",
    "volume_price_divergence",
    "performance",
    "high_low",
    "nr7",
]

AGENT_FAMILIES = {
    "momentum": "trend",
    "moving_average": "trend",
    "moving_average_crossover": "trend",
    "macd": "trend",
    "adx_dmi": "trend",
    "supertrend": "trend",
    "performance": "trend",
    "bollinger": "mean_reversion",
    "mean_reversion": "mean_reversion",
    "rsi": "mean_reversion",
    "vwap": "volume_confirmation",
    "obv": "volume_confirmation",
    "pvt": "volume_confirmation",
    "volume_price_divergence": "volume_confirmation",
    "high_low": "breakout",
    "nr7": "breakout",
}


def make_synthetic_ohlcv(tickers, periods=260):
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq="B")
    frames = []

    for i, ticker in enumerate(tickers):
        base = 80.0 + i * 15.0
        trend = np.linspace(0.0, periods * (0.08 + i * 0.005), periods)
        swing = 3.0 * np.sin(np.arange(periods) / (7.0 + i))
        drift = 1.5 * np.cos(np.arange(periods) / (17.0 + i))
        close = base + trend + swing + drift
        close = np.maximum(close, 1.0)

        open_ = close * (1.0 + 0.002 * np.sin(np.arange(periods) / 5.0 + i))
        high = np.maximum(open_, close) + 0.8 + 0.2 * np.cos(np.arange(periods) / 6.0 + i)
        low = np.minimum(open_, close) - 0.8 - 0.2 * np.sin(np.arange(periods) / 8.0 + i)
        low = np.maximum(low, 0.5)
        volume = 1_000_000 + i * 250_000 + np.arange(periods) * 1_500

        df = pd.DataFrame(
            {
                (ticker, "Open"): open_,
                (ticker, "High"): high,
                (ticker, "Low"): low,
                (ticker, "Close"): close,
                (ticker, "Volume"): volume,
            },
            index=idx,
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        frames.append(df)

    return pd.concat(frames, axis=1).sort_index()


def read_tickers_file(path):
    file_path = Path(path)
    if not file_path.exists():
        raise SystemExit(f"Tickers file not found: {path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
        for col in ("ticker", "Ticker", "tickers", "Tickers"):
            if col in df.columns:
                return [str(x).strip() for x in df[col].dropna().unique()]
        first = df.columns[0]
        return [str(x).strip() for x in df[first].dropna().unique()]

    text = file_path.read_text()
    return [s.strip() for s in text.replace(",", "\n").splitlines() if s.strip()]


def fetch_tickers_by_region(region):
    getter = GetTickers()
    try:
        res = getter.get_tickers_by_market_cap(region)
    finally:
        getter.close()

    all_tickers = []
    for values in res.values():
        all_tickers.extend(values)

    seen = set()
    deduped = []
    for ticker in all_tickers:
        if ticker not in seen:
            seen.add(ticker)
            deduped.append(ticker)
    return deduped


def load_price_df(tickers, args):
    if args.use_synthetic:
        print(f"Using synthetic OHLCV data for {tickers}")
        return make_synthetic_ohlcv(tickers, periods=args.synthetic_periods)

    start = args.start
    end = args.end

    if (start is None or end is None) and args.lookback_days is not None:
        end_ts = pd.Timestamp.today().normalize()
        start_ts = end_ts - pd.tseries.offsets.BDay(args.lookback_days)
        start = start_ts.strftime("%Y-%m-%d")
        end = end_ts.strftime("%Y-%m-%d")

    if not start or not end:
        raise SystemExit("Provide --start and --end, or use --lookback-days (unless --use-synthetic).")

    print(f"Fetching OHLCV for {tickers} from {start} to {end} (interval={args.interval})...")
    getter = GetStockDataTest(
        stocks=tickers,
        startdate=start,
        enddate=end,
        interval=args.interval,
        data_types=["Open", "High", "Low", "Close", "Volume"],
    )
    return getter.getData()


def parse_agent_names(agent_arg):
    requested = [a.strip().lower() for a in agent_arg.split(",") if a.strip()]
    if not requested or requested == ["all"]:
        return list(AGENT_ORDER)

    unknown = sorted(set(requested) - set(AGENT_ORDER))
    if unknown:
        raise SystemExit(f"Unknown agent(s): {', '.join(unknown)}. Available: {', '.join(AGENT_ORDER)}")

    return requested


def build_agent(agent_name, price_df):
    stock_count = len(price_df.columns.get_level_values(0).unique())

    if agent_name == "momentum":
        return MomentumAgent(price_df, lookbacks=[20, 60], score_mode="z")
    if agent_name == "moving_average":
        return MovingAverageAgent(price_df, short_window=50, long_window=200)
    if agent_name == "moving_average_crossover":
        return MovingAverageCrossoverAgent(price_df, short_window=50, long_window=200)
    if agent_name == "macd":
        return MACDAgent(price_df, short_window=12, long_window=26, signal_window=9)
    if agent_name == "bollinger":
        return BollingerBandsAgent(price_df, period=20, num_std_dev=2.0)
    if agent_name == "adx_dmi":
        return ADXDMIAgent(price_df, period=14, adx_threshold=20.0)
    if agent_name == "mean_reversion":
        return MeanReversionAgent(price_df, lookback_period=20, threshold=2.0)
    if agent_name == "rsi":
        return RSIAgent(price_df, period=14, upper_band=70, lower_band=30)
    if agent_name == "supertrend":
        return SupertrendAgent(price_df, period=10, multiplier=3.0)
    if agent_name == "vwap":
        return VWAPAgent(price_df, period=20, threshold=0.03)
    if agent_name == "obv":
        return OBVAgent(price_df, threshold=0.05)
    if agent_name == "pvt":
        return PVTAgent(price_df, threshold=0.05)
    if agent_name == "volume_price_divergence":
        return VolumePriceDivergenceAgent(price_df, window=2, threshold=0.005)
    if agent_name == "performance":
        return PerformanceBasedAgent(
            price_df,
            period_length=20,
            top_n=max(1, min(5, stock_count)),
            holding_period=20,
        )
    if agent_name == "high_low":
        return HighLowAgent(price_df)
    if agent_name == "nr7":
        return NR7BreakoutAgent(price_df, hold_days=5, take_shorts=False)

    raise KeyError(f"Unhandled agent: {agent_name}")


def build_agent_summary(agent_name, agent, recs):
    strategy_key = f"{agent.algorithm_name}_return"
    returns_df = pd.DataFrame(agent.returns_data).T if agent.returns_data else pd.DataFrame()

    if returns_df.empty:
        avg_strategy = float("nan")
        median_strategy = float("nan")
        avg_buyhold = float("nan")
        profitable = 0
        avg_entries = float("nan")
    else:
        avg_strategy = float(returns_df[strategy_key].mean())
        median_strategy = float(returns_df[strategy_key].median())
        avg_buyhold = float(returns_df["buy_and_hold_return"].mean())
        profitable = int((returns_df[strategy_key] > 0).sum())
        avg_entries = float(returns_df["total_entries"].mean())

    recs = recs.copy()
    if recs.empty:
        latest_buys = 0
        latest_sells = 0
        latest_holds = 0
        avg_score = float("nan")
        top_pick = None
        top_pick_score = float("nan")
    else:
        latest_buys = int((recs["Recommendation"] == "Buy").sum())
        latest_sells = int((recs["Recommendation"] == "Sell").sum())
        latest_holds = int((recs["Recommendation"] == "Hold").sum())
        avg_score = float(recs["Score"].dropna().mean()) if recs["Score"].notna().any() else float("nan")
        top_pick = str(recs.iloc[0]["Stock"])
        top_pick_score = float(recs.iloc[0]["Score"]) if pd.notna(recs.iloc[0]["Score"]) else float("nan")

    return {
        "Agent": agent_name,
        "Algorithm": agent.algorithm_name,
        "Stocks": len(agent.signal_data),
        "AvgStrategyReturnPct": avg_strategy,
        "MedianStrategyReturnPct": median_strategy,
        "AvgBuyHoldReturnPct": avg_buyhold,
        "ProfitableStocks": profitable,
        "AvgEntries": avg_entries,
        "LatestBuys": latest_buys,
        "LatestSells": latest_sells,
        "LatestHolds": latest_holds,
        "AvgScore": avg_score,
        "TopPick": top_pick,
        "TopPickScore": top_pick_score,
    }


def build_consensus_table(all_recs):
    if all_recs.empty:
        return pd.DataFrame()

    rows = []
    for stock, group in all_recs.groupby("Stock"):
        ordered = group.sort_values("Score", ascending=False, na_position="last")
        rows.append(
            {
                "Stock": stock,
                "BuyCount": int((group["Recommendation"] == "Buy").sum()),
                "SellCount": int((group["Recommendation"] == "Sell").sum()),
                "HoldCount": int((group["Recommendation"] == "Hold").sum()),
                "MeanScore": float(group["Score"].dropna().mean()) if group["Score"].notna().any() else float("nan"),
                "BestAgent": ordered.iloc[0]["Agent"],
                "BestAgentScore": ordered.iloc[0]["Score"],
                "SupportingAgents": ", ".join(group["Agent"].tolist()),
            }
        )

    consensus = pd.DataFrame(rows)
    return consensus.sort_values(["BuyCount", "MeanScore"], ascending=[False, False], na_position="last").reset_index(drop=True)


def build_family_summary_table(all_recs, summary_df):
    if all_recs.empty:
        return pd.DataFrame()

    recs = all_recs.copy()
    recs["Family"] = recs["Agent"].map(AGENT_FAMILIES).fillna("other")

    summary_by_agent = pd.DataFrame()
    if summary_df is not None and not summary_df.empty and "Agent" in summary_df.columns:
        summary_by_agent = summary_df.copy()
        summary_by_agent["Family"] = summary_by_agent["Agent"].map(AGENT_FAMILIES).fillna("other")

    rows = []
    family_order = ["trend", "mean_reversion", "volume_confirmation", "breakout", "other"]
    families = [f for f in family_order if f in recs["Family"].unique()]

    for family in families:
        group = recs[recs["Family"] == family]
        ordered_group = group.sort_values("Score", ascending=False, na_position="last")
        buy_group = group[group["Recommendation"] == "Buy"]
        sell_group = group[group["Recommendation"] == "Sell"]
        hold_group = group[group["Recommendation"] == "Hold"]
        family_agents = group["Agent"].drop_duplicates().tolist()

        family_summary = summary_by_agent[summary_by_agent["Family"] == family] if not summary_by_agent.empty else pd.DataFrame()
        if family_summary.empty:
            avg_strategy_return = float("nan")
            median_strategy_return = float("nan")
            avg_agent_score = float("nan")
            top_agent = None
            top_agent_return = float("nan")
        else:
            avg_strategy_return = float(family_summary["AvgStrategyReturnPct"].mean())
            median_strategy_return = float(family_summary["AvgStrategyReturnPct"].median())
            avg_agent_score = float(family_summary["AvgScore"].mean())
            top_agent_row = family_summary.sort_values("AvgStrategyReturnPct", ascending=False, na_position="last").iloc[0]
            top_agent = top_agent_row["Agent"]
            top_agent_return = top_agent_row["AvgStrategyReturnPct"]

        if buy_group.empty:
            top_buy_stock = None
            top_buy_stock_count = 0
            top_buy_stock_mean_score = float("nan")
        else:
            buy_stock_summary = (
                buy_group.groupby("Stock")
                .agg(
                    BuyCount=("Stock", "size"),
                    MeanScore=("Score", "mean"),
                )
                .sort_values(["BuyCount", "MeanScore"], ascending=[False, False], na_position="last")
                .reset_index()
            )
            top_buy_stock = buy_stock_summary.iloc[0]["Stock"]
            top_buy_stock_count = int(buy_stock_summary.iloc[0]["BuyCount"])
            top_buy_stock_mean_score = float(buy_stock_summary.iloc[0]["MeanScore"])

        total_signals = len(group)
        buy_count = int(len(buy_group))
        sell_count = int(len(sell_group))
        hold_count = int(len(hold_group))

        rows.append(
            {
                "Family": family,
                "AgentCount": int(len(family_agents)),
                "Agents": _join_agents(family_agents),
                "StocksCovered": int(group["Stock"].nunique()),
                "TotalSignals": total_signals,
                "BuySignals": buy_count,
                "SellSignals": sell_count,
                "HoldSignals": hold_count,
                "NetSignals": buy_count - sell_count,
                "BuySignalPct": buy_count / total_signals if total_signals else 0.0,
                "SellSignalPct": sell_count / total_signals if total_signals else 0.0,
                "MeanScore": float(group["Score"].mean()) if group["Score"].notna().any() else float("nan"),
                "BestAgent": ordered_group.iloc[0]["Agent"],
                "BestAgentScore": ordered_group.iloc[0]["Score"],
                "AvgAgentStrategyReturnPct": avg_strategy_return,
                "MedianAgentStrategyReturnPct": median_strategy_return,
                "AvgAgentScore": avg_agent_score,
                "TopAgentByReturn": top_agent,
                "TopAgentStrategyReturnPct": top_agent_return,
                "TopBuyStock": top_buy_stock,
                "TopBuyStockCount": top_buy_stock_count,
                "TopBuyStockMeanScore": top_buy_stock_mean_score,
            }
        )

    family_summary_df = pd.DataFrame(rows)
    return family_summary_df.sort_values(
        ["NetSignals", "BuySignalPct", "AvgAgentStrategyReturnPct"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _percentile_rank_desc(series):
    if len(series) <= 1:
        return pd.Series(1.0, index=series.index, dtype=float)

    ranks = series.rank(method="first", ascending=False)
    return 1.0 - ((ranks - 1.0) / (len(series) - 1.0))


def _join_agents(values):
    return ", ".join(str(v) for v in values if pd.notna(v) and str(v))


def _support_count(group, family, recommendation):
    mask = (group["Family"] == family) & (group["Recommendation"] == recommendation)
    return int(mask.sum())


def _shortlist_tier(buy_count, sell_count, trend_buy_count, volume_buy_count, buy_family_breadth):
    if buy_count >= 4 and trend_buy_count >= 2 and volume_buy_count >= 1 and sell_count == 0:
        return "TierA"
    if buy_count >= 3 and trend_buy_count >= 1 and buy_family_breadth >= 2 and sell_count <= 1:
        return "TierB"
    if buy_count >= 2 and buy_family_breadth >= 2 and sell_count <= 1:
        return "TierC"
    if buy_count >= 2 and sell_count >= 2:
        return "Mixed"
    if sell_count > buy_count:
        return "Avoid"
    return "Watch"


def build_stock_shortlist_table(all_recs):
    if all_recs.empty:
        return pd.DataFrame()

    ranked = all_recs.copy()
    ranked["Family"] = ranked["Agent"].map(AGENT_FAMILIES).fillna("other")
    ranked["AgentRankPct"] = ranked.groupby("Agent")["Score"].transform(_percentile_rank_desc)

    rows = []
    for stock, group in ranked.groupby("Stock"):
        ordered = group.sort_values("Score", ascending=False, na_position="last")
        buy_group = group[group["Recommendation"] == "Buy"]
        sell_group = group[group["Recommendation"] == "Sell"]
        hold_group = group[group["Recommendation"] == "Hold"]
        total_agents = len(group)

        buy_count = int(len(buy_group))
        sell_count = int(len(sell_group))
        hold_count = int(len(hold_group))
        net_bias = buy_count - sell_count
        conflict_count = min(buy_count, sell_count)
        buy_support_pct = buy_count / total_agents if total_agents else 0.0
        sell_support_pct = sell_count / total_agents if total_agents else 0.0
        mean_score = float(group["Score"].dropna().mean()) if group["Score"].notna().any() else float("nan")
        buy_mean_score = float(buy_group["Score"].dropna().mean()) if buy_group["Score"].notna().any() else float("nan")
        mean_agent_rank_pct = float(group["AgentRankPct"].mean()) if group["AgentRankPct"].notna().any() else float("nan")
        buy_rank_pct = float(buy_group["AgentRankPct"].mean()) if buy_group["AgentRankPct"].notna().any() else 0.0
        trend_buy_count = _support_count(group, "trend", "Buy")
        mean_reversion_buy_count = _support_count(group, "mean_reversion", "Buy")
        volume_buy_count = _support_count(group, "volume_confirmation", "Buy")
        breakout_buy_count = _support_count(group, "breakout", "Buy")
        buy_family_breadth = int(buy_group["Family"].nunique()) if not buy_group.empty else 0
        support_families = _join_agents(buy_group["Family"].drop_duplicates().tolist())
        supporting_agents = _join_agents(buy_group["Agent"].tolist())
        opposing_agents = _join_agents(sell_group["Agent"].tolist())
        best_buy_agent = buy_group.sort_values("Score", ascending=False, na_position="last").iloc[0]["Agent"] if not buy_group.empty else None
        best_buy_score = buy_group.sort_values("Score", ascending=False, na_position="last").iloc[0]["Score"] if not buy_group.empty else float("nan")

        if buy_count > sell_count:
            consensus_recommendation = "Buy"
        elif sell_count > buy_count:
            consensus_recommendation = "Sell"
        else:
            consensus_recommendation = "Hold"

        shortlist_tier = _shortlist_tier(
            buy_count=buy_count,
            sell_count=sell_count,
            trend_buy_count=trend_buy_count,
            volume_buy_count=volume_buy_count,
            buy_family_breadth=buy_family_breadth,
        )

        rows.append(
            {
                "Stock": stock,
                "ShortlistTier": shortlist_tier,
                "ConsensusRecommendation": consensus_recommendation,
                "TotalAgents": total_agents,
                "BuyCount": buy_count,
                "SellCount": sell_count,
                "HoldCount": hold_count,
                "NetBias": net_bias,
                "ConflictCount": conflict_count,
                "BuySupportPct": buy_support_pct,
                "SellSupportPct": sell_support_pct,
                "TrendBuyCount": trend_buy_count,
                "MeanReversionBuyCount": mean_reversion_buy_count,
                "VolumeBuyCount": volume_buy_count,
                "BreakoutBuyCount": breakout_buy_count,
                "BuyFamilyBreadth": buy_family_breadth,
                "SupportFamilies": support_families,
                "MeanScore": mean_score,
                "BuyMeanScore": buy_mean_score,
                "MeanAgentRankPct": mean_agent_rank_pct,
                "BuyAgentRankPct": buy_rank_pct,
                "BestAgent": ordered.iloc[0]["Agent"],
                "BestAgentScore": ordered.iloc[0]["Score"],
                "BestBuyAgent": best_buy_agent,
                "BestBuyScore": best_buy_score,
                "SupportingAgents": supporting_agents,
                "OpposingAgents": opposing_agents,
            }
        )

    shortlist = pd.DataFrame(rows)
    tier_order = {"TierA": 0, "TierB": 1, "TierC": 2, "Watch": 3, "Mixed": 4, "Avoid": 5}
    shortlist["TierOrder"] = shortlist["ShortlistTier"].map(tier_order).fillna(99)
    shortlist = shortlist.sort_values(
        ["TierOrder", "BuyCount", "ConflictCount", "BuyFamilyBreadth", "BuyAgentRankPct", "MeanScore"],
        ascending=[True, False, True, False, False, False],
        na_position="last",
    ).drop(columns=["TierOrder"]).reset_index(drop=True)
    return shortlist


def build_stock_ranking_table(all_recs, summary_df=None):
    return build_stock_shortlist_table(all_recs)


def resolve_output_path(path_str, timestamp_output):
    if not path_str:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "{ts}" in path_str:
        return path_str.replace("{ts}", timestamp)

    if timestamp_output:
        p = Path(path_str)
        suffix = p.suffix or ".csv"
        stem = p.stem if p.suffix else p.name
        return str(p.with_name(f"{stem}_{timestamp}{suffix}"))

    return path_str


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run and compare technical trading agents")
    parser.add_argument("--agents", type=str, default="all", help=f"Comma-separated agents or 'all'. Available: {', '.join(AGENT_ORDER)}")
    parser.add_argument("--list-agents", action="store_true", help="Print available agent names and exit")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA", help="Comma-separated ticker list")
    parser.add_argument("--tickers-file", type=str, default=None, help="Path to a file with tickers (csv or txt)")
    parser.add_argument("--fetch-tickers", type=str, default=None, help="Fetch tickers from Yahoo by region (e.g. US)")
    parser.add_argument("--fetch-out", type=str, default=None, help="If --fetch-tickers is used, save fetched tickers to this file")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=260, help="If --start/--end omitted, use last N business days")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic OHLCV data instead of Yahoo")
    parser.add_argument("--synthetic-periods", type=int, default=260, help="Synthetic dataset length in bars")
    parser.add_argument("--persistence", type=int, default=1, help="Require stable Position for N rows in recommendations")
    parser.add_argument("--top-n-per-agent", type=int, default=None, help="Keep only top N recommendations per agent")
    parser.add_argument("--summary-output", type=str, default="technical_agent_summary.csv", help="CSV path for per-agent summary")
    parser.add_argument("--recommendations-output", type=str, default="technical_agent_recommendations.csv", help="CSV path for combined per-agent recommendations")
    parser.add_argument("--consensus-output", type=str, default="technical_agent_consensus.csv", help="CSV path for grouped stock consensus")
    parser.add_argument("--family-summary-output", type=str, default="technical_agent_family_summary.csv", help="CSV path for family-level summary")
    parser.add_argument(
        "--shortlist-output",
        "--ranking-output",
        dest="shortlist_output",
        type=str,
        default="technical_agent_shortlist.csv",
        help="CSV path for tiered stock shortlist",
    )
    parser.add_argument("--timestamp-output", action="store_true", help="Append timestamp to outputs unless '{ts}' is present")

    args = parser.parse_args(argv)

    if args.list_agents:
        print("\n".join(AGENT_ORDER))
        return 0

    if args.fetch_tickers:
        print(f"Fetching tickers for region {args.fetch_tickers}...")
        tickers = fetch_tickers_by_region(args.fetch_tickers)
        if args.fetch_out:
            Path(args.fetch_out).write_text("\n".join(tickers))
            print(f"Saved fetched tickers to {args.fetch_out}")
    elif args.tickers_file:
        tickers = read_tickers_file(args.tickers_file)
    else:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    if not tickers:
        raise SystemExit("No tickers resolved.")

    selected_agents = parse_agent_names(args.agents)
    price_df = load_price_df(tickers, args)
    if price_df.empty:
        raise SystemExit("No price data available.")

    summary_rows = []
    recommendation_frames = []
    failed_agents = []

    for agent_name in selected_agents:
        try:
            print(f"Running {agent_name}...")
            agent = build_agent(agent_name, price_df)
            recs = evaluations_from_agent(agent, persistence=args.persistence, top_n=args.top_n_per_agent, save_path=None)
            recs.insert(0, "Agent", agent_name)
            summary_rows.append(build_agent_summary(agent_name, agent, recs))
            recommendation_frames.append(recs)
        except Exception as exc:
            failed_agents.append((agent_name, str(exc)))

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["AvgStrategyReturnPct", "ProfitableStocks", "AvgScore"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    all_recs = pd.concat(recommendation_frames, ignore_index=True) if recommendation_frames else pd.DataFrame()
    consensus_df = build_consensus_table(all_recs)
    family_summary_df = build_family_summary_table(all_recs, summary_df)
    shortlist_df = build_stock_shortlist_table(all_recs)

    summary_out = resolve_output_path(args.summary_output, args.timestamp_output)
    recs_out = resolve_output_path(args.recommendations_output, args.timestamp_output)
    consensus_out = resolve_output_path(args.consensus_output, args.timestamp_output)
    family_summary_out = resolve_output_path(args.family_summary_output, args.timestamp_output)
    shortlist_out = resolve_output_path(args.shortlist_output, args.timestamp_output)

    if summary_out and not summary_df.empty:
        summary_df.to_csv(summary_out, index=False)
    if recs_out and not all_recs.empty:
        all_recs.to_csv(recs_out, index=False)
    if consensus_out and not consensus_df.empty:
        consensus_df.to_csv(consensus_out, index=False)
    if family_summary_out and not family_summary_df.empty:
        family_summary_df.to_csv(family_summary_out, index=False)
    if shortlist_out and not shortlist_df.empty:
        shortlist_df.to_csv(shortlist_out, index=False)

    print("\nAgent Summary:")
    print(summary_df if not summary_df.empty else "No agent summaries produced.")

    print("\nConsensus:")
    print(consensus_df if not consensus_df.empty else "No consensus rows produced.")

    print("\nFamily Summary:")
    print(family_summary_df if not family_summary_df.empty else "No family summary rows produced.")

    print("\nShortlist:")
    print(shortlist_df if not shortlist_df.empty else "No shortlist rows produced.")

    if failed_agents:
        print("\nFailed Agents:")
        for agent_name, error in failed_agents:
            print(f"- {agent_name}: {error}")

    if summary_out and not summary_df.empty:
        print(f"\nWrote summary to {summary_out}")
    if recs_out and not all_recs.empty:
        print(f"Wrote recommendations to {recs_out}")
    if consensus_out and not consensus_df.empty:
        print(f"Wrote consensus to {consensus_out}")
    if family_summary_out and not family_summary_df.empty:
        print(f"Wrote family summary to {family_summary_out}")
    if shortlist_out and not shortlist_df.empty:
        print(f"Wrote shortlist to {shortlist_out}")

    return {
        "summary": summary_df,
        "recommendations": all_recs,
        "consensus": consensus_df,
        "family_summary": family_summary_df,
        "shortlist": shortlist_df,
        "ranking": shortlist_df,
        "failed_agents": failed_agents,
    }


if __name__ == "__main__":
    main()
