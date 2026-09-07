import argparse
import html
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_SUMMARY = "technical_agent_summary.csv"
DEFAULT_RECOMMENDATIONS = "technical_agent_recommendations.csv"
DEFAULT_CONSENSUS = "technical_agent_consensus.csv"
DEFAULT_FAMILY_SUMMARY = "technical_agent_family_summary.csv"
DEFAULT_SHORTLIST = "technical_agent_shortlist.csv"
DEFAULT_OUTPUT = "outputs/dashboard/technical_dashboard.html"

TIER_ORDER = {
    "TierA": 0,
    "TierB": 1,
    "TierC": 2,
    "Watch": 3,
    "Mixed": 4,
    "Avoid": 5,
}

TIER_LABELS = {
    "TierA": "Tier A",
    "TierB": "Tier B",
    "TierC": "Tier C",
    "Watch": "Watch",
    "Mixed": "Mixed",
    "Avoid": "Avoid",
}


def read_csv_or_empty(path):
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def format_number(value, digits=2):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def format_percent(value, digits=1):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}%"


def format_integer(value):
    if pd.isna(value):
        return "—"
    return str(int(value))


def normalize_shortlist(shortlist_df):
    if shortlist_df.empty:
        return shortlist_df

    shortlist = shortlist_df.copy()
    shortlist["TierOrder"] = shortlist["ShortlistTier"].map(TIER_ORDER).fillna(99)
    shortlist = shortlist.sort_values(
        ["TierOrder", "BuyCount", "ConflictCount", "BuyFamilyBreadth", "BuyAgentRankPct", "MeanScore"],
        ascending=[True, False, True, False, False, False],
        na_position="last",
    ).drop(columns=["TierOrder"])
    return shortlist.reset_index(drop=True)


def build_dashboard_context(summary_df, family_df, consensus_df, shortlist_df, recommendations_df):
    summary = summary_df.copy() if summary_df is not None else pd.DataFrame()
    family = family_df.copy() if family_df is not None else pd.DataFrame()
    consensus = consensus_df.copy() if consensus_df is not None else pd.DataFrame()
    shortlist = normalize_shortlist(shortlist_df.copy() if shortlist_df is not None else pd.DataFrame())
    recommendations = recommendations_df.copy() if recommendations_df is not None else pd.DataFrame()

    metric_values = {
        "agents": len(summary),
        "families": len(family),
        "shortlist_rows": len(shortlist),
        "consensus_rows": len(consensus),
        "recommendation_rows": len(recommendations),
        "tier_a_rows": int((shortlist["ShortlistTier"] == "TierA").sum()) if not shortlist.empty else 0,
        "buy_rows": int((shortlist["ConsensusRecommendation"] == "Buy").sum()) if not shortlist.empty else 0,
        "sell_rows": int((shortlist["ConsensusRecommendation"] == "Sell").sum()) if not shortlist.empty else 0,
    }

    if not summary.empty and "AvgStrategyReturnPct" in summary.columns:
        summary = summary.sort_values("AvgStrategyReturnPct", ascending=False, na_position="last").reset_index(drop=True)

    if not family.empty and "NetSignals" in family.columns:
        family = family.sort_values(
            ["NetSignals", "BuySignalPct", "AvgAgentStrategyReturnPct"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    if not consensus.empty and "BuyCount" in consensus.columns:
        consensus = consensus.sort_values(
            ["BuyCount", "MeanScore"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "family": family,
        "consensus": consensus,
        "shortlist": shortlist,
        "recommendations": recommendations,
        "metrics": metric_values,
    }


def render_metric_card(title, value, subtitle=None, tone="neutral"):
    subtitle_html = f'<div class="card-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    return f"""
    <div class="metric-card metric-{tone}">
      <div class="metric-title">{html.escape(title)}</div>
      <div class="metric-value">{html.escape(str(value))}</div>
      {subtitle_html}
    </div>
    """


def render_agent_summary_table(df):
    if df.empty:
        return "<p class='empty-state'>No agent summary data found.</p>"

    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"""
            <tr>
              <td>{html.escape(str(row.get("Agent", "")))}</td>
              <td>{html.escape(str(row.get("Algorithm", "")))}</td>
              <td class="num">{format_integer(row.get("Stocks"))}</td>
              <td class="num">{format_number(row.get("AvgStrategyReturnPct"))}</td>
              <td class="num">{format_number(row.get("MedianStrategyReturnPct"))}</td>
              <td class="num">{format_number(row.get("AvgBuyHoldReturnPct"))}</td>
              <td class="num">{format_integer(row.get("ProfitableStocks"))}</td>
              <td class="num">{format_number(row.get("AvgEntries"))}</td>
              <td class="num">{format_number(row.get("AvgScore"))}</td>
              <td>{html.escape(str(row.get("TopPick", "")))}</td>
            </tr>
            """
        )

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Agent</th>
          <th>Algorithm</th>
          <th class="num">Stocks</th>
          <th class="num">Avg Return</th>
          <th class="num">Median Return</th>
          <th class="num">Buy/Hold</th>
          <th class="num">Profitable</th>
          <th class="num">Avg Entries</th>
          <th class="num">Avg Score</th>
          <th>Top Pick</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


def render_family_summary_table(df):
    if df.empty:
        return "<p class='empty-state'>No family summary data found.</p>"

    rows = []
    for _, row in df.iterrows():
        buy_pct = float(row.get("BuySignalPct") or 0.0)
        sell_pct = float(row.get("SellSignalPct") or 0.0)
        rows.append(
            f"""
            <tr>
              <td><strong>{html.escape(str(row.get("Family", "")))}</strong></td>
              <td class="num">{format_integer(row.get("AgentCount"))}</td>
              <td>{html.escape(str(row.get("Agents", "")))}</td>
              <td class="num">{format_integer(row.get("StocksCovered"))}</td>
              <td class="num">{format_integer(row.get("TotalSignals"))}</td>
              <td class="num">{format_integer(row.get("BuySignals"))}</td>
              <td class="num">{format_integer(row.get("SellSignals"))}</td>
              <td class="num">{format_integer(row.get("HoldSignals"))}</td>
              <td class="num">{format_integer(row.get("NetSignals"))}</td>
              <td>
                <div class="bar-wrap">
                  <div class="bar buy" style="width: {min(buy_pct * 100.0, 100.0):.1f}%"></div>
                </div>
                <span class="bar-label">{format_percent(buy_pct * 100.0)}</span>
              </td>
              <td>
                <div class="bar-wrap">
                  <div class="bar sell" style="width: {min(sell_pct * 100.0, 100.0):.1f}%"></div>
                </div>
                <span class="bar-label">{format_percent(sell_pct * 100.0)}</span>
              </td>
              <td class="num">{format_number(row.get("AvgAgentStrategyReturnPct"))}</td>
              <td>{html.escape(str(row.get("TopAgentByReturn", "")))}</td>
              <td>{html.escape(str(row.get("TopBuyStock", "")))}</td>
            </tr>
            """
        )

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Family</th>
          <th class="num">Agents</th>
          <th>Agent List</th>
          <th class="num">Stocks</th>
          <th class="num">Signals</th>
          <th class="num">Buys</th>
          <th class="num">Sells</th>
          <th class="num">Holds</th>
          <th class="num">Net</th>
          <th>Buy Share</th>
          <th>Sell Share</th>
          <th class="num">Avg Strategy Return</th>
          <th>Top Agent</th>
          <th>Top Buy Stock</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


def render_consensus_table(df):
    if df.empty:
        return "<p class='empty-state'>No consensus data found.</p>"

    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"""
            <tr>
              <td><strong>{html.escape(str(row.get("Stock", "")))}</strong></td>
              <td class="num">{format_integer(row.get("BuyCount"))}</td>
              <td class="num">{format_integer(row.get("SellCount"))}</td>
              <td class="num">{format_integer(row.get("HoldCount"))}</td>
              <td class="num">{format_number(row.get("MeanScore"))}</td>
              <td>{html.escape(str(row.get("BestAgent", "")))}</td>
              <td class="num">{format_number(row.get("BestAgentScore"))}</td>
              <td>{html.escape(str(row.get("SupportingAgents", "")))}</td>
            </tr>
            """
        )

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Stock</th>
          <th class="num">Buy</th>
          <th class="num">Sell</th>
          <th class="num">Hold</th>
          <th class="num">Mean Score</th>
          <th>Best Agent</th>
          <th class="num">Best Score</th>
          <th>Supporting Agents</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


def tier_badge(tier):
    label = TIER_LABELS.get(tier, tier or "Unknown")
    tier_key = str(tier or "unknown").lower()
    return f"<span class='tier-badge tier-{html.escape(tier_key)}'>{html.escape(label)}</span>"


def render_shortlist_rows(df):
    if df.empty:
        return "<tr><td colspan='17' class='empty-state'>No shortlist data found.</td></tr>"

    rows = []
    for _, row in df.iterrows():
        search_text = " ".join(
            str(row.get(col, ""))
            for col in ("Stock", "ShortlistTier", "ConsensusRecommendation", "SupportFamilies", "SupportingAgents", "OpposingAgents", "BestAgent", "BestBuyAgent")
        ).lower()
        rows.append(
            f"""
            <tr data-tier="{html.escape(str(row.get("ShortlistTier", "")))}" data-search="{html.escape(search_text)}">
              <td><strong>{html.escape(str(row.get("Stock", "")))}</strong></td>
              <td>{tier_badge(row.get("ShortlistTier"))}</td>
              <td>{html.escape(str(row.get("ConsensusRecommendation", "")))}</td>
              <td class="num">{format_integer(row.get("TotalAgents"))}</td>
              <td class="num">{format_integer(row.get("BuyCount"))}</td>
              <td class="num">{format_integer(row.get("SellCount"))}</td>
              <td class="num">{format_integer(row.get("HoldCount"))}</td>
              <td class="num">{format_integer(row.get("NetBias"))}</td>
              <td class="num">{format_integer(row.get("ConflictCount"))}</td>
              <td class="num">{format_percent(row.get("BuySupportPct") * 100.0 if pd.notna(row.get("BuySupportPct")) else float("nan"))}</td>
              <td class="num">{format_percent(row.get("SellSupportPct") * 100.0 if pd.notna(row.get("SellSupportPct")) else float("nan"))}</td>
              <td class="num">{format_integer(row.get("TrendBuyCount"))}</td>
              <td class="num">{format_integer(row.get("MeanReversionBuyCount"))}</td>
              <td class="num">{format_integer(row.get("VolumeBuyCount"))}</td>
              <td class="num">{format_integer(row.get("BreakoutBuyCount"))}</td>
              <td>{html.escape(str(row.get("SupportFamilies", "")))}</td>
              <td>{html.escape(str(row.get("SupportingAgents", "")))}</td>
            </tr>
            """
        )

    return "".join(rows)


def render_shortlist_table(df):
    if df.empty:
        return "<p class='empty-state'>No shortlist data found.</p>"

    return f"""
    <div class="toolbar">
      <input id="shortlist-search" type="search" placeholder="Search ticker, agent, or family..." />
      <select id="tier-filter">
        <option value="">All tiers</option>
        <option value="TierA">Tier A</option>
        <option value="TierB">Tier B</option>
        <option value="TierC">Tier C</option>
        <option value="Watch">Watch</option>
        <option value="Mixed">Mixed</option>
        <option value="Avoid">Avoid</option>
      </select>
    </div>
    <div class="table-scroll">
      <table class="data-table shortlist-table" id="shortlist-table">
        <thead>
          <tr>
            <th>Stock</th>
            <th>Tier</th>
            <th>Consensus</th>
            <th class="num">Agents</th>
            <th class="num">Buys</th>
            <th class="num">Sells</th>
            <th class="num">Holds</th>
            <th class="num">Net</th>
            <th class="num">Conflict</th>
            <th class="num">Buy Share</th>
            <th class="num">Sell Share</th>
            <th class="num">Trend Buys</th>
            <th class="num">Reversion Buys</th>
            <th class="num">Volume Buys</th>
            <th class="num">Breakout Buys</th>
            <th>Support Families</th>
            <th>Supporting Agents</th>
          </tr>
        </thead>
        <tbody>
          {render_shortlist_rows(df)}
        </tbody>
      </table>
    </div>
    """


def render_html(context):
    metrics = context["metrics"]
    summary_html = render_agent_summary_table(context["summary"])
    family_html = render_family_summary_table(context["family"])
    consensus_html = render_consensus_table(context["consensus"])
    shortlist_html = render_shortlist_table(context["shortlist"])

    cards = "".join(
        [
            render_metric_card("Agents", metrics["agents"], "Per-agent summary rows", "neutral"),
            render_metric_card("Families", metrics["families"], "Trend, mean reversion, volume, breakout", "neutral"),
            render_metric_card("Shortlist Rows", metrics["shortlist_rows"], "Screened stocks", "neutral"),
            render_metric_card("Tier A", metrics["tier_a_rows"], "Highest conviction tier", "positive"),
            render_metric_card("Buy Consensus", metrics["buy_rows"], "Shortlist buy recommendations", "positive"),
            render_metric_card("Sell Consensus", metrics["sell_rows"], "Shortlist sell recommendations", "negative"),
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Technical Agent Dashboard</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #ffffff;
      --panel-alt: #fbfaf7;
      --text: #1f2933;
      --muted: #667085;
      --line: #d8d2c7;
      --accent: #0f766e;
      --accent-soft: #d9f0ed;
      --positive: #166534;
      --positive-soft: #dcfce7;
      --negative: #991b1b;
      --negative-soft: #fee2e2;
      --warning: #92400e;
      --warning-soft: #fef3c7;
      --shadow: 0 10px 30px rgba(31, 41, 51, 0.08);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 30%),
        radial-gradient(circle at top right, rgba(22, 101, 52, 0.08), transparent 28%),
        var(--bg);
    }}
    .shell {{ max-width: 1600px; margin: 0 auto; padding: 28px 22px 48px; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
    }}
    .hero-card, .panel, .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero-card {{
      padding: 26px;
      background: linear-gradient(135deg, rgba(15, 118, 110, 0.08), rgba(255, 255, 255, 0.96) 42%);
    }}
    h1, h2, h3 {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      letter-spacing: -0.02em;
    }}
    h1 {{ font-size: 34px; margin-bottom: 8px; }}
    .subtitle {{ color: var(--muted); line-height: 1.5; }}
    .hero-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.08);
      color: var(--accent);
      font-weight: 700;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric-card {{ padding: 16px; }}
    .metric-title {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .metric-value {{ margin-top: 8px; font-size: 28px; font-weight: 800; }}
    .card-subtitle {{ margin-top: 6px; font-size: 13px; color: var(--muted); }}
    .positive .metric-value {{ color: var(--positive); }}
    .negative .metric-value {{ color: var(--negative); }}
    .panel {{ padding: 18px; margin-bottom: 18px; }}
    .panel h2 {{ font-size: 22px; margin-bottom: 10px; }}
    .panel-subtitle {{ color: var(--muted); margin-bottom: 14px; line-height: 1.4; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: var(--panel);
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid #ebe5da;
      padding: 10px 10px;
      vertical-align: top;
      text-align: left;
      white-space: nowrap;
    }}
    .data-table th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--panel-alt);
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .table-scroll {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
      max-height: 700px;
    }}
    .num {{ text-align: right !important; font-variant-numeric: tabular-nums; }}
    .empty-state {{
      padding: 20px;
      color: var(--muted);
      font-style: italic;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .toolbar input, .toolbar select {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      background: var(--panel);
      min-width: 240px;
    }}
    .tier-badge {{
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .tier-tiera {{ background: var(--positive-soft); color: var(--positive); }}
    .tier-tierb {{ background: #dbeafe; color: #1d4ed8; }}
    .tier-tierc {{ background: var(--warning-soft); color: var(--warning); }}
    .tier-watch {{ background: #e5e7eb; color: #374151; }}
    .tier-mixed {{ background: #fce7f3; color: #be185d; }}
    .tier-avoid {{ background: var(--negative-soft); color: var(--negative); }}
    .bar-wrap {{
      width: 120px;
      height: 10px;
      background: #ece7dc;
      border-radius: 999px;
      overflow: hidden;
      display: inline-block;
      vertical-align: middle;
      margin-right: 8px;
    }}
    .bar {{
      height: 100%;
      border-radius: inherit;
    }}
    .bar.buy {{ background: var(--positive); }}
    .bar.sell {{ background: var(--negative); }}
    .bar-label {{ color: var(--muted); font-size: 12px; }}
    .section-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }}
    @media (max-width: 1200px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    }}
    @media (max-width: 900px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .toolbar input, .toolbar select {{ min-width: 100%; }}
    }}
    @media (max-width: 560px) {{
      .metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-card">
        <h1>Technical Agent Dashboard</h1>
        <div class="subtitle">
          A compact view across agent summaries, family rollups, consensus output, and the shortlist screen.
          Use the shortlist filter to inspect the 700-ticker universe without reading raw CSVs.
        </div>
        <div class="hero-meta">
          <span class="pill">Generated {html.escape(context["generated_at"])}</span>
          <span class="pill">{metrics["shortlist_rows"]} shortlist rows</span>
          <span class="pill">{metrics["tier_a_rows"]} Tier A picks</span>
        </div>
      </div>
      <div class="hero-card">
        <h3>Interpretation</h3>
        <div class="subtitle" style="margin-top: 10px;">
          The shortlist is tiered, not score-blended. That keeps the output readable when many agents and many tickers are involved.
          Families are grouped as trend, mean reversion, volume confirmation, and breakout.
        </div>
      </div>
    </section>

    <section class="metrics">
      {cards}
    </section>

    <div class="section-grid">
      <section class="panel">
        <h2>Agent Summary</h2>
        <div class="panel-subtitle">Per-agent performance and current output profile.</div>
        {summary_html}
      </section>

      <section class="panel">
        <h2>Family Summary</h2>
        <div class="panel-subtitle">Grouped by strategy family with buy/sell mix and strongest stock/agent signals.</div>
        {family_html}
      </section>

      <section class="panel">
        <h2>Shortlist</h2>
        <div class="panel-subtitle">Filter by tier or search a ticker, agent, or family. This is the main screen for selecting candidates.</div>
        {shortlist_html}
      </section>

      <section class="panel">
        <h2>Consensus</h2>
        <div class="panel-subtitle">Grouped stock consensus across agents.</div>
        {consensus_html}
      </section>
    </div>
  </main>
  <script>
    const search = document.getElementById("shortlist-search");
    const tierFilter = document.getElementById("tier-filter");
    const table = document.getElementById("shortlist-table");

    function applyFilters() {{
      if (!table) return;
      const query = (search && search.value ? search.value.trim().toLowerCase() : "");
      const tier = tierFilter ? tierFilter.value : "";
      const rows = table.querySelectorAll("tbody tr");
      rows.forEach((row) => {{
        const rowTier = row.getAttribute("data-tier") || "";
        const haystack = row.getAttribute("data-search") || "";
        const tierOk = !tier || rowTier === tier;
        const searchOk = !query || haystack.includes(query);
        row.style.display = tierOk && searchOk ? "" : "none";
      }});
    }}

    if (search) search.addEventListener("input", applyFilters);
    if (tierFilter) tierFilter.addEventListener("change", applyFilters);
    applyFilters();
  </script>
</body>
</html>"""


def build_dashboard(args):
    summary_df = read_csv_or_empty(args.summary)
    recommendations_df = read_csv_or_empty(args.recommendations)
    consensus_df = read_csv_or_empty(args.consensus)
    family_df = read_csv_or_empty(args.family_summary)
    shortlist_df = read_csv_or_empty(args.shortlist)

    context = build_dashboard_context(summary_df, family_df, consensus_df, shortlist_df, recommendations_df)
    html_text = render_html(context)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a self-contained technical agent dashboard HTML report")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY, help="Per-agent summary CSV")
    parser.add_argument("--recommendations", default=DEFAULT_RECOMMENDATIONS, help="Combined recommendations CSV")
    parser.add_argument("--consensus", default=DEFAULT_CONSENSUS, help="Consensus CSV")
    parser.add_argument("--family-summary", default=DEFAULT_FAMILY_SUMMARY, help="Family summary CSV")
    parser.add_argument("--shortlist", default=DEFAULT_SHORTLIST, help="Shortlist CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="HTML output path")
    args = parser.parse_args(argv)

    output_path = build_dashboard(args)
    print(f"Wrote dashboard to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    main()
