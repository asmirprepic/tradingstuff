import unittest

import pandas as pd

from scripts.technical_dashboard import build_dashboard_context, render_html


class TechnicalDashboardTests(unittest.TestCase):
    def test_dashboard_context_sorts_shortlist_and_exposes_metrics(self):
        summary_df = pd.DataFrame(
            [
                {"Agent": "momentum", "Algorithm": "Momentum", "Stocks": 2, "AvgStrategyReturnPct": 12.0, "MedianStrategyReturnPct": 10.0, "AvgBuyHoldReturnPct": 8.0, "ProfitableStocks": 2, "AvgEntries": 5.0, "AvgScore": 1.2, "TopPick": "MSFT"},
                {"Agent": "rsi", "Algorithm": "RSI", "Stocks": 2, "AvgStrategyReturnPct": -4.0, "MedianStrategyReturnPct": -5.0, "AvgBuyHoldReturnPct": 8.0, "ProfitableStocks": 1, "AvgEntries": 3.0, "AvgScore": -0.7, "TopPick": "AAPL"},
            ]
        )
        family_df = pd.DataFrame(
            [
                {"Family": "trend", "AgentCount": 1, "Agents": "momentum", "StocksCovered": 2, "TotalSignals": 2, "BuySignals": 2, "SellSignals": 0, "HoldSignals": 0, "NetSignals": 2, "BuySignalPct": 1.0, "SellSignalPct": 0.0, "AvgAgentStrategyReturnPct": 12.0, "TopAgentByReturn": "momentum", "TopBuyStock": "MSFT"},
            ]
        )
        consensus_df = pd.DataFrame(
            [
                {"Stock": "MSFT", "BuyCount": 2, "SellCount": 0, "HoldCount": 1, "MeanScore": 1.0, "BestAgent": "momentum", "BestAgentScore": 1.5, "SupportingAgents": "momentum, rsi"},
                {"Stock": "AAPL", "BuyCount": 0, "SellCount": 2, "HoldCount": 1, "MeanScore": -1.0, "BestAgent": "rsi", "BestAgentScore": -0.5, "SupportingAgents": "momentum, rsi"},
            ]
        )
        shortlist_df = pd.DataFrame(
            [
                {"Stock": "AAPL", "ShortlistTier": "Avoid", "ConsensusRecommendation": "Sell", "TotalAgents": 2, "BuyCount": 0, "SellCount": 2, "HoldCount": 0, "NetBias": -2, "ConflictCount": 0, "BuySupportPct": 0.0, "SellSupportPct": 1.0, "TrendBuyCount": 0, "MeanReversionBuyCount": 0, "VolumeBuyCount": 0, "BreakoutBuyCount": 0, "BuyFamilyBreadth": 0, "SupportFamilies": "", "MeanScore": -1.2, "BuyMeanScore": float("nan"), "MeanAgentRankPct": 0.2, "BuyAgentRankPct": 0.0, "BestAgent": "rsi", "BestAgentScore": -0.5, "BestBuyAgent": "", "BestBuyScore": float("nan"), "SupportingAgents": "", "OpposingAgents": "momentum, rsi"},
                {"Stock": "MSFT", "ShortlistTier": "TierA", "ConsensusRecommendation": "Buy", "TotalAgents": 2, "BuyCount": 2, "SellCount": 0, "HoldCount": 0, "NetBias": 2, "ConflictCount": 0, "BuySupportPct": 1.0, "SellSupportPct": 0.0, "TrendBuyCount": 2, "MeanReversionBuyCount": 0, "VolumeBuyCount": 1, "BreakoutBuyCount": 0, "BuyFamilyBreadth": 2, "SupportFamilies": "trend, volume_confirmation", "MeanScore": 1.5, "BuyMeanScore": 1.6, "MeanAgentRankPct": 0.9, "BuyAgentRankPct": 0.9, "BestAgent": "momentum", "BestAgentScore": 1.5, "BestBuyAgent": "momentum", "BestBuyScore": 1.5, "SupportingAgents": "momentum, rsi", "OpposingAgents": ""},
            ]
        )

        context = build_dashboard_context(summary_df, family_df, consensus_df, shortlist_df, pd.DataFrame())

        self.assertEqual(context["metrics"]["agents"], 2)
        self.assertEqual(context["metrics"]["tier_a_rows"], 1)
        self.assertEqual(context["shortlist"].iloc[0]["Stock"], "MSFT")

        html = render_html(context)
        self.assertIn("Technical Agent Dashboard", html)
        self.assertIn("Agent Summary", html)
        self.assertIn("Family Summary", html)
        self.assertIn("Shortlist", html)
        self.assertIn("MSFT", html)
        self.assertIn("Tier A", html)


if __name__ == "__main__":
    unittest.main()
