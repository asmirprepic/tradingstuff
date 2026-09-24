import unittest
from io import StringIO
from unittest.mock import Mock, patch

import pandas as pd
from rich.console import Console

from scripts.trading_cli import (
    build_parser,
    build_runner_args,
    choose_ticker_universe,
    dispatch,
    main,
    render_selector,
    render_terminal_dashboard,
    select_items,
    ticker_groups,
    terminal_dashboard,
)


class TradingCLITests(unittest.TestCase):
    def test_help_text_formats(self):
        help_text = build_parser().format_help()

        self.assertIn("quality", help_text)
        self.assertIn("dashboard", help_text)

    def test_dispatch_forwards_arguments_to_runner(self):
        module = Mock()
        module.main.return_value = {"ok": True}
        with patch("scripts.trading_cli.import_module", return_value=module) as importer:
            result = dispatch("ml", ["--agents", "qda", "--tickers", "AAA"])

        importer.assert_called_once_with("scripts.run_ml_agents")
        module.main.assert_called_once_with(["--agents", "qda", "--tickers", "AAA"])
        self.assertEqual(result, {"ok": True})

    def test_alias_and_separator_are_supported(self):
        module = Mock()
        with patch("scripts.trading_cli.import_module", return_value=module) as importer:
            dispatch("tech", ["--", "--agents", "momentum"])

        importer.assert_called_once_with("scripts.run_technical_agents")
        module.main.assert_called_once_with(["--agents", "momentum"])

    def test_main_without_command_prints_help(self):
        with patch.object(build_parser().__class__, "print_help") as print_help:
            result = main([])

        self.assertEqual(result, 0)
        print_help.assert_called_once()

    def test_unknown_command_fails_cleanly(self):
        with self.assertRaisesRegex(SystemExit, "Unknown command"):
            dispatch("unknown", [])

    def test_build_runner_args_supports_ml_and_ticker_file(self):
        self.assertEqual(
            build_runner_args("ml", "AAA,MSFT", True, agents="qda", extra="--mode live"),
            ["--tickers", "AAA,MSFT", "--use-synthetic", "--agents", "qda", "--mode", "live"],
        )
        self.assertEqual(
            build_runner_args("technical", "approved.csv", False, "300", "momentum,rsi"),
            ["--tickers-file", "approved.csv", "--lookback-days", "300", "--agents", "momentum,rsi"],
        )

    def test_terminal_dashboard_render_contains_tools_and_controls(self):
        console = Console(record=True, width=100, color_system=None)
        console.print(render_terminal_dashboard(selected=2, status="READY"))
        rendered = console.export_text()

        self.assertIn("MARKET LAB", rendered)
        self.assertIn("DATA QUALITY", rendered)
        self.assertIn("MACHINE LEARNING", rendered)
        self.assertIn("ARROWS", rendered)

    def test_terminal_dashboard_quits_from_keyboard(self):
        console = Console(file=StringIO(), force_terminal=True, width=100)

        self.assertEqual(terminal_dashboard(console=console, key_reader=lambda: "Q"), 0)

    def test_scroll_selector_toggles_and_accepts_items(self):
        keys = iter(["DOWN", "SPACE", "ENTER"])
        console = Console(file=StringIO(), force_terminal=True, width=100)

        selected = select_items(
            "AGENTS",
            ["momentum", "rsi", "vwap"],
            initially_selected=["momentum", "rsi"],
            console=console,
            key_reader=lambda: next(keys),
        )

        self.assertEqual(selected, ["momentum"])

    def test_scroll_selector_supports_back_and_renders_controls(self):
        console = Console(record=True, width=100, color_system=None)
        console.print(render_selector("TICKERS", ["AAA", "BBB"], {0}, 0, 0))
        rendered = console.export_text()
        self.assertIn("SPACE toggle", rendered)
        self.assertIn("/ search", rendered)
        self.assertIn("[x] AAA", rendered)
        quiet_console = Console(file=StringIO(), force_terminal=True, width=100)
        self.assertIsNone(select_items("TICKERS", ["AAA"], console=quiet_console, key_reader=lambda: "B"))

    def test_scroll_selector_search_toggles_matching_original_item(self):
        keys = iter(["/", "SPACE", "ENTER"])
        console = Console(file=StringIO(), force_terminal=True, width=100)

        selected = select_items(
            "AGENTS",
            ["momentum", "rsi", "vwap"],
            initially_selected=["momentum", "rsi", "vwap"],
            console=console,
            key_reader=lambda: next(keys),
            search_input_fn=lambda _: "rsi",
        )

        self.assertEqual(selected, ["momentum", "vwap"])

    def test_render_selector_shows_empty_search_result(self):
        console = Console(record=True, width=100, color_system=None)
        console.print(render_selector("TICKERS", ["AAA", "BBB"], set(), 0, 0, query="ZZZ"))
        rendered = console.export_text()

        self.assertIn("No matches", rendered)
        self.assertIn("Filter: ZZZ", rendered)

    def test_ticker_groups_extracts_quality_and_shortlist_groups(self):
        frame = pd.DataFrame(
            [
                {"Stock": "AAA", "Status": "Pass", "ShortlistTier": "TierA"},
                {"Stock": "BBB", "Status": "Warning", "ShortlistTier": "TierC"},
            ]
        )

        groups = ticker_groups(frame)

        self.assertEqual(groups["Status"]["Pass"], ["AAA"])
        self.assertEqual(groups["ShortlistTier"]["TierC"], ["BBB"])

    def test_large_universe_can_be_used_without_browsing_tickers(self):
        console = Console(file=StringIO(), force_terminal=True, width=100)
        with patch("scripts.trading_cli._ticker_choices", return_value=[f"T{i}" for i in range(700)]), patch(
            "scripts.trading_cli._groups_for_universe", return_value={}
        ):
            result = choose_ticker_universe(
                "tickers.txt",
                console=console,
                key_reader=lambda: "ENTER",
            )

        self.assertEqual(result, "tickers.txt")

    def test_quality_csv_defaults_to_pass_group(self):
        keys = iter(["DOWN", "ENTER", "ENTER"])
        console = Console(file=StringIO(), force_terminal=True, width=100)
        groups = {"Status": {"Pass": ["AAA"], "Warning": ["BBB"]}}
        tickers = ["AAA", "BBB"] + [f"T{i}" for i in range(29)]
        with patch("scripts.trading_cli._ticker_choices", return_value=tickers), patch(
            "scripts.trading_cli._groups_for_universe", return_value=groups
        ):
            result = choose_ticker_universe(
                "quality.csv",
                console=console,
                key_reader=lambda: next(keys),
            )

        self.assertEqual(result, "AAA")


if __name__ == "__main__":
    unittest.main()
