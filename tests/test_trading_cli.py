import unittest
from unittest.mock import Mock, patch

from scripts.trading_cli import build_parser, dispatch, main


class TradingCLITests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
