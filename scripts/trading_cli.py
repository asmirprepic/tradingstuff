import argparse
from importlib import import_module


COMMANDS = {
    "quality": ("scripts.run_data_quality", "Audit OHLCV data and create an approved ticker file."),
    "technical": ("scripts.run_technical_agents", "Run technical agents and build shortlist CSVs."),
    "ml": ("scripts.run_ml_agents", "Train or load selected ML agents."),
    "dashboard": ("scripts.technical_dashboard", "Build the technical HTML dashboard."),
}

ALIASES = {
    "tech": "technical",
    "machine-learning": "ml",
    "data-quality": "quality",
}


def build_parser():
    command_help = "\n".join(
        f"  {name:<10} {description}" for name, (_, description) in COMMANDS.items()
    )
    parser = argparse.ArgumentParser(
        prog="trading-cli",
        description="Small command line entry point for the trading workflow.",
        epilog=(
            "Commands:\n"
            f"{command_help}\n\n"
            "Examples:\n"
            "  python -m scripts.trading_cli quality --use-synthetic --tickers AAPL,MSFT\n"
            "  python -m scripts.trading_cli technical --tickers-file data_quality_approved_tickers.csv --agents momentum,rsi\n"
            "  python -m scripts.trading_cli ml --tickers AAPL,MSFT --agents qda,spline_logistic\n"
            "  python -m scripts.trading_cli dashboard --latest-manifest-dir .",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", help="quality, technical, ml, or dashboard")
    return parser


def dispatch(command, command_args):
    command = ALIASES.get(command, command)
    if command not in COMMANDS:
        available = ", ".join(COMMANDS)
        raise SystemExit(f"Unknown command '{command}'. Available: {available}")
    if command_args[:1] == ["--"]:
        command_args = command_args[1:]
    module = import_module(COMMANDS[command][0])
    return module.main(command_args)


def main(argv=None):
    parser = build_parser()
    args, command_args = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    return dispatch(args.command.lower(), command_args)


if __name__ == "__main__":
    main()
