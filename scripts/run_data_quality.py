import argparse
import json
from datetime import datetime
from pathlib import Path

from agents.data_quality_agent import DataQualityAgent
from scripts.run_technical_agents import (
    fetch_tickers_by_region,
    load_price_df,
    read_tickers_file,
    resolve_output_path,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit OHLCV data before running trading agents")
    parser.add_argument("--tickers", default="AAPL,MSFT,NVDA")
    parser.add_argument("--tickers-file", default=None)
    parser.add_argument("--fetch-tickers", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback-days", type=int, default=260)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--synthetic-periods", type=int, default=260)
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--recent-window", type=int, default=20)
    parser.add_argument("--stale-run-limit", type=int, default=5)
    parser.add_argument("--extreme-return", type=float, default=0.35)
    parser.add_argument("--include-warnings", action="store_true", help="Include Warning stocks in approved output")
    parser.add_argument("--output", default="data_quality_report.csv")
    parser.add_argument("--approved-output", default="data_quality_approved_tickers.csv")
    parser.add_argument("--manifest-output", default="data_quality_manifest.json")
    parser.add_argument("--timestamp-output", action="store_true")
    args = parser.parse_args(argv)

    if args.fetch_tickers:
        tickers = fetch_tickers_by_region(args.fetch_tickers)
    elif args.tickers_file:
        tickers = read_tickers_file(args.tickers_file)
    else:
        tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]
    if not tickers:
        raise SystemExit("No tickers resolved.")

    data = load_price_df(tickers, args)
    if data.empty:
        raise SystemExit("No price data available.")
    scanner = DataQualityAgent(
        data,
        min_history=args.min_history,
        recent_window=args.recent_window,
        stale_run_limit=args.stale_run_limit,
        extreme_return=args.extreme_return,
    )
    report = scanner.run()
    accepted_statuses = ["Pass", "Warning"] if args.include_warnings else ["Pass"]
    approved = report.loc[report["Status"].isin(accepted_statuses), ["Stock", "Status", "QualityScore"]]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = resolve_output_path(args.output, args.timestamp_output, run_id)
    approved_path = resolve_output_path(args.approved_output, args.timestamp_output, run_id)
    manifest_path = resolve_output_path(args.manifest_output, args.timestamp_output, run_id)
    report.to_csv(report_path, index=False)
    approved.to_csv(approved_path, index=False)
    manifest = {
        "schema_version": 1,
        "run_type": "data_quality",
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "configuration": {
            "min_history": args.min_history,
            "recent_window": args.recent_window,
            "stale_run_limit": args.stale_run_limit,
            "extreme_return": args.extreme_return,
            "include_warnings": args.include_warnings,
        },
        "counts": report["Status"].value_counts().to_dict(),
        "outputs": {
            "report": str(Path(report_path).resolve()),
            "approved_tickers": str(Path(approved_path).resolve()),
        },
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nData Quality Summary:")
    print(report["Status"].value_counts().reindex(["Pass", "Warning", "Reject"], fill_value=0))
    print(f"Wrote report to {Path(report_path).resolve()}")
    print(f"Wrote approved tickers to {Path(approved_path).resolve()}")
    print(f"Wrote run manifest to {Path(manifest_path).resolve()}")
    return {"report": report, "approved": approved, "manifest": manifest}


if __name__ == "__main__":
    main()
