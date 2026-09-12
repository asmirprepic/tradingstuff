import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd

from agents.ml_based.anomaly.autoencoder_agent import AutoencoderAgent
from agents.ml_based.classical.gaussian_process_agent import GaussianProcessAgent
from agents.ml_based.classical.knn_agent import KNNAgent
from agents.ml_based.classical.logistic_reg_agent import LRAgent
from agents.ml_based.classical.naive_bayes_agent import NaiveBayesAgent
from agents.ml_based.classical.svm_agent import SVMAgent
from agents.ml_based.clustering.clustering_agent import ClusteringFilteredKNNAgent
from agents.ml_based.deep_learning.cnn_agent import CNNAgent
from agents.ml_based.deep_learning.lstm_agent import LSTMAgent
from agents.ml_based.deep_learning.lstm_attention_agent import LSTMAttentionAgent
from agents.ml_based.deep_learning.nn_classification_agent import DenseNNAgent
from agents.ml_based.deep_learning.tcn_agent import TCNAgent
from agents.ml_based.deep_learning.transformer_agent import TransformerAgent
from agents.ml_based.regime.hmm_based_agent import HMMRegimeAgent
from agents.utils.evaluate import evaluations_from_agent
from scripts.run_technical_agents import (
    fetch_tickers_by_region,
    load_price_df,
    make_synthetic_ohlcv,
    read_tickers_file,
    resolve_output_path,
)


AGENT_ORDER = [
    "logistic_reg",
    "naive_bayes",
    "svm",
    "knn",
    "gaussian_process",
    "clustering_knn",
    "hmm_regime",
    "autoencoder",
    "dense_nn",
    "cnn",
    "lstm",
    "lstm_attention",
    "tcn",
    "transformer",
]

AGENT_GROUPS = {
    "classical": ["logistic_reg", "naive_bayes", "svm", "knn", "gaussian_process"],
    "lightweight": ["logistic_reg", "naive_bayes", "svm", "knn"],
    "specialized": ["clustering_knn", "hmm_regime", "autoencoder"],
    "deep": ["dense_nn", "cnn", "lstm", "lstm_attention", "tcn", "transformer"],
    "all": AGENT_ORDER,
}


def parse_agent_names(agent_arg):
    requested = [name.strip().lower() for name in agent_arg.split(",") if name.strip()]
    if not requested:
        return list(AGENT_GROUPS["lightweight"])

    selected = []
    for name in requested:
        expanded = AGENT_GROUPS.get(name, [name])
        for agent_name in expanded:
            if agent_name not in selected:
                selected.append(agent_name)

    unknown = sorted(set(selected) - set(AGENT_ORDER))
    if unknown:
        raise SystemExit(f"Unknown ML agent(s): {', '.join(unknown)}. Available: {', '.join(AGENT_ORDER)}")
    return selected


def tickers_from_shortlist(path, tiers):
    shortlist_path = Path(path)
    if not shortlist_path.exists():
        raise SystemExit(f"Technical shortlist not found: {path}")
    shortlist = pd.read_csv(shortlist_path)
    if "Stock" not in shortlist.columns:
        raise SystemExit("Technical shortlist must contain a Stock column.")
    if tiers and "ShortlistTier" in shortlist.columns:
        shortlist = shortlist[shortlist["ShortlistTier"].isin(tiers)]
    return shortlist["Stock"].dropna().astype(str).drop_duplicates().tolist()


def build_agent(agent_name, price_df, args):
    if agent_name == "logistic_reg":
        return LRAgent(price_df)
    if agent_name == "naive_bayes":
        return NaiveBayesAgent(price_df)
    if agent_name == "svm":
        return SVMAgent(price_df)
    if agent_name == "knn":
        return KNNAgent(price_df, proba_threshold=args.proba_threshold)
    if agent_name == "gaussian_process":
        return GaussianProcessAgent(
            price_df,
            max_samples=args.gaussian_max_samples,
            proba_threshold=args.proba_threshold,
        )
    if agent_name == "clustering_knn":
        return ClusteringFilteredKNNAgent(price_df)
    if agent_name == "hmm_regime":
        return HMMRegimeAgent(price_df)
    if agent_name == "autoencoder":
        return AutoencoderAgent(price_df, epochs=args.epochs, verbose=args.verbose)
    if agent_name == "dense_nn":
        return DenseNNAgent(price_df)
    if agent_name == "cnn":
        return CNNAgent(price_df, epochs=args.epochs, verbose=args.verbose)
    if agent_name == "lstm":
        return LSTMAgent(price_df, epochs=args.epochs, verbose=args.verbose)
    if agent_name == "lstm_attention":
        return LSTMAttentionAgent(price_df, epochs=args.epochs, verbose=args.verbose)
    if agent_name == "tcn":
        return TCNAgent(price_df, epochs=args.epochs, verbose=args.verbose)
    if agent_name == "transformer":
        return TransformerAgent(price_df, epochs=args.epochs, verbose=args.verbose)
    raise KeyError(f"Unhandled ML agent: {agent_name}")


def safe_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unknown"


def artifact_path(root, agent_name, stock):
    return Path(root) / safe_component(agent_name) / safe_component(stock)


def run_agent(agent_name, agent, stocks, args):
    failures = []
    artifact_rows = []
    agent.signal_data = {}

    for stock in stocks:
        model_dir = artifact_path(args.artifact_dir, agent_name, stock)
        loaded = False
        if args.reuse_artifacts and (model_dir / "manifest.json").exists():
            try:
                agent.load_model_artifact(model_dir, trusted=True)
                signals = agent.predict_signals(stock, mode=args.mode)
                agent.signal_data[stock] = signals
                loaded = True
            except Exception as exc:
                artifact_rows.append({"Agent": agent_name, "Stock": stock, "Status": "load_failed", "Path": str(model_dir), "Error": str(exc)})

        if not loaded:
            try:
                signals = agent.generate_signal_strategy(stock, mode=args.mode)
                agent.signal_data[stock] = signals
                if args.save_artifacts:
                    manifest_path = agent.save_model_artifact(stock, model_dir)
                    artifact_rows.append({"Agent": agent_name, "Stock": stock, "Status": "saved", "Path": str(manifest_path), "Error": None})
            except Exception as exc:
                failures.append({"agent": agent_name, "stock": stock, "error": str(exc)})
                continue
        elif loaded:
            artifact_rows.append({"Agent": agent_name, "Stock": stock, "Status": "loaded", "Path": str(model_dir / "manifest.json"), "Error": None})

    agent.calculate_returns()
    return failures, artifact_rows


def build_agent_summary(agent_name, agent, recs, failures):
    training = agent.training_summary() if agent.training_info or getattr(agent, "train_data", {}) else pd.DataFrame()
    returns = pd.DataFrame(agent.returns_data).T if agent.returns_data else pd.DataFrame()
    strategy_column = f"{agent.algorithm_name}_return"
    return {
        "Agent": agent_name,
        "Algorithm": agent.algorithm_name,
        "SuccessfulStocks": len(agent.signal_data),
        "FailedStocks": failures,
        "BuyCount": int((recs["Recommendation"] == "Buy").sum()) if not recs.empty else 0,
        "SellCount": int((recs["Recommendation"] == "Sell").sum()) if not recs.empty else 0,
        "HoldCount": int((recs["Recommendation"] == "Hold").sum()) if not recs.empty else 0,
        "AvgScore": float(recs["Score"].mean()) if not recs.empty and recs["Score"].notna().any() else float("nan"),
        "AvgStrategyReturnPct": float(returns[strategy_column].mean()) if strategy_column in returns else float("nan"),
        "AvgAccuracy": float(training["Accuracy"].mean()) if "Accuracy" in training else float("nan"),
        "AvgPrecision": float(training["Precision"].mean()) if "Precision" in training else float("nan"),
        "AvgRecall": float(training["Recall"].mean()) if "Recall" in training else float("nan"),
        "AvgF1Score": float(training["F1Score"].mean()) if "F1Score" in training else float("nan"),
    }


def write_dataframe(df, path):
    if not path or df.empty:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path.resolve()


def write_manifest(path, run_id, args, tickers, agents, outputs, row_counts, failures):
    manifest_path = Path(path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_type": "ml_agents",
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data": {
            "source": "synthetic" if args.use_synthetic else "market",
            "tickers": tickers,
            "ticker_count": len(tickers),
            "technical_shortlist": args.technical_shortlist,
            "shortlist_tiers": args.shortlist_tiers,
            "start": args.start,
            "end": args.end,
            "lookback_days": args.lookback_days,
            "interval": args.interval,
        },
        "configuration": {
            "agents": agents,
            "mode": args.mode,
            "persistence": args.persistence,
            "proba_threshold": args.proba_threshold,
            "epochs": args.epochs,
            "reuse_artifacts": args.reuse_artifacts,
            "save_artifacts": args.save_artifacts,
            "artifact_dir": str(Path(args.artifact_dir).resolve()),
        },
        "outputs": {name: str(value) if value else None for name, value in outputs.items()},
        "row_counts": row_counts,
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run selected ML trading agents")
    parser.add_argument("--agents", default="lightweight", help="Comma-separated agents or groups: lightweight, classical, specialized, deep, all")
    parser.add_argument("--list-agents", action="store_true")
    parser.add_argument("--tickers", default="AAPL,MSFT,NVDA")
    parser.add_argument("--tickers-file", default=None)
    parser.add_argument("--fetch-tickers", default=None)
    parser.add_argument("--technical-shortlist", default=None)
    parser.add_argument("--shortlist-tiers", default="TierA,TierB,TierC")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback-days", type=int, default=260)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--synthetic-periods", type=int, default=260)
    parser.add_argument("--mode", choices=("backtest", "live"), default="backtest")
    parser.add_argument("--persistence", type=int, default=1)
    parser.add_argument("--top-n-per-agent", type=int, default=None)
    parser.add_argument("--proba-threshold", type=float, default=0.55)
    parser.add_argument("--gaussian-max-samples", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--reuse-artifacts", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--artifact-dir", default="outputs/models")
    parser.add_argument("--summary-output", default="ml_agent_summary.csv")
    parser.add_argument("--training-output", default="ml_agent_training_summary.csv")
    parser.add_argument("--recommendations-output", default="ml_agent_recommendations.csv")
    parser.add_argument("--artifacts-output", default="ml_agent_artifacts.csv")
    parser.add_argument("--manifest-output", default="ml_agent_run_manifest.json")
    parser.add_argument("--timestamp-output", action="store_true")
    args = parser.parse_args(argv)

    if args.list_agents:
        print("\n".join(AGENT_ORDER))
        return 0

    selected_agents = parse_agent_names(args.agents)
    tiers = [tier.strip() for tier in args.shortlist_tiers.split(",") if tier.strip()]
    args.shortlist_tiers = tiers
    if args.technical_shortlist:
        tickers = tickers_from_shortlist(args.technical_shortlist, tiers)
    elif args.fetch_tickers:
        tickers = fetch_tickers_by_region(args.fetch_tickers)
    elif args.tickers_file:
        tickers = read_tickers_file(args.tickers_file)
    else:
        tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]
    if not tickers:
        raise SystemExit("No tickers resolved.")

    price_df = load_price_df(tickers, args)
    summary_rows = []
    training_frames = []
    recommendation_frames = []
    artifact_rows = []
    failures = []

    for agent_name in selected_agents:
        print(f"Running {agent_name}...")
        try:
            agent = build_agent(agent_name, price_df, args)
            agent_failures, agent_artifacts = run_agent(agent_name, agent, tickers, args)
            failures.extend(agent_failures)
            artifact_rows.extend(agent_artifacts)
            recs = evaluations_from_agent(agent, persistence=args.persistence, top_n=args.top_n_per_agent)
            if not recs.empty:
                recs.insert(0, "Agent", agent_name)
                recommendation_frames.append(recs)
            if agent.training_info or getattr(agent, "train_data", {}):
                training = agent.training_summary()
                training.insert(0, "RunnerAgent", agent_name)
                training_frames.append(training)
            summary_rows.append(build_agent_summary(agent_name, agent, recs, len(agent_failures)))
        except Exception as exc:
            failures.append({"agent": agent_name, "stock": None, "error": str(exc)})

    summary_df = pd.DataFrame(summary_rows)
    training_df = pd.concat(training_frames, ignore_index=True) if training_frames else pd.DataFrame()
    recommendations_df = pd.concat(recommendation_frames, ignore_index=True) if recommendation_frames else pd.DataFrame()
    artifacts_df = pd.DataFrame(artifact_rows)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_args = {
        "summary": args.summary_output,
        "training": args.training_output,
        "recommendations": args.recommendations_output,
        "artifacts": args.artifacts_output,
    }
    outputs = {
        name: write_dataframe(df, resolve_output_path(path, args.timestamp_output, run_id))
        for (name, path), df in zip(
            output_args.items(),
            (summary_df, training_df, recommendations_df, artifacts_df),
        )
    }
    manifest_out = resolve_output_path(args.manifest_output, args.timestamp_output, run_id)
    manifest_path = write_manifest(
        manifest_out,
        run_id,
        args,
        tickers,
        selected_agents,
        outputs,
        {
            "summary": len(summary_df),
            "training": len(training_df),
            "recommendations": len(recommendations_df),
            "artifacts": len(artifacts_df),
        },
        failures,
    )

    print("\nML Agent Summary:")
    print(summary_df if not summary_df.empty else "No summaries produced.")
    for name, path in outputs.items():
        if path:
            print(f"Wrote {name} to {path}")
    print(f"Wrote run manifest to {manifest_path}")
    if failures:
        print(f"Failures: {len(failures)} (see manifest)")

    return {
        "summary": summary_df,
        "training": training_df,
        "recommendations": recommendations_df,
        "artifacts": artifacts_df,
        "failures": failures,
        "manifest_path": manifest_path,
    }


if __name__ == "__main__":
    main()
