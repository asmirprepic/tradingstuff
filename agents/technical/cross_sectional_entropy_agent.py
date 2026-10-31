import numpy as np
import pandas as pd

from agents.base_agents.trading_agent import TradingAgent
from utilities.return_entropy import calculate_return_entropy


class CrossSectionalEntropyAgent(TradingAgent):
    """Test and trade leader continuation conditional on return-contribution entropy."""

    def __init__(
        self,
        data,
        leader_contribution_share=0.60,
        maximum_leaders=10,
        minimum_stocks=10,
        entropy_quantile=0.30,
        entropy_lookback=104,
        minimum_entropy_weeks=26,
        forward_horizon_weeks=1,
        calibration_weeks=104,
        minimum_calibration_events=8,
        require_positive_market_week=True,
        require_falling_entropy=True,
        decision_frequency="W-FRI",
        price_type="Close",
        auto_generate=True,
    ):
        super().__init__(data)
        self.algorithm_name = "CrossSectionalReturnEntropy"
        self.score_column = "EntropyLeaderScore"
        self.leader_contribution_share = float(leader_contribution_share)
        self.maximum_leaders = int(maximum_leaders)
        self.minimum_stocks = int(minimum_stocks)
        self.entropy_quantile = float(entropy_quantile)
        self.entropy_lookback = int(entropy_lookback)
        self.minimum_entropy_weeks = int(minimum_entropy_weeks)
        self.forward_horizon_weeks = int(forward_horizon_weeks)
        self.calibration_weeks = int(calibration_weeks)
        self.minimum_calibration_events = int(minimum_calibration_events)
        self.require_positive_market_week = bool(require_positive_market_week)
        self.require_falling_entropy = bool(require_falling_entropy)
        self.decision_frequency = decision_frequency
        self.price_type = price_type
        self._validate_parameters()

        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        self.entropy_history = pd.DataFrame()
        self.event_history = pd.DataFrame()
        self.score_matrix = pd.DataFrame()
        self.first_score_position = None
        self.first_tradable_position = None
        if auto_generate:
            self.run_all()

    def _validate_parameters(self):
        if not 0.0 < self.leader_contribution_share <= 1.0:
            raise ValueError("leader_contribution_share must be in (0, 1].")
        if self.maximum_leaders < 1:
            raise ValueError("maximum_leaders must be positive.")
        if self.minimum_stocks < 2:
            raise ValueError("minimum_stocks must be at least 2.")
        if not 0.0 < self.entropy_quantile < 1.0:
            raise ValueError("entropy_quantile must be in (0, 1).")
        positive = (
            self.entropy_lookback,
            self.minimum_entropy_weeks,
            self.forward_horizon_weeks,
            self.calibration_weeks,
            self.minimum_calibration_events,
        )
        if any(value < 1 for value in positive):
            raise ValueError("Entropy windows, horizons, and counts must be positive.")
        if self.minimum_entropy_weeks > self.entropy_lookback:
            raise ValueError("minimum_entropy_weeks cannot exceed entropy_lookback.")
        try:
            pd.tseries.frequencies.to_offset(self.decision_frequency)
        except (TypeError, ValueError) as exc:
            raise ValueError("decision_frequency must be a valid pandas frequency.") from exc

    def _decision_dates(self):
        indexed_dates = pd.Series(self.data.index, index=self.data.index)
        grouped = indexed_dates.resample(self.decision_frequency).last().dropna()
        completed = grouped.index <= self.data.index[-1]
        return pd.DatetimeIndex(grouped.loc[completed].tolist())

    @staticmethod
    def _forward_event(record, maturity_date, close):
        formation = close.loc[record["FormationDate"]]
        maturity = close.loc[maturity_date]
        stock_returns = maturity.div(formation).sub(1.0).replace([np.inf, -np.inf], np.nan)
        leaders = [stock for stock in record["Leaders"] if stock in stock_returns.index]
        leader_return = stock_returns.reindex(leaders).dropna().mean()
        universe_return = stock_returns.dropna().mean()
        return {
            **record,
            "MaturityDate": maturity_date,
            "LeaderForwardReturn": leader_return,
            "UniverseForwardReturn": universe_return,
            "LeaderExcessReturn": leader_return - universe_return,
            "LeaderContinued": bool(leader_return > universe_return),
        }

    def _continuation_statistics(self, completed):
        sample = completed[-self.calibration_weeks:]
        qualifying = []
        for row in sample:
            qualifies = row["LowEntropy"]
            if self.require_falling_entropy:
                qualifies &= row["EntropyFalling"]
            if self.require_positive_market_week:
                qualifies &= row["PositiveMarketWeek"]
            if qualifies and np.isfinite(row["LeaderExcessReturn"]):
                qualifying.append(row)
        if len(qualifying) < self.minimum_calibration_events:
            return np.nan, np.nan, len(qualifying)
        values = np.asarray([row["LeaderExcessReturn"] for row in qualifying], dtype=float)
        return float(values.mean()), float(np.mean(values > 0.0)), len(values)

    def _generate_all_signals(self):
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        index = close.index
        stocks = close.columns
        decision_dates = self._decision_dates()
        score_matrix = pd.DataFrame(np.nan, index=index, columns=stocks)
        share_matrix = pd.DataFrame(np.nan, index=index, columns=stocks)
        entropy_rows = []
        pending = []
        completed = []

        for decision_index in range(1, len(decision_dates)):
            date = decision_dates[decision_index]
            previous_date = decision_dates[decision_index - 1]
            still_pending = []
            for record in pending:
                if record["MaturityDecisionIndex"] <= decision_index:
                    completed.append(self._forward_event(record, date, close))
                else:
                    still_pending.append(record)
            pending = still_pending

            weekly_returns = close.loc[date].div(close.loc[previous_date]).sub(1.0)
            metrics = calculate_return_entropy(
                weekly_returns,
                leader_contribution_share=self.leader_contribution_share,
                maximum_leaders=self.maximum_leaders,
                minimum_stocks=self.minimum_stocks,
            )
            prior_entropy = pd.Series(
                [row["UpsideEntropy"] for row in entropy_rows], dtype=float
            ).dropna().iloc[-self.entropy_lookback:]
            threshold = (
                float(prior_entropy.quantile(self.entropy_quantile))
                if len(prior_entropy) >= self.minimum_entropy_weeks
                else np.nan
            )
            previous_entropy = entropy_rows[-1]["UpsideEntropy"] if entropy_rows else np.nan
            entropy_change = metrics["UpsideEntropy"] - previous_entropy
            low_entropy = bool(
                np.isfinite(threshold)
                and np.isfinite(metrics["UpsideEntropy"])
                and metrics["UpsideEntropy"] <= threshold
            )
            falling = bool(np.isfinite(entropy_change) and entropy_change < 0.0)
            positive_market = bool(
                np.isfinite(metrics["MarketReturn"]) and metrics["MarketReturn"] > 0.0
            )
            continuation_edge, continuation_hit_rate, calibration_events = (
                self._continuation_statistics(completed)
            )
            score_matrix.loc[date] = 0.0
            if (
                self.first_tradable_position is None
                and np.isfinite(threshold)
            ):
                self.first_tradable_position = index.get_loc(date)
            eligible_state = low_entropy
            if self.require_falling_entropy:
                eligible_state &= falling
            if self.require_positive_market_week:
                eligible_state &= positive_market

            shares = metrics["UpsideShares"].reindex(stocks).fillna(0.0)
            share_matrix.loc[date] = shares
            if eligible_state and np.isfinite(continuation_edge) and continuation_edge > 0.0:
                leaders = metrics["Leaders"]
                score_matrix.loc[date, leaders] = (
                    shares.reindex(leaders) * continuation_edge
                )
                if self.first_score_position is None:
                    self.first_score_position = index.get_loc(date)

            entropy_rows.append({
                "Date": date,
                "MarketReturn": metrics["MarketReturn"],
                "PositiveBreadth": metrics["PositiveBreadth"],
                "UpsideEntropy": metrics["UpsideEntropy"],
                "DownsideEntropy": metrics["DownsideEntropy"],
                "EntropyChange": entropy_change,
                "EntropyThreshold": threshold,
                "LowEntropy": low_entropy,
                "EntropyFalling": falling,
                "PositiveMarketWeek": positive_market,
                "EffectiveContributors": metrics["EffectiveContributors"],
                "Top5Share": metrics["Top5Share"],
                "Leaders": metrics["Leaders"],
                "ContinuationEdge": continuation_edge,
                "ContinuationHitRate": continuation_hit_rate,
                "CalibrationEvents": calibration_events,
                "SignalActive": bool(score_matrix.loc[date].gt(0.0).any()),
            })
            maturity_index = decision_index + self.forward_horizon_weeks
            if maturity_index < len(decision_dates) and metrics["Leaders"]:
                pending.append({
                    "FormationDate": date,
                    "MaturityDecisionIndex": maturity_index,
                    "UpsideEntropy": metrics["UpsideEntropy"],
                    "EntropyChange": entropy_change,
                    "LowEntropy": low_entropy,
                    "EntropyFalling": falling,
                    "PositiveMarketWeek": positive_market,
                    "Leaders": metrics["Leaders"],
                    "HorizonWeeks": self.forward_horizon_weeks,
                })

        self.entropy_history = (
            pd.DataFrame(entropy_rows).set_index("Date")
            if entropy_rows else pd.DataFrame()
        )
        self.event_history = pd.DataFrame(completed)
        self.score_matrix = score_matrix
        held_scores = score_matrix.ffill().fillna(0.0)
        self.signal_data = {}
        for stock in stocks:
            signals = pd.DataFrame(index=index)
            signals["price"] = close[stock]
            signals["return"] = np.log(close[stock] / close[stock].shift(1))
            signals["WeeklyReturn"] = np.nan
            signals.loc[decision_dates[1:], "WeeklyReturn"] = [
                close.loc[decision_dates[position], stock]
                / close.loc[decision_dates[position - 1], stock] - 1.0
                for position in range(1, len(decision_dates))
            ]
            signals["PositiveContributionShare"] = share_matrix[stock]
            signals["EntropyLeaderScore"] = held_scores[stock]
            signals["SignalStrength"] = held_scores[stock]
            signals["Position"] = held_scores[stock].gt(0.0).astype(int)
            signals["Signal"] = np.sign(signals["Position"].diff().fillna(0)).astype(int)
            self.signal_data[stock] = signals

    def generate_signal_strategy(self, stock, mode="backtest"):
        if mode not in {"backtest", "live"}:
            raise ValueError("mode must be 'backtest' or 'live'.")
        if stock not in self.signal_data:
            raise KeyError(f"No signal data for stock={stock}")
        return self.signal_data[stock].tail(1) if mode == "live" else self.signal_data[stock]

    def run_all(self, mode="backtest"):
        self._generate_all_signals()
        self.calculate_returns()
        return self.signal_data

    def plot_entropy_history(self, ax=None):
        """Plot upside entropy, its historical threshold, and active regimes."""
        if self.entropy_history.empty:
            raise ValueError("No entropy history is available.")
        import matplotlib.pyplot as plt

        if ax is None:
            figure, ax = plt.subplots(figsize=(11, 5))
        else:
            figure = ax.figure
        history = self.entropy_history
        ax.plot(history.index, history["UpsideEntropy"], label="Upside entropy")
        ax.plot(
            history.index,
            history["EntropyThreshold"],
            linestyle="--",
            label="Low-entropy threshold",
        )
        active = history["SignalActive"]
        ax.scatter(
            history.index[active],
            history.loc[active, "UpsideEntropy"],
            marker="v",
            color="tab:green",
            label="Leader continuation active",
            zorder=3,
        )
        ax.set_ylabel("Normalized entropy")
        ax.set_title("Cross-sectional upside return entropy")
        ax.legend()
        figure.tight_layout()
        return figure, ax

    def latest_leader_contributions(self):
        """Return contribution shares for the latest measured week."""
        if self.entropy_history.empty:
            return pd.Series(dtype=float)
        date = self.entropy_history.index[-1]
        shares = {
            stock: self.signal_data[stock].loc[date, "PositiveContributionShare"]
            for stock in self.stocks_in_data
        }
        return pd.Series(shares, dtype=float).dropna().sort_values(ascending=False)

    def calibration_status(self):
        """Summarize whether the continuation rule has enough causal history."""
        if self.entropy_history.empty:
            return {
                "EntropyReady": False,
                "CalibrationReady": False,
                "MaximumCalibrationEvents": 0,
                "RequiredCalibrationEvents": self.minimum_calibration_events,
                "ActiveSignalWeeks": 0,
                "LatestContinuationEdge": np.nan,
            }
        history = self.entropy_history
        maximum_events = int(history["CalibrationEvents"].max())
        latest_edge = history["ContinuationEdge"].iloc[-1]
        return {
            "EntropyReady": bool(history["EntropyThreshold"].notna().any()),
            "CalibrationReady": maximum_events >= self.minimum_calibration_events,
            "MaximumCalibrationEvents": maximum_events,
            "RequiredCalibrationEvents": self.minimum_calibration_events,
            "ActiveSignalWeeks": int(history["SignalActive"].sum()),
            "LatestContinuationEdge": (
                float(latest_edge) if np.isfinite(latest_edge) else np.nan
            ),
        }
