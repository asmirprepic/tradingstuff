"""Causal stock-level regimes for research and strategy filtering."""
import warnings

import numpy as np
import pandas as pd

from agents.base_agents.trading_agent import TradingAgent
from utilities.stock_regime import fit_regime, filter_step, forecast


class StockRegimeAgent(TradingAgent):
    def __init__(self, data, estimation_window=252, refit_interval=20,
                 market_returns=None, ar_threshold=.05, maxiter=200,
                 price_type="Close", verbose=False, auto_generate=True,
                 prediction_horizon=1):
        super().__init__(data)
        if (not isinstance(data.index, pd.DatetimeIndex) or not data.index.is_unique
                or not data.index.is_monotonic_increasing or data.empty):
            raise ValueError("data needs a nonempty, sorted, unique DatetimeIndex")
        if not isinstance(data.columns, pd.MultiIndex) or data.columns.nlevels != 2:
            raise ValueError("Expected (Stock, Field) columns")
        if not data.columns.is_unique:
            raise ValueError("Duplicate columns")
        for name, value, minimum in [("estimation_window", estimation_window, 40),
                                     ("refit_interval", refit_interval, 1),
                                     ("maxiter", maxiter, 1),
                                     ("prediction_horizon", prediction_horizon, 1)]:
            if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(ar_threshold) or not 0 <= ar_threshold < 1:
            raise ValueError("ar_threshold must be in [0, 1)")
        self.estimation_window = int(estimation_window)
        self.refit_interval = int(refit_interval)
        self.maxiter = int(maxiter)
        self.prediction_horizon = int(prediction_horizon)
        self.ar_threshold = ar_threshold
        self.verbose = verbose
        self.price_type = price_type
        self.algorithm_name = "StockRegime"
        self.score_column = "ExpectedReturn"
        self.close = data.xs(price_type, level=1, axis=1).astype(float)
        self.returns = np.log(self.close.where(self.close > 0)).diff()
        self.stocks_in_data = self.close.columns
        if len(self.close.columns) < 2 and market_returns is None:
            raise ValueError("Supply market_returns for a single-stock universe")
        if market_returns is not None:
            if not isinstance(market_returns, pd.Series) or not market_returns.index.is_unique:
                raise ValueError("market_returns must be a unique-index Series of log returns")
            if market_returns.index.tz != data.index.tz:
                raise ValueError("market_returns timezone must match data")
            market_returns = pd.to_numeric(market_returns, errors="coerce").reindex(data.index)
        self.market_returns = market_returns
        self.fit_history = {}
        if auto_generate:
            self.run_all()

    def generate_signal_strategy(self, stock, mode="backtest"):
        if mode not in {"backtest", "live"}:
            raise ValueError("mode must be backtest or live")
        r = self.returns[stock]
        market = (self.market_returns if self.market_returns is not None else
                  self.returns.drop(columns=stock).mean(axis=1))
        inputs = pd.DataFrame({"r": r, "lag": r.shift(1), "market": market})
        inputs = inputs.replace([np.inf, -np.inf], np.nan)
        columns = ["P_State0", "P_State1", "P_Continuation", "P_Reversal", "P_Neutral",
                   "AR0", "AR1", "ExpectedReturn", "ForecastVolatility"]
        out = pd.DataFrame(np.nan, index=r.index, columns=columns)
        out["Regime"] = "UNAVAILABLE"
        model, probability = None, None
        fit_rows = []
        for t in range(self.estimation_window + 2, len(inputs)):
            if (t - self.estimation_window - 2) % self.refit_interval == 0:
                train = inputs.iloc[t-self.estimation_window:t]
                record = {"Date": r.index[t], "TrainingEnd": r.index[t-1]}
                try:
                    if train.isna().any().any():
                        raise ValueError("missing training observations; gaps are not compressed")
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        model = fit_regime(train.r, train.lag, train.market, self.maxiter)
                    probability = model["probability"]
                    record.update(Status="OK", LogLikelihood=model["llf"],
                                  Warnings="; ".join(str(w.message) for w in captured))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as exc:
                    model = None
                    record.update(Status="FAILED", Error=str(exc))
                fit_rows.append(record)
                if self.verbose:
                    print(f"{stock} {r.index[t]}: {record['Status']}", flush=True)
            if model is None:
                continue
            if inputs.iloc[t].isna().any():
                model = None  # Restart from a complete window at the next scheduled refit.
                continue
            probability = filter_step(model, probability, *inputs.iloc[t].to_numpy())
            mean, volatility, _ = forecast(
                model, probability, r.iloc[t], self.prediction_horizon)
            phi = model["coef"][:, 1]
            continuation = float(probability[phi > self.ar_threshold].sum())
            reversal = float(probability[phi < -self.ar_threshold].sum())
            neutral = max(0., 1 - continuation - reversal)
            out.loc[r.index[t], columns] = [*probability, continuation, reversal, neutral,
                                            *phi, mean, volatility]
            probs = [continuation, reversal, neutral]
            out.loc[r.index[t], "Regime"] = (
                ["CONTINUATION", "REVERSAL", "NEUTRAL"][int(np.argmax(probs))]
                if max(probs) >= .6 else "UNCERTAIN")
        out["price"] = self.close[stock]
        out["PredictionHorizon"] = self.prediction_horizon
        out["return"] = r
        out["SignalStrength"] = out.ExpectedReturn
        out["Valid"] = out.ExpectedReturn.notna()
        out["Position"] = out.ExpectedReturn.gt(0).astype(int)
        out["Signal"] = np.sign(out.Position.diff().fillna(0)).astype(int)
        self.signal_data[stock] = out
        self.fit_history[stock] = pd.DataFrame(fit_rows)
        return out

    def run_all(self, mode="backtest"):
        self.signal_data = {}
        self.fit_history = {}
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock, mode)
        return self
