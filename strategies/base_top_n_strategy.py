from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseTopNStrategy(ABC):
    """Shared portfolio construction and evaluation for long-only top-N strategies."""

    def __init__(self, data, strategy_name, top_n=10, rebalance_frequency="W-FRI",
                 price_type="Close", transaction_cost=0.0, evaluation_start=None,
                 performance_benchmark=None, execution_mode="next_open"):
        self.strategy_name = strategy_name
        self.data = data
        self.top_n = int(top_n)
        self.rebalance_frequency = rebalance_frequency
        self.price_type = price_type
        self.transaction_cost = float(transaction_cost)
        self.requested_evaluation_start = (
            None if evaluation_start is None else pd.Timestamp(evaluation_start)
        )
        self.performance_benchmark = performance_benchmark
        self.execution_mode = str(execution_mode).lower().strip()
        self.benchmark_name = "EqualWeightBuyHold"
        self._validate_inputs()
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        if self.top_n > len(self.stocks_in_data):
            raise ValueError("top_n cannot exceed the number of stocks in the dataset.")

        self.agent = None
        self.score_matrix = pd.DataFrame()
        self.eligibility_matrix = pd.DataFrame()
        self.target_weights = pd.DataFrame()
        self.effective_weights = pd.DataFrame()
        self.selection_log = pd.DataFrame()
        self.portfolio_returns = pd.Series(dtype=float)
        self.equity_curve = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)
        self.benchmark_equity_curve = pd.Series(dtype=float)
        self.evaluation_start = None
        self.weight_rebalance_dates = pd.DatetimeIndex([])
        self.overnight_weights = pd.DataFrame()
        self.intraday_weights = pd.DataFrame()
        self.overnight_returns = pd.Series(dtype=float)
        self.intraday_returns = pd.Series(dtype=float)
        self.gross_portfolio_returns = pd.Series(dtype=float)
        self.execution_flags = pd.Series(dtype=bool)

    def _validate_inputs(self):
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            raise ValueError("data must be a non-empty pandas DataFrame.")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("data must use a DatetimeIndex.")
        if not self.data.index.is_monotonic_increasing or not self.data.index.is_unique:
            raise ValueError("data index must be sorted and unique.")
        if not isinstance(self.data.columns, pd.MultiIndex) or self.data.columns.nlevels < 2:
            raise ValueError("data must have (ticker, field) MultiIndex columns.")
        if self.top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost cannot be negative.")
        if self.execution_mode not in {
            "next_open", "next_close", "legacy_same_close"
        }:
            raise ValueError(
                "execution_mode must be 'next_open', 'next_close', "
                "or 'legacy_same_close'."
            )
        if self.price_type not in self.data.columns.get_level_values(1):
            raise ValueError(f"Price field {self.price_type!r} was not found in data.")
        try:
            pd.tseries.frequencies.to_offset(self.rebalance_frequency)
        except (TypeError, ValueError) as exc:
            raise ValueError("rebalance_frequency must be a valid pandas period frequency.") from exc

    def _rebalance_dates(self):
        indexed_dates = pd.Series(self.data.index, index=self.data.index)
        grouped = indexed_dates.resample(self.rebalance_frequency).last().dropna()
        # Resample also creates a bucket for an unfinished final week/month.
        completed = grouped.index <= self.data.index[-1]
        dates = grouped.loc[completed]
        return pd.DatetimeIndex(dates.tolist())

    def _resolve_evaluation_start(self, natural_start):
        if self.requested_evaluation_start is None:
            return natural_start

        index = self.data.index
        try:
            position = index.searchsorted(self.requested_evaluation_start, side="left")
        except TypeError as exc:
            raise ValueError(
                "evaluation_start must have the same timezone convention as the data index."
            ) from exc
        if position >= len(index):
            raise ValueError("evaluation_start is later than the final data row.")

        resolved = index[position]
        if natural_start is None or resolved < natural_start:
            required = natural_start if natural_start is not None else "unavailable"
            raise ValueError(
                f"evaluation_start resolves to {resolved}, but this configuration's "
                f"first executable date is {required}. Provide more pre-start history "
                "or choose a later evaluation_start."
            )
        return resolved

    def _build_target_weights(self, minimum_history):
        if self.score_matrix.empty or self.eligibility_matrix.empty:
            raise RuntimeError("Signal matrices must be populated before building weights.")
        index = self.data.index
        weights = pd.DataFrame(np.nan, index=index, columns=self.stocks_in_data, dtype=float)
        valid_dates = [
            date for date in self._rebalance_dates() if index.get_loc(date) >= minimum_history
        ]
        for date in valid_dates:
            weights.loc[date] = 0.0
            scores = self.score_matrix.loc[date].where(self.eligibility_matrix.loc[date]).dropna()
            selected = scores.nlargest(self.top_n).index
            weights.loc[date, selected] = 1.0 / self.top_n

        self.target_weights = weights.ffill().fillna(0.0)
        self.effective_weights = self.target_weights.shift(1).fillna(0.0)
        self.weight_rebalance_dates = pd.DatetimeIndex(valid_dates)
        natural_start = None
        if valid_dates:
            first_position = index.get_loc(valid_dates[0]) + 1
            if first_position < len(index):
                natural_start = index[first_position]
        self.evaluation_start = self._resolve_evaluation_start(natural_start)
        return self.target_weights

    @abstractmethod
    def generate_target_weights(self):
        """Generate signal matrices and convert them to target weights."""

    def calculate_returns(self):
        if self.target_weights.empty:
            raise RuntimeError("Generate target weights before calculating returns.")
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        if self.execution_mode != "legacy_same_close":
            return self._calculate_realistic_returns(close)

        daily_returns = close.pct_change(fill_method=None)
        gross_returns = (daily_returns * self.effective_weights).sum(axis=1, min_count=1).fillna(0.0)
        turnover = self.effective_weights.diff().abs().sum(axis=1).fillna(0.0)
        self.portfolio_returns = gross_returns - turnover * self.transaction_cost
        self.overnight_weights = pd.DataFrame(
            0.0, index=close.index, columns=close.columns
        )
        self.intraday_weights = self.effective_weights.copy()
        self.overnight_returns = pd.Series(0.0, index=close.index)
        self.intraday_returns = gross_returns.copy()
        self.gross_portfolio_returns = gross_returns.copy()
        self.execution_flags = turnover.gt(0.0)

        self._finalize_return_outputs(close, turnover)
        return self.portfolio_returns

    def _execution_schedule(self, index):
        schedule = {}
        for decision_date in self.weight_rebalance_dates:
            if decision_date not in index:
                continue
            execution_position = index.get_loc(decision_date) + 1
            if execution_position < len(index):
                schedule[index[execution_position]] = decision_date
        return schedule

    @staticmethod
    def _drift_weights(weights, stock_returns, portfolio_return):
        denominator = 1.0 + portfolio_return
        if denominator <= 0.0:
            return pd.Series(0.0, index=weights.index, dtype=float)
        return (weights * (1.0 + stock_returns) / denominator).clip(lower=0.0)

    @staticmethod
    def _validate_required_returns(stock_returns, weights, date, segment):
        required = weights[weights.gt(1e-15)].index
        if required.empty:
            return
        invalid = stock_returns.reindex(required).loc[
            lambda values: ~np.isfinite(values)
        ]
        if not invalid.empty:
            stocks = ", ".join(map(str, invalid.index[:5]))
            raise ValueError(
                f"Missing or invalid {segment} prices on {date} for held stocks: "
                f"{stocks}."
            )

    def _calculate_realistic_returns(self, close):
        if self.execution_mode == "next_open":
            if "Open" not in self.data.columns.get_level_values(1):
                raise ValueError("execution_mode='next_open' requires Open prices.")
            open_prices = self.data.xs("Open", level=1, axis=1).astype(float)
            missing = close.columns.difference(open_prices.columns)
            if len(missing):
                raise ValueError(
                    "execution_mode='next_open' requires Open prices for every stock."
                )
            open_prices = open_prices.reindex(columns=close.columns)
            overnight_stock_returns = open_prices.div(close.shift(1)).sub(1.0)
            intraday_stock_returns = close.div(open_prices).sub(1.0)
        else:
            close_returns = close.pct_change(fill_method=None)

        effective = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        overnight_weights = effective.copy()
        intraday_weights = effective.copy()
        overnight_returns = pd.Series(0.0, index=close.index, dtype=float)
        intraday_returns = pd.Series(0.0, index=close.index, dtype=float)
        gross_returns = pd.Series(0.0, index=close.index, dtype=float)
        net_returns = pd.Series(0.0, index=close.index, dtype=float)
        turnover = pd.Series(0.0, index=close.index, dtype=float)
        executions = pd.Series(False, index=close.index, dtype=bool)
        schedule = self._execution_schedule(close.index)
        end_weights = pd.Series(0.0, index=close.columns, dtype=float)

        for date in close.index:
            decision_date = schedule.get(date)
            if self.execution_mode == "next_open":
                overnight_weights.loc[date] = end_weights
                overnight_stock = overnight_stock_returns.loc[date]
                self._validate_required_returns(
                    overnight_stock, end_weights, date, "close-to-open"
                )
                overnight_return = float(
                    (end_weights * overnight_stock.fillna(0.0)).sum()
                )
                pretrade_weights = self._drift_weights(
                    end_weights,
                    overnight_stock.fillna(0.0),
                    overnight_return,
                )
                active_weights = pretrade_weights
                if decision_date is not None:
                    active_weights = self.target_weights.loc[decision_date].fillna(0.0)
                    turnover.loc[date] = (active_weights - pretrade_weights).abs().sum()
                    executions.loc[date] = True
                intraday_stock = intraday_stock_returns.loc[date]
                self._validate_required_returns(
                    intraday_stock, active_weights, date, "open-to-close"
                )
                intraday_return = float(
                    (active_weights * intraday_stock.fillna(0.0)).sum()
                )
                overnight_returns.loc[date] = overnight_return
                intraday_returns.loc[date] = intraday_return
                overnight_weights.loc[date] = end_weights
                intraday_weights.loc[date] = active_weights
                effective.loc[date] = active_weights
                gross_returns.loc[date] = (
                    (1.0 + overnight_return) * (1.0 + intraday_return) - 1.0
                )
                cost_rate = turnover.loc[date] * self.transaction_cost
                if cost_rate >= 1.0:
                    raise ValueError("Transaction costs consume the entire portfolio.")
                net_returns.loc[date] = (
                    (1.0 + overnight_return)
                    * (1.0 - cost_rate)
                    * (1.0 + intraday_return)
                    - 1.0
                )
                end_weights = self._drift_weights(
                    active_weights,
                    intraday_stock.fillna(0.0),
                    intraday_return,
                )
            else:
                active_weights = end_weights.copy()
                effective.loc[date] = active_weights
                overnight_weights.loc[date] = active_weights
                close_stock = close_returns.loc[date]
                self._validate_required_returns(
                    close_stock, active_weights, date, "close-to-close"
                )
                gross_return = float(
                    (active_weights * close_stock.fillna(0.0)).sum()
                )
                pretrade_weights = self._drift_weights(
                    active_weights,
                    close_stock.fillna(0.0),
                    gross_return,
                )
                end_weights = pretrade_weights
                if decision_date is not None:
                    target = self.target_weights.loc[decision_date].fillna(0.0)
                    turnover.loc[date] = (target - pretrade_weights).abs().sum()
                    end_weights = target
                    executions.loc[date] = True
                gross_returns.loc[date] = gross_return
                overnight_returns.loc[date] = gross_return
                cost_rate = turnover.loc[date] * self.transaction_cost
                if cost_rate >= 1.0:
                    raise ValueError("Transaction costs consume the entire portfolio.")
                net_returns.loc[date] = (
                    (1.0 + gross_return) * (1.0 - cost_rate) - 1.0
                )

        self.effective_weights = effective
        self.overnight_weights = overnight_weights
        self.intraday_weights = intraday_weights
        self.overnight_returns = overnight_returns
        self.intraday_returns = intraday_returns
        self.gross_portfolio_returns = gross_returns
        self.execution_flags = executions
        self.portfolio_returns = net_returns
        self._finalize_return_outputs(close, turnover)
        return self.portfolio_returns

    def _finalize_return_outputs(self, close, turnover):
        """Build benchmark, equity curves, and the common strategy log."""

        benchmark_returns = pd.Series(0.0, index=close.index, dtype=float)
        if self.evaluation_start is not None and self.performance_benchmark is None:
            evaluation_close = close.loc[self.evaluation_start:]
            start_prices = evaluation_close.iloc[0].replace(0.0, np.nan).dropna()
            benchmark_equity = evaluation_close[start_prices.index].div(start_prices).mean(axis=1)
            benchmark_returns.loc[benchmark_equity.index] = benchmark_equity.pct_change().fillna(0.0)
        elif self.evaluation_start is not None:
            benchmark = self.performance_benchmark
            prices = benchmark if isinstance(benchmark, pd.Series) else getattr(benchmark, "prices", None)
            if not isinstance(prices, pd.Series) or prices.empty:
                raise ValueError(
                    "performance_benchmark must be a non-empty price Series or an object "
                    "with a non-empty prices Series."
                )
            self.benchmark_name = str(
                getattr(benchmark, "name", None) or prices.name or "ExternalBenchmark"
            )
            external_prices = pd.to_numeric(prices, errors="coerce").copy()
            external_index = pd.DatetimeIndex(external_prices.index)
            if external_index.tz is not None:
                external_index = external_index.tz_localize(None)
            external_prices.index = external_index.normalize()
            external_prices = external_prices.groupby(level=0).last().sort_index()

            strategy_dates = close.index
            normalized_dates = strategy_dates
            if normalized_dates.tz is not None:
                normalized_dates = normalized_dates.tz_localize(None)
            normalized_dates = normalized_dates.normalize()
            aligned_prices = external_prices.reindex(normalized_dates).ffill()
            aligned_prices.index = strategy_dates
            evaluation_prices = aligned_prices.loc[self.evaluation_start:]
            if evaluation_prices.empty or pd.isna(evaluation_prices.iloc[0]):
                raise ValueError(
                    "performance_benchmark has no price on or before evaluation_start."
                )
            benchmark_returns.loc[evaluation_prices.index] = (
                evaluation_prices.pct_change(fill_method=None).fillna(0.0)
            )
        self.benchmark_returns = benchmark_returns

        if self.evaluation_start is not None:
            self.portfolio_returns.loc[self.portfolio_returns.index < self.evaluation_start] = 0.0
        else:
            self.portfolio_returns[:] = 0.0
        self.equity_curve = (1.0 + self.portfolio_returns).cumprod()
        self.benchmark_equity_curve = (1.0 + self.benchmark_returns).cumprod()
        rebalance_set = set(self._rebalance_dates())
        self.selection_log = pd.DataFrame({
            "Selected_Stocks": self.target_weights.apply(
                lambda row: list(row.index[row.gt(0.0)]), axis=1
            ),
            "InvestedWeight": self.target_weights.sum(axis=1),
            "N_Held": self.target_weights.gt(0.0).sum(axis=1).astype(int),
            "Rebalance": [date in rebalance_set for date in self.data.index],
            "Turnover": turnover,
            "Execution": self.execution_flags.reindex(close.index).fillna(False),
            "ExecutionMode": self.execution_mode,
            "OvernightReturn": self.overnight_returns.reindex(close.index).fillna(0.0),
            "IntradayReturn": self.intraday_returns.reindex(close.index).fillna(0.0),
            "GrossPortfolioReturn": self.gross_portfolio_returns.reindex(close.index).fillna(0.0),
            "PortfolioReturn": self.portfolio_returns,
            "Equity": self.equity_curve,
            "BenchmarkEquity": self.benchmark_equity_curve,
        }, index=self.data.index)

    @staticmethod
    def _performance_metrics(returns):
        returns = returns.dropna()
        if returns.empty:
            return {name: np.nan for name in (
                "Total_Return_%", "Annualized_Return_%", "Annualized_Volatility_%",
                "Sharpe_Ratio", "Max_Drawdown_%"
            )}
        equity = (1.0 + returns).cumprod()
        volatility = returns.std(ddof=1)
        return {
            "Total_Return_%": (equity.iloc[-1] - 1.0) * 100.0,
            "Annualized_Return_%": (equity.iloc[-1] ** (252.0 / len(returns)) - 1.0) * 100.0,
            "Annualized_Volatility_%": volatility * np.sqrt(252.0) * 100.0,
            "Sharpe_Ratio": returns.mean() / volatility * np.sqrt(252.0) if volatility > 0 else np.nan,
            "Max_Drawdown_%": (equity / equity.cummax() - 1.0).min() * 100.0,
        }

    def evaluate_performance(self):
        if self.equity_curve.empty:
            raise RuntimeError("Run the strategy before evaluating performance.")
        if self.evaluation_start is None:
            return pd.DataFrame()
        strategy = self._performance_metrics(self.portfolio_returns.loc[self.evaluation_start:])
        benchmark = self._performance_metrics(self.benchmark_returns.loc[self.evaluation_start:])
        result = pd.DataFrame([
            {"Portfolio": self.strategy_name, **strategy},
            {"Portfolio": self.benchmark_name, **benchmark},
        ])
        result["Outperformance_%"] = result["Total_Return_%"] - benchmark["Total_Return_%"]
        return result.round(4)

    def run_all(self):
        self.generate_target_weights()
        self.calculate_returns()
        return self
