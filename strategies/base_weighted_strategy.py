import numpy as np
import pandas as pd

from strategies.base_top_n_strategy import BaseTopNStrategy


class BaseWeightedStrategy(BaseTopNStrategy):
    """Shared execution layer for long-only strategies with arbitrary weights."""

    def __init__(
        self,
        data,
        strategy_name,
        rebalance_frequency="W-FRI",
        price_type="Close",
        transaction_cost=0.0,
        evaluation_start=None,
        performance_benchmark=None,
        execution_mode="next_open",
    ):
        if not isinstance(data, pd.DataFrame) or not isinstance(data.columns, pd.MultiIndex):
            stock_count = 1
        else:
            stock_count = len(data.columns.get_level_values(0).unique())
        super().__init__(
            data,
            strategy_name,
            top_n=max(1, stock_count),
            rebalance_frequency=rebalance_frequency,
            price_type=price_type,
            transaction_cost=transaction_cost,
            evaluation_start=evaluation_start,
            performance_benchmark=performance_benchmark,
            execution_mode=execution_mode,
        )

    def _finalize_weight_targets(self, rebalance_weights, valid_dates):
        """Validate sparse rebalance targets and apply next-bar execution."""
        weights = pd.DataFrame(
            rebalance_weights,
            index=self.data.index,
            columns=self.stocks_in_data,
            dtype=float,
        )
        finite = weights.stack(future_stack=True).dropna()
        if not np.isfinite(finite.to_numpy(dtype=float)).all():
            raise ValueError("Target weights must be finite where supplied.")
        if finite.lt(-1e-12).any():
            raise ValueError("Long-only target weights cannot be negative.")
        row_sums = weights.sum(axis=1, min_count=1).dropna()
        if row_sums.gt(1.0 + 1e-9).any():
            raise ValueError("Long-only target weights cannot exceed 100% exposure.")

        self.target_weights = weights.ffill().fillna(0.0).clip(lower=0.0)
        self.effective_weights = self.target_weights.shift(1).fillna(0.0)
        self.weight_rebalance_dates = pd.DatetimeIndex(valid_dates)
        natural_start = None
        valid_dates = list(valid_dates)
        if valid_dates:
            first_position = self.data.index.get_loc(valid_dates[0]) + 1
            if first_position < len(self.data.index):
                natural_start = self.data.index[first_position]
        self.evaluation_start = self._resolve_evaluation_start(natural_start)
        return self.target_weights

    def calculate_returns(self):
        """Apply targets on the next bar and let holdings drift until rebalance."""
        if self.target_weights.empty:
            raise RuntimeError("Generate target weights before calculating returns.")
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        if self.execution_mode != "legacy_same_close":
            return self._calculate_realistic_returns(close)

        daily_returns = close.pct_change(fill_method=None).fillna(0.0)
        effective = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        portfolio_returns = pd.Series(0.0, index=close.index, dtype=float)
        turnover = pd.Series(0.0, index=close.index, dtype=float)

        executions = {}
        for decision_date in self.weight_rebalance_dates:
            position = close.index.get_loc(decision_date) + 1
            if position < len(close.index):
                executions[close.index[position]] = decision_date

        end_weights = pd.Series(0.0, index=close.columns, dtype=float)
        for date in close.index:
            start_weights = end_weights.copy()
            if date in executions:
                target = self.target_weights.loc[executions[date]].fillna(0.0)
                turnover.loc[date] = (target - end_weights).abs().sum()
                start_weights = target
            effective.loc[date] = start_weights

            gross_return = float((start_weights * daily_returns.loc[date]).sum())
            trading_cost = turnover.loc[date] * self.transaction_cost
            net_return = gross_return - trading_cost
            portfolio_returns.loc[date] = net_return
            denominator = 1.0 + net_return
            if denominator <= 0.0:
                end_weights[:] = 0.0
            else:
                end_weights = (
                    start_weights * (1.0 + daily_returns.loc[date]) / denominator
                ).clip(lower=0.0)

        self.effective_weights = effective
        self.portfolio_returns = portfolio_returns
        self.overnight_weights = pd.DataFrame(
            0.0, index=close.index, columns=close.columns
        )
        self.intraday_weights = effective.copy()
        self.overnight_returns = pd.Series(0.0, index=close.index)
        self.intraday_returns = portfolio_returns + turnover * self.transaction_cost
        self.gross_portfolio_returns = self.intraday_returns.copy()
        self.execution_flags = turnover.gt(0.0)
        self._finalize_return_outputs(close, turnover)
        return self.portfolio_returns
