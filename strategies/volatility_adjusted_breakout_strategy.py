import numpy as np
import pandas as pd

from agents.technical.volatility_adjusted_breakout_agent import (
    VolatilityAdjustedBreakoutAgent,
)
from strategies.base_top_n_strategy import BaseTopNStrategy


class VolatilityAdjustedBreakoutStrategy(BaseTopNStrategy):
    """Persistent breakout portfolio with ATR-based trailing exits."""

    def __init__(self, data, breakout_window=60, atr_window=20,
                 stop_atr_multiple=3.0, top_n=10, rebalance_frequency="B",
                 price_type="Close", transaction_cost=0.0, auto_generate=True,
                 evaluation_start=None, performance_benchmark=None):
        self.breakout_window = int(breakout_window)
        self.atr_window = int(atr_window)
        self.stop_atr_multiple = float(stop_atr_multiple)
        if self.breakout_window < 1:
            raise ValueError("breakout_window must be a positive integer.")
        if self.atr_window < 1:
            raise ValueError("atr_window must be a positive integer.")
        if self.stop_atr_multiple <= 0:
            raise ValueError("stop_atr_multiple must be positive.")
        super().__init__(
            data,
            "VolatilityAdjustedBreakout",
            top_n,
            rebalance_frequency,
            price_type,
            transaction_cost,
            evaluation_start,
            performance_benchmark,
        )
        self.entry_matrix = pd.DataFrame()
        self.exit_matrix = pd.DataFrame()
        self.trailing_stop_matrix = pd.DataFrame()
        if auto_generate:
            self.run_all()

    def generate_target_weights(self):
        self.agent = VolatilityAdjustedBreakoutAgent(
            self.data,
            breakout_window=self.breakout_window,
            atr_window=self.atr_window,
            price_type=self.price_type,
            auto_generate=True,
        )
        index = self.data.index
        columns = self.stocks_in_data
        close = self.data.xs(self.price_type, level=1, axis=1).astype(float)
        atr = pd.DataFrame(index=index, columns=columns, dtype=float)
        self.score_matrix = pd.DataFrame(index=index, columns=columns, dtype=float)
        self.eligibility_matrix = pd.DataFrame(False, index=index, columns=columns)

        for stock in columns:
            signals = self.agent.signal_data[stock]
            atr[stock] = signals["ATR"]
            self.score_matrix[stock] = signals["BreakoutScore"]
            self.eligibility_matrix[stock] = signals["Position"].eq(1)

        weights = pd.DataFrame(0.0, index=index, columns=columns)
        self.entry_matrix = pd.DataFrame(False, index=index, columns=columns)
        self.exit_matrix = pd.DataFrame(False, index=index, columns=columns)
        self.trailing_stop_matrix = pd.DataFrame(np.nan, index=index, columns=columns)
        minimum_history = max(self.breakout_window, self.atr_window - 1)
        decision_dates = {
            date for date in self._rebalance_dates()
            if index.get_loc(date) >= minimum_history
        }
        held = set()
        peak_closes = {}

        for position, date in enumerate(index):
            if position < minimum_history:
                continue

            for stock in list(held):
                current_close = close.at[date, stock]
                current_atr = atr.at[date, stock]
                if pd.notna(current_close):
                    peak_closes[stock] = max(peak_closes[stock], current_close)
                if pd.notna(current_atr):
                    self.trailing_stop_matrix.at[date, stock] = (
                        peak_closes[stock] - self.stop_atr_multiple * current_atr
                    )

            if date in decision_dates:
                exits = [
                    stock for stock in held
                    if pd.notna(self.trailing_stop_matrix.at[date, stock])
                    and close.at[date, stock] < self.trailing_stop_matrix.at[date, stock]
                ]
                for stock in exits:
                    held.remove(stock)
                    peak_closes.pop(stock, None)
                    self.exit_matrix.at[date, stock] = True

                slots = self.top_n - len(held)
                if slots > 0:
                    scores = self.score_matrix.loc[date].where(
                        self.eligibility_matrix.loc[date]
                    ).dropna()
                    candidates = scores.drop(labels=list(held), errors="ignore").nlargest(slots)
                    for stock in candidates.index:
                        held.add(stock)
                        peak_closes[stock] = close.at[date, stock]
                        self.entry_matrix.at[date, stock] = True
                        self.trailing_stop_matrix.at[date, stock] = (
                            peak_closes[stock]
                            - self.stop_atr_multiple * atr.at[date, stock]
                        )

            if held:
                weights.loc[date, list(held)] = 1.0 / self.top_n

        self.target_weights = weights
        self.effective_weights = self.target_weights.shift(1).fillna(0.0)
        valid_dates = sorted(decision_dates)
        natural_start = None
        if valid_dates:
            first_position = index.get_loc(valid_dates[0]) + 1
            if first_position < len(index):
                natural_start = index[first_position]
        self.evaluation_start = self._resolve_evaluation_start(natural_start)
        return self.target_weights

    def calculate_returns(self):
        returns = super().calculate_returns()
        self.selection_log["Entries"] = self.entry_matrix.apply(
            lambda row: list(row.index[row]), axis=1
        )
        self.selection_log["Exits"] = self.exit_matrix.apply(
            lambda row: list(row.index[row]), axis=1
        )
        return returns
