from agents.base_agents.trading_agent import TradingAgent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PerformanceBasedAgent(TradingAgent):
    """
    Cross-sectional momentum selector that holds the top-N stocks for a fixed holding period.
    """

    def __init__(
        self,
        data,
        period_length=5,
        top_n=5,
        holding_period=20,
        price_type="Close",
        auto_generate=True,
    ):
        super().__init__(data)
        self.algorithm_name = "PerformanceBasedAgent"
        self.score_column = "LookbackReturn"

        self.period_length = int(period_length)
        self.top_n = int(top_n)
        self.holding_period = int(holding_period)
        if self.period_length < 1:
            raise ValueError("period_length must be a positive integer.")
        if self.top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        if self.holding_period < 1:
            raise ValueError("holding_period must be a positive integer.")

        self.price_type = price_type
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        self.holdings_matrix = pd.DataFrame(index=self.data.index, columns=self.stocks_in_data, dtype=np.int8)
        self.selection_log = pd.DataFrame(index=self.data.index)
        self.portfolio_log_returns = pd.Series(dtype=float)
        self.cumulative_returns = pd.Series(dtype=float)

        if auto_generate:
            self.run_all()

    def _build_holdings_matrix(self):
        close = self.data.xs(self.price_type, level=1, axis=1)
        lookback_returns = np.log(close / close.shift(self.period_length))
        one_period_returns = np.log(close / close.shift(1))
        ranked_returns = lookback_returns.rank(axis=1, method="first", ascending=False)

        selection_matrix = pd.DataFrame(0, index=close.index, columns=self.stocks_in_data, dtype=np.int8)
        holding_matrix = pd.DataFrame(0, index=close.index, columns=self.stocks_in_data, dtype=np.int8)

        valid_dates = [
            pos
            for pos, date in enumerate(close.index)
            if lookback_returns.loc[date].dropna().shape[0] >= self.top_n
        ]

        if valid_dates:
            first_valid_pos = valid_dates[0]
            for start_pos in range(first_valid_pos, len(close.index), self.holding_period):
                date = close.index[start_pos]
                ranked = lookback_returns.loc[date].dropna()
                top_stocks = ranked.nlargest(self.top_n).index
                selection_matrix.loc[date, top_stocks] = 1

                end_pos = min(start_pos + self.holding_period, len(close.index))
                holding_matrix.iloc[start_pos:end_pos, holding_matrix.columns.isin(top_stocks)] = 1
                holding_matrix.iloc[start_pos:end_pos, ~holding_matrix.columns.isin(top_stocks)] = 0

        return close, one_period_returns, lookback_returns, ranked_returns, selection_matrix, holding_matrix

    def generate_signal_strategy(self, stock=None, mode="backtest"):
        (
            close,
            one_period_returns,
            lookback_returns,
            ranked_returns,
            selection_matrix,
            holding_matrix,
        ) = self._build_holdings_matrix()

        self.holdings_matrix = holding_matrix
        self.selection_matrix = selection_matrix
        self.signal_data = {}

        for ticker in self.stocks_in_data:
            df = pd.DataFrame(index=self.data.index)
            df["return"] = one_period_returns[ticker]
            df["LookbackReturn"] = lookback_returns[ticker]
            df["Rank"] = ranked_returns[ticker]
            df["Position"] = holding_matrix[ticker].astype(int)
            sig = df["Position"].diff().fillna(0).astype(int)
            df["Signal"] = sig.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            self.signal_data[ticker] = df

        if stock is None:
            return self.signal_data
        if stock not in self.signal_data:
            raise KeyError(f"Stock {stock} not found in signal data.")
        return self.signal_data[stock]

    def calculate_returns(self):
        super().calculate_returns()

        prices = self.data.xs(self.price_type, level=1, axis=1).copy()
        stock_logret = np.log(prices / prices.shift(1))
        hold = self.holdings_matrix.reindex(prices.index).fillna(0).astype(float)
        hold_lag = hold.shift(1).fillna(0)

        n_held = hold_lag.sum(axis=1).replace(0, np.nan)
        port_logret = (stock_logret * hold_lag).sum(axis=1) / n_held
        port_logret = port_logret.fillna(0.0)

        self.portfolio_log_returns = port_logret
        self.cumulative_returns = port_logret.cumsum()

        self.selection_log = pd.DataFrame(
            {
                "Selected_Stocks": hold.apply(lambda row: list(row.index[row.values == 1]), axis=1),
                "Log_Return": port_logret,
                "Cum_Log_Return": self.cumulative_returns,
                "N_Held": hold.sum(axis=1).astype(int),
            },
            index=prices.index,
        )

    def run_all(self, mode="backtest"):
        self.generate_signal_strategy(mode=mode)
        self.calculate_returns()

    def plot_signals(self):
        plt.figure(figsize=(14, 7))
        for stock in self.stocks_in_data:
            plt.plot(self.data.index, self.data[(stock, self.price_type)], label=f"{stock} Price")
            buy_signals = self.holdings_matrix[self.holdings_matrix[stock] == 1].index
            plt.scatter(
                buy_signals,
                self.data[(stock, self.price_type)].loc[buy_signals],
                marker="^",
                color="g",
                label=f"{stock} Buy Signal",
            )

        plt.title("Stock Prices and Trading Signals")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_returns(self):
        fig, ax = plt.subplots(figsize=(14, 7))
        strategy_return = self.cumulative_returns.iloc[-1] * 100
        buy_and_hold_return = (
            (
                self.data.xs(self.price_type, level=1, axis=1).iloc[-1]
                / self.data.xs(self.price_type, level=1, axis=1).iloc[0]
                - 1
            ).mean()
            * 100
        )

        bars1 = ax.bar(["Strategy Return"], [strategy_return], width=0.4, label="Strategy Return")
        bars2 = ax.bar(["Buy and Hold Return"], [buy_and_hold_return], width=0.4, label="Buy and Hold Return")

        ax.set_ylabel("Returns (%)")
        ax.set_title("Strategy Returns vs Buy and Hold Returns")
        ax.legend()

        self._autolabel(ax, bars1)
        self._autolabel(ax, bars2)

        fig.tight_layout()
        plt.show()

    def _autolabel(self, ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                "{}".format(round(height, 2)),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    def plot_selected_stocks(self):
        selected_stocks = self.holdings_matrix[self.holdings_matrix == 1]
        plt.figure(figsize=(14, 7))
        for stock in self.stocks_in_data:
            stock_selection = selected_stocks[stock][selected_stocks[stock] == 1].index
            plt.scatter(stock_selection, [stock] * len(stock_selection), marker="o", label=f"{stock} Selected")

        plt.title("Selected Stocks Over Time")
        plt.xlabel("Date")
        plt.ylabel("Stock")
        plt.grid(True)
        plt.show()

    def plot_returns_time(self):
        fig, ax = plt.subplots(figsize=(14, 7))
        self.cumulative_returns.plot(ax=ax, label="Strategy Cumulative Returns")

        for date in self.cumulative_returns.index:
            if pd.isna(self.cumulative_returns.loc[date]):
                continue
            selected_stocks = self.holdings_matrix.loc[date][self.holdings_matrix.loc[date] == 1].index
            selected_text = ", ".join(selected_stocks)
            ax.annotate(
                selected_text,
                (date, self.cumulative_returns.loc[date]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

        plt.title("Strategy Cumulative Returns with Selected Stocks")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_top_stocks_for_date(self, date, top_n=3):
        if date not in self.data.index:
            raise ValueError(f"Date {date} not found in the dataset.")
        date_idx = self.data.index.get_loc(date)
        if date_idx < self.period_length:
            raise ValueError(f"Not enough historical data to calculate returns for {date}.")

        returns = pd.Series(index=self.stocks_in_data, dtype=np.float64)
        for stock in self.stocks_in_data:
            prev_price = self.data.loc[self.data.index[date_idx - self.period_length], (stock, self.price_type)]
            current_price = self.data.loc[date, (stock, self.price_type)]
            returns[stock] = np.log(current_price / prev_price)

        return returns.nlargest(top_n).index.tolist()
