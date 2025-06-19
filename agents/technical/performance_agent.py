from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PerformanceBasedAgent(TradingAgent):
    """
    A momentum-based trading agent that selects top-performing stocks over a lookback window
    and holds them for a fixed period.

    Args:
        data (pd.DataFrame): MultiIndex columns with (stock, price_type).
        period_length (int): Lookback window for return ranking.
        top_n (int): Number of top stocks to select.
        holding_period (int): Number of days to hold selected stocks.
        price_type (str): Price type to use (e.g., 'Close').
        auto_generate (bool): Whether to generate signals and returns immediately.
    """

    def __init__(self, data, period_length=5, top_n=5, holding_period=20, price_type='Close', auto_generate=True):
        super().__init__(data)
        self.algorithm_name = "PerformanceBased"
        self.period_length = period_length
        self.top_n = top_n
        self.holding_period = holding_period
        self.price_type = price_type
        self.stocks_in_data = data.columns.get_level_values(0).unique()
        self.signal_data = None

        if auto_generate:
            self.generate_signal_strategy()
            self.calculate_returns()

    def generate_signal_strategy(self):
        prices = self.data.xs(self.price_type, level=1, axis=1)
        log_returns = np.log(prices / prices.shift(self.period_length))
        signals = pd.DataFrame(0, index=prices.index, columns=self.stocks_in_data)

        for date in log_returns.index[self.period_length:]:
            top_stocks = log_returns.loc[date].nlargest(self.top_n).index
            signals.loc[date, top_stocks] = 1

        # Apply holding logic
        holding_signals = pd.DataFrame(0, index=signals.index, columns=signals.columns)
        for i in range(len(signals)):
            if signals.iloc[i].sum() > 0:
                top = signals.columns[signals.iloc[i] == 1]
                holding_range = range(i, min(i + self.holding_period, len(signals)))
                holding_signals.iloc[holding_range, holding_signals.columns.isin(top)] = 1

        self.signal_data = holding_signals

    def calculate_returns(self):
        prices = self.data.xs(self.price_type, level=1, axis=1)
        daily_returns = prices.pct_change().fillna(0)

        strategy_returns = (daily_returns * self.signal_data).mean(axis=1)
        self.strategy_log_returns = np.log(1 + strategy_returns.fillna(0))
        self.cumulative_returns = self.strategy_log_returns.cumsum()

    def plot_returns(self):
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(self.cumulative_returns, label="Strategy")

        prices = self.data.xs(self.price_type, level=1, axis=1)
        bh_returns = np.log(prices / prices.shift(1)).mean(axis=1).cumsum()
        ax.plot(bh_returns, label="Buy & Hold (Avg)")

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Log Returns")
        ax.legend()
        ax.grid(True)
        plt.show()
        return fig, ax

    def plot_signals(self):
        plt.figure(figsize=(14, 7))
        for stock in self.stocks_in_data:
            plt.plot(self.data.index, self.data[(stock, self.price_type)], label=f'{stock} Price')
            buy_signals = self.signal_data[self.signal_data[stock] == 1].index
            plt.scatter(buy_signals, self.data[(stock, self.price_type)].loc[buy_signals], marker='^', color='g', label=f'{stock} Buy Signal')

        plt.title('Stock Prices and Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_selected_stocks(self):
        selected_stocks = self.signal_data[self.signal_data == 1]
        plt.figure(figsize=(14, 7))
        for stock in self.stocks_in_data:
            stock_selection = selected_stocks[stock][selected_stocks[stock] == 1].index
            plt.scatter(stock_selection, [stock] * len(stock_selection), marker='o', label=f'{stock} Selected')

        plt.title('Selected Stocks Over Time')
        plt.xlabel('Date')
        plt.ylabel('Stock')
        plt.grid(True)
        plt.show()

    def plot_returns_time(self):
        fig, ax = plt.subplots(figsize=(14, 7))
        self.cumulative_returns.plot(ax=ax, label='Strategy Cumulative Returns')

        for date in self.cumulative_returns.index:
            if pd.isna(self.cumulative_returns.loc[date]):
                continue
            selected_stocks = self.signal_data.loc[date][self.signal_data.loc[date] == 1].index
            if not selected_stocks.empty:
                ax.annotate(', '.join(selected_stocks), (date, self.cumulative_returns.loc[date]),
                            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=7)

        ax.set_title('Cumulative Returns with Selected Stocks')
        ax.set_xlabel('Date')
        ax.set_ylabel('Log Return')
        ax.legend()
        ax.grid(True)
        plt.show()

    def _autolabel(self, ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    def get_top_stocks_for_date(self, date, top_n=None):
        top_n = top_n or self.top_n
        idx = self.data.index.get_loc(date)
        if idx < self.period_length:
            raise ValueError("Not enough history for top selection.")
        prices = self.data.xs(self.price_type, level=1, axis=1)
        prev = prices.iloc[idx - self.period_length]
        curr = prices.iloc[idx]
        returns = np.log(curr / prev)
        return returns.nlargest(top_n).index.tolist()
