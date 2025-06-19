from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MovingAverageAgent(TradingAgent):
    """
    A trading agent that generates trading signals based on when the short-term moving average
    crosses the long-term moving average. Only long positions are taken.

    Attributes:
        short_window (int): Window size for short-term moving average.
        long_window (int): Window size for long-term moving average.
        price_type (str): The price column to use ('Close' by default).
        auto_generate (bool): Whether to auto-generate signals on initialization.
    """

    def __init__(self, data, short_window=50, long_window=200, price_type='Close', auto_generate=True):
        super().__init__(data)
        self.algorithm_name = "MovingAverage"
        self.short_window = short_window
        self.long_window = long_window
        self.price_type = price_type
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        if auto_generate:
            for stock in self.stocks_in_data:
                self.generate_signal_strategy(stock)
            self.calculate_returns()

    def generate_signal_strategy(self, stock):
        """
        Generates trading signals for a given stock based on moving average crossover.
        Only long positions are considered.

        Args:
            stock (str): The stock symbol to generate signals for.
        """
        close_price = self.data[(stock, self.price_type)]
        signals = pd.DataFrame(index=close_price.index)
        signals['price'] = close_price
        signals['SMA_short'] = close_price.rolling(window=self.short_window).mean()
        signals['SMA_long'] = close_price.rolling(window=self.long_window).mean()

        # Generate positions
        signals['Position'] = (signals['SMA_short'] > signals['SMA_long']).astype(int)
        signals['Signal'] = signals['Position'].diff()
        signals['Position'] = signals['Position'].ffill()

        # Calculate return
        signals['return'] = np.log(close_price / close_price.shift(1))


        signals['buy'] = (signals['Signal'] == 1)
        signals['sell'] = (signals['Signal'] == -1)

        self.signal_data[stock] = signals

    def plot(self, stock):
        """
        Plots price, moving averages, and buy/sell signals for a given stock.

        Args:
            stock (str): The stock symbol to plot.

        Returns:
            tuple: The matplotlib figure and axis.
        """
        fig, ax = super().plot(stock)
        close_price = self.data[(stock, self.price_type)]
        sma_short = close_price.rolling(window=self.short_window).mean()
        sma_long = close_price.rolling(window=self.long_window).mean()

        ax.plot(self.data.index, sma_short, label=f'{self.short_window}-Day SMA', color='blue', linestyle='--', linewidth=1.5)
        ax.plot(self.data.index, sma_long, label=f'{self.long_window}-Day SMA', color='orange', linestyle='--', linewidth=1.5)

        plt.legend()
        return fig, ax
