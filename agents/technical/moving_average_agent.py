from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class MovingAverageAgent(TradingAgent):
    """
    Long-only SMA crossover agent.

    Attributes:
        short_window (int): Window size for short-term moving average.
        long_window (int): Window size for long-term moving average.
        price_type (str): The price column to use ('Close' by default).
        auto_generate (bool): Whether to auto-generate signals on initialization.
    """

    def __init__(self, data, short_window=50, long_window=200, price_type='Close', auto_generate=True):
        super().__init__(data)
        self.algorithm_name = "MovingAverage"
        # Used by TradingAgent.score_now() for ranking "today" recommendations.
        self.score_column = "SignalStrength"

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
        signals["price"] = close_price
        signals["SMA_short"] = close_price.rolling(window=int(self.short_window)).mean()
        signals["SMA_long"] = close_price.rolling(window=int(self.long_window)).mean()

        # Position: long when short SMA above long SMA, else flat (long-only).
        signals["Position"] = (signals["SMA_short"] > signals["SMA_long"]).astype(int)

        # Signal: discrete trade events derived from Position changes (-1/0/1).
        sig = signals["Position"].diff().fillna(0).astype(int)
        signals["Signal"] = sig.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # Calculate return
        signals["return"] = np.log(close_price / close_price.shift(1))

        # Strength score for ranking: normalized SMA spread.
        signals["SignalStrength"] = (signals["SMA_short"] - signals["SMA_long"]) / signals["SMA_long"]

        signals["buy"] = (signals["Signal"] == 1)
        signals["sell"] = (signals["Signal"] == -1)

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
        sma_short = close_price.rolling(window=int(self.short_window)).mean()
        sma_long = close_price.rolling(window=int(self.long_window)).mean()

        ax.plot(
            self.data.index,
            sma_short,
            label=f"{self.short_window}-Day SMA",
            color="blue",
            linestyle="--",
            linewidth=1.5,
        )
        ax.plot(
            self.data.index,
            sma_long,
            label=f"{self.long_window}-Day SMA",
            color="orange",
            linestyle="--",
            linewidth=1.5,
        )

        plt.legend()
        return fig, ax
