from agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MACDAgent(TradingAgent):
    """
    A trading agent that generates trading signals based on the Moving Average Convergence Divergence (MACD).
    MACD is a trend-following momentum indicator that shows the relationship between two moving averages of prices.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock prices. It should have a 
                             MultiIndex of (stock_symbol, data_type) where data_type includes 'Close'.
        short_window (int, optional): The short-term period for the EMA. Defaults to 12.
        long_window (int, optional): The long-term period for the EMA. Defaults to 26.
        signal_window (int, optional): The signal line EMA period. Defaults to 9.

    Attributes:
        algorithm_name (str): Name of the algorithm, set to "MACD".
        short_window (int): Short-term EMA period.
        long_window (int): Long-term EMA period.
        signal_window (int): Signal line EMA period.
        signal_data (dict): A dictionary to store signal data for each stock.
    """

    def __init__(self, data, short_window=12, long_window=26, signal_window=9):
        super().__init__(data)
        self.algorithm_name = "MACD"
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()


        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock)
        self.calculate_returns()

    def generate_signal_strategy(self, stock):
        """
        Generates trading signals for the specified stock based on the MACD indicator.
        A buy signal is generated when the MACD crosses above its signal line, and a sell 
        signal is generated when the MACD crosses below its signal line.

        Args:
            stock (str): The stock symbol for which to generate signals.

        The method updates the `signal_data` attribute with signals for the given stock.
        """
        signals = pd.DataFrame(index=self.data.index)
        close_price = self.data[(stock, 'Close')]

        # Calculate MACD and Signal Line
        signals['EMA_fast'] = close_price.ewm(span=self.short_window, adjust=False).mean()
        signals['EMA_slow'] = close_price.ewm(span=self.long_window, adjust=False).mean()
        signals['MACD'] = signals['EMA_fast'] - signals['EMA_slow']
        signals['Signal_line'] = signals['MACD'].ewm(span=self.signal_window, adjust=False).mean()

        # Generate signals based on MACD
        signals['Position'] = 0
        signals.loc[signals['MACD'] > signals['Signal_line'], 'Position'] = 1  # Buy signal
        signals.loc[signals['MACD'] < signals['Signal_line'], 'Position'] = -1 # Sell signal
        

        # Forward fill positions to maintain until explicitly changed
        signals['Position'] = signals['Position'].ffill()

        # Calculate signal as the change in position
        signals['Signal']=0
        signals.loc[signals['Position']>signals['Position'].shift(1),'Signal'] = 1
        signals.loc[signals['Position']<signals['Position'].shift(1),'Signal'] = -1

        signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))
        
        self.signal_data[stock] = signals
