from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np

class MovingAverageCrossoverAgent(TradingAgent):
    """
    A trading agent that generates trading signals based on moving average crossovers.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock prices or relevant trading data.
        short_window (int, optional): The number of periods for the short moving average. Defaults to 50.
        long_window (int, optional): The number of periods for the long moving average. Defaults to 200.
    """

    def __init__(self, data, short_window=50, long_window=200):
        super().__init__(data)
        self.algorithm_name = "MovingAverageCrossover"
        self.short_window = short_window
        self.long_window = long_window
        self.price_type = 'Close'
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        # Initialize signal_data dictionary to store signals for each stock
        self.signal_data = {}

        # Handle multiple stocks
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock)
        
        # Assuming the base class handles return calculation, 
        # so we don't call self.calculate_returns() here

    def generate_signal_strategy(self, stock):
        """
        Generates trading signals for the specified stock based on moving average crossovers.

        Args:
            stock (str): The stock symbol for which to generate signals.
        """
        signals = pd.DataFrame(index=self.data.index)
        signals['Short_MA'] = self.data[(stock, 'Close')].rolling(window=self.short_window, min_periods=1).mean()
        signals['Long_MA'] = self.data[(stock, 'Close')].rolling(window=self.long_window, min_periods=1).mean()
        signals['Position'] = np.where(signals['Short_MA'] > signals['Long_MA'], 1, -1)
        signals['Signal'] = signals['Position'].diff().fillna(0).astype(int)
        signals['Signal'] = signals['Signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        self.signal_data[stock] = signals

