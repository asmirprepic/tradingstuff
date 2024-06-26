from agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MeanReversionAgent(TradingAgent):
  """
  A trading agent that generates trading signals based on mean reversion.

  Args:
      data (pd.DataFrame): A DataFrame containing the stock prices or relevant trading data.
      lookback_period (int, optional): The number of periods to look back for calculating the mean. Defaults to 20.
      threshold (float, optional): The threshold for detecting overbought or oversold conditions. Defaults to 2.
  """

  def __init__(self, data, lookback_period=20, threshold=2):
    super().__init__(data)
    self.algorithm_name = "MeanReversion"
    self.lookback_period = lookback_period
    self.threshold = threshold
    self.price_type = 'Close'
    self.stocks_in_data = self.data.columns.get_level_values(0).unique()

    # Initialize signal_data dictionary to store signals for each stock
    self.signal_data = {}

    # Handle multiple stocks
    for stock in self.stocks_in_data:
        self.generate_signal_strategy(stock)
    
    self.calculate_returns()

  def generate_signal_strategy(self, stock):
    """
    Generates trading signals for the specified stock based on mean reversion.

    Args:
        stock (str): The stock symbol for which to generate signals.
    """
    signals = pd.DataFrame(index=self.data.index)
    signals['price'] = self.data[(stock, 'Close')]
    signals['mean'] = signals['price'].rolling(window=self.lookback_period).mean()
    signals['std'] = signals['price'].rolling(window=self.lookback_period).std()
    signals['z_score'] = (signals['price'] - signals['mean']) / signals['std']
    signals['Position'] = np.where(signals['z_score'] > self.threshold, -1, 
                                    np.where(signals['z_score'] < -self.threshold, 1, 0))
    signals['Signal'] = signals['Position'].diff().fillna(0).astype(int)

    self.signal_data[stock] = signals

  def plot(self, stock):
    fig, ax = super().plot(stock)
    
    ax.plot(self.data.index, self.data[(stock, 'Close')], label='Close Price')
    signals = self.signal_data[stock]
    ax.plot(signals.index, signals['mean'], label='Mean', linestyle='--')
    ax.fill_between(signals.index, signals['mean'] - self.threshold * signals['std'], 
                    signals['mean'] + self.threshold * signals['std'], color='gray', alpha=0.2)
    
    ax.set_title(f'{stock} - Mean Reversion Strategy')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.show()

    return fig, ax
