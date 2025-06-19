from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MeanReversionAgent(TradingAgent):
  """
  A trading agent based on a mean reversion strategy using z-scores.

  Attributes:
      lookback_period (int): Window used to compute rolling mean and std.
      threshold (float): Z-score threshold for entering trades.
      price_type (str): Price column to use.
      auto_generate (bool): Whether to auto-generate signals and returns at init.
  """

  def __init__(self, data, lookback_period=20, threshold=2.0, price_type='Close', auto_generate=True):
      super().__init__(data)
      self.algorithm_name = "MeanReversion"
      self.lookback_period = lookback_period
      self.threshold = threshold
      self.price_type = price_type
      self.stocks_in_data = self.data.columns.get_level_values(0).unique()

      if auto_generate:
          for stock in self.stocks_in_data:
              self.generate_signal_strategy(stock)
          self.calculate_returns()

  def generate_signal_strategy(self, stock):
      """
      Generates trading signals based on z-score thresholds for mean reversion.

      Args:
          stock (str): Stock symbol.
      """
      price = self.data[(stock, self.price_type)]
      signals = pd.DataFrame(index=price.index)
      signals['price'] = price
      signals['mean'] = price.rolling(window=self.lookback_period).mean()
      signals['std'] = price.rolling(window=self.lookback_period).std()
      signals['z_score'] = (price - signals['mean']) / signals['std']

      # Position: long if price is very low vs mean, short if very high, otherwise flat
      signals['Position'] = np.where(signals['z_score'] < -self.threshold, 1,
                                      np.where(signals['z_score'] > self.threshold, -1, 0))
      signals['Signal'] = signals['Position'].diff().fillna(0).astype(int)

      # Forward-fill position to simulate holding
      signals['Position'] = signals['Position'].replace(to_replace=0, method='ffill').fillna(0)

      # Calculate returns
      signals['return'] = np.log(price / price.shift(1))

      self.signal_data[stock] = signals

  def plot(self, stock):
      """
      Plot the price with mean reversion bands and signals.

      Args:
          stock (str): Stock symbol.
      """
      fig, ax = super().plot(stock)

      signals = self.signal_data[stock]
      price = self.data[(stock, self.price_type)]


      ax.plot(signals.index, signals['mean'], label='Rolling Mean', linestyle='--')
      upper = signals['mean'] + self.threshold * signals['std']
      lower = signals['mean'] - self.threshold * signals['std']
      ax.fill_between(signals.index, lower, upper, color='gray', alpha=0.2, label=f'±{self.threshold}σ Band')

      ax.set_title(f'{stock} - Mean Reversion Strategy')
      ax.legend()
      plt.show()

      return fig, ax
