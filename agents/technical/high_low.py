from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class HighLowAgent(TradingAgent):
  """
  A trading agent that buys stocks at new 50-point highs and sells at new 40-point lows.

  Args:
      data (pd.DataFrame): A DataFrame containing the stock prices or relevant trading data.
  """

  def __init__(self, data, auto_generate=True):
      super().__init__(data)
      self.algorithm_name = "HighLow"
      self.high_lookback = 50  # Look back for 50 points for highs
      self.low_lookback = 40   # Look back for 40 points for lows

      self.stocks_in_data = self.data.columns.get_level_values(0).unique()

      if auto_generate:
          self.run_all()

  def generate_signal_strategy(self, stock):
      """
      Generates trading signals for the specified stock based on new highs and lows.

      Args:
          stock (str): The stock symbol for which to generate signals.
      """
      signals = pd.DataFrame(index=self.data.index)
      price = self.data[(stock,'Close')]
      signals['price'] = price

      # Calculate rolling highs and lows
      signals['rolling_max'] = price.rolling(window=self.high_lookback).max()
      signals['rolling_min'] = price.rolling(window=self.low_lookback).min()

      # Determine buy and sell signals
      signals['buy_signal'] = signals['price'] > signals['rolling_max'].shift(1)
      signals['sell_signal'] = signals['price'] < signals['rolling_min'].shift(1)

      position = pd.Series(0, index=signals.index, dtype=int)
      for i in range(1, len(signals)):
          prev_pos = position.iloc[i - 1]
          if bool(signals['buy_signal'].iloc[i]):
              position.iloc[i] = 1
          elif bool(signals['sell_signal'].iloc[i]):
              position.iloc[i] = -1
          else:
              position.iloc[i] = prev_pos

      signals['Position'] = position

      signals['Signal'] = signals['Position'].diff()
      signals['Signal'] = signals['Signal'].apply(lambda x: max(min(x, 1), -1))

      signals['return'] = np.log(price/price.shift(1))

      # Calculate positions for visualization and trading signals
      signals['position'] = signals['Position']

      self.signal_data[stock] = signals
      return signals

  def run_all(self, mode="backtest"):
      self.signal_data = {}
      for stock in self.stocks_in_data:
          self.generate_signal_strategy(stock)
      self.calculate_returns()

  def plot(self, stock):
      fig, ax = super().plot(stock)
      #plt.scatter(self.data.index, self.signal_data[stock]['price'], label='Price', color='blue')
      #plt.scatter(self.data.index, self.signal_data[stock]['rolling_max'], label=f'50-Day High', color='green')
      #plt.scatter(self.data.index, self.signal_data[stock]['rolling_min'], label=f'40-Day Low', color='red')
      plt.legend()
      return fig, ax
