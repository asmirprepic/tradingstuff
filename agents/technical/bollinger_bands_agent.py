from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BollingerBandsAgent(TradingAgent):
  """
  A trading agent that generates trading signals based on Bollinger Bands. 
  Bollinger Bands are a type of statistical chart characterizing the prices and 
  volatility over time of a financial instrument or commodity, using a formulaic 
  method employing standard deviations from a moving average.

  Args:
      data (pd.DataFrame): A DataFrame containing the stock prices. It should have a 
                            MultiIndex of (stock_symbol, data_type) where data_type includes 'Close'.
      period (int, optional): The period for calculating the Simple Moving Average (SMA) and 
                              standard deviation. Defaults to 20.
      num_std_dev (float, optional): The number of standard deviations to determine the upper 
                                      and lower bands. Defaults to 2.

  Attributes:
      algorithm_name (str): Name of the algorithm, set to "BollingerBands".
      period (int): The period over which the SMA and standard deviation are calculated.
      num_std_dev (float): The number of standard deviations for the bands.
      signal_data (dict): A dictionary to store signal data for each stock.
  """

  def __init__(self, data, period=20, num_std_dev=2):
      super().__init__(data)
      self.algorithm_name = "BollingerBands"
      self.period = period
      self.num_std_dev = num_std_dev
      self.stocks_in_data = self.data.columns.get_level_values(0).unique()

      for stock in self.stocks_in_data:
          self.generate_signal_strategy(stock)
      self.calculate_returns()

  def generate_signal_strategy(self, stock):
      """
      Generates trading signals for the specified stock based on Bollinger Bands. 
      Buy signals are generated when the stock price crosses below the lower band, 
      and sell signals are generated when the stock price crosses above the upper band.

      Args:
          stock (str): The stock symbol for which to generate signals.

      The method updates the `signal_data` attribute with signals for the given stock.
      """
      signals = pd.DataFrame(index=self.data.index)
      close_price = self.data[(stock, 'Close')]

      # Calculate Bollinger Bands
      signals['SMA'] = close_price.rolling(window=self.period).mean()
      signals['std_dev'] = close_price.rolling(window=self.period).std()
      signals['lower_band'] = signals['SMA'] - (signals['std_dev'] * self.num_std_dev)
      signals['upper_band'] = signals['SMA'] + (signals['std_dev'] * self.num_std_dev)

      # Generate signals based on Bollinger Bands
      signals['Position'] = np.nan
      signals.loc[close_price < signals['lower_band'], 'Position'] = 1  # Buy signal
      signals.loc[close_price > signals['upper_band'], 'Position'] = -1 # Sell signal

      # Neutral position when RSI is between the upper and lower bands
      signals.loc[(close_price <= signals['upper_band']) & (close_price >= signals['lower_band']), 'Position'] = 0

      # Forward fill positions to maintain until explicitly changed
      signals['Position'] = signals['Position'].ffill().fillna(0)

      # Calculate signal as the change in position
      signals['Signal']=0
      signals.loc[signals['Position']>signals['Position'].shift(1),'Signal'] = 1
      signals.loc[signals['Position']<signals['Position'].shift(1),'Signal'] = -1

      signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))


      self.signal_data[stock] = signals


