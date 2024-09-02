from agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RSIAgent(TradingAgent):

  """
    A trading agent that generates trading signals based on the Relative Strength Index (RSI).
    The RSI is a momentum oscillator that measures the speed and change of price movements.
    The agent considers both long and short positions based on the RSI being overbought or oversold.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock prices. 
                             It should have a MultiIndex of (stock_symbol, data_type) 
                             where data_type includes 'Close'.
        period (int, optional): The number of periods to calculate the RSI. Defaults to 14.
        upper_band (float, optional): The upper RSI threshold to indicate overbought conditions. Defaults to 70.
        lower_band (float, optional): The lower RSI threshold to indicate oversold conditions. Defaults to 30.

    Attributes:
        algorithm_name (str): Name of the algorithm, set to "RSI".
        period (int): The number of periods over which to calculate the RSI.
        upper_band (float): Upper RSI threshold for overbought conditions.
        lower_band (float): Lower RSI threshold for oversold conditions.
        price_type (str): Data type for price, set to 'Close'.
        signal_data (dict): A dictionary to store signal data for each stock.
    """
  
  def __init__(self,data,period = 14,upper_band = 70,lower_band =30):
    super().__init__(data)
    self.algorithm_name='RSI'
    self.period = period
    self.upper_band = upper_band
    self.lower_band = lower_band
    self.price_type = 'Close'
    self.stocks_in_data = self.data.columns.get_level_values(0).unique()

    for stock in self.stocks_in_data:
      self.generate_signal_strategy(stock)

    self.calculate_returns()

  def generate_signal_strategy(self,stock):
    """
      Generates trading signals for the specified stock based on the RSI indicator. 
      A long position is considered when RSI is below the lower threshold (oversold), 
      and a short position is considered when RSI is above the upper threshold (overbought).

      Args:
          stock (str): The stock symbol for which to generate signals.

      The method updates the `signal_data` attribute with signals for the given stock.
      """

    signals =pd.DataFrame(index = self.data.index)
    price = self.data[(stock,'Close')]

    # Calculate RSI
    delta = price.diff()
    gain = (delta.where(delta>0,0)).rolling(window = self.period).mean()
    loss = (-delta.where(delta < 0,0)).rolling(window = self.period).mean()

    rs = gain/loss
    signals['RSI'] = 100-(100/(1+rs))

    # Generate signals for long and short positions
    signals['Position'] =np.nan
    signals.loc[signals['RSI']<self.lower_band,'Position'] =1
    signals.loc[signals['RSI']>self.upper_band,'Position'] =-1

    
    # Forward fill positions to maintain until explicitly changed
    #signals['Position'] = signals['Position'].ffill()

    # Ensure Signal is only -1, 0, or 1
    signals['Signal'] = 0
    signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
    signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1
    signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))

    self.signal_data[stock] = signals





    