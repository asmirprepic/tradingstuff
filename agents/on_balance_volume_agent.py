from agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class OBVAgent(TradingAgent):
  """
    A trading agent that generates trading signals based on the On-Balance Volume (OBV) indicator.
    This agent considers both long and short positions based on the relationship of OBV to its 
    Exponential Moving Average (EMA).

    Args:
        data (pd.DataFrame): A DataFrame containing the stock prices and volumes. 
                             It should have a MultiIndex of (stock_symbol, data_type) 
                             where data_type includes 'Close' and 'Volume'.
        threshold (float, optional): A threshold to determine significant deviation 
                                     from the OBV EMA for generating signals. Defaults to 0.05.

    Attributes:
        algorithm_name (str): Name of the algorithm, set to "OBV".
        threshold (float): Threshold for determining significant deviation for signals.
        price_type (str): Data type for price, set to 'Close'.
        signal_data (dict): A dictionary to store signal data for each stock.
    """
  def __init__(self,data,threshold =0.05):
    super().__init__(data)
    self.threshold = threshold
    self.algorithm_name = "OBV"
    self.stocks_in_data = self.data.columns.get_level_values(0).unique()

    for stock in self.stocks_in_data:
      self.generate_signal_strategy(stock)
      
    self.calculate_returns()


  def generate_signal_strategy(self,stock):
    """
      Generates trading signals for the specified stock based on the OBV indicator. 
      The agent enters a long position when OBV is significantly above its EMA and 
      a short position when significantly below.

      Args:
          stock (str): The stock symbol for which to generate signals.

      The method updates the `signal_data` attribute with signals for the given stock.
    """
    signals = pd.DataFrame(index = self.data.index)  
    price = self.data[(stock,'Close')]
    volume = self.data[(stock,'Volume')]

    # Calculate OBV
    signals['OBV'] = (volume*(-price.diff().le(0)*2-1)).cumsum()
    signals['OBV_EMA'] = signals['OBV'].ewm(span =20).mean()

    # Generate signals
    signals['Position'] = np.nan
    signals.loc[signals['OBV']>signals['OBV_EMA']*(1+self.threshold),'Position'] =1
    signals.loc[signals['OBV']<signals['OBV_EMA']*(1-self.threshold),'Position'] =-1
    
    signals.loc[(signals['OBV'] <= signals['OBV_EMA'] * (1 + self.threshold)) & 
                (signals['OBV'] >= signals['OBV_EMA'] * (1 - self.threshold)), 'Position'] = 0
    
    #Forward fill the position
    signals['Position'] = signals['Position'].ffill().fillna(0)

    signals['Signal'] = 0
    signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
    signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1
    signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))
    


    self.signal_data[stock] = signals
