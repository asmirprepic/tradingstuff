from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class VWAPAgent(TradingAgent):
  """
    A trading agent that generates trading signals based on the Volume-Weighted Average Price (VWAP).
    The agent considers both long and short positions based on the relationship of the closing price 
    to the VWAP adjusted by a specified threshold.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock prices and volumes. 
                             It should have a MultiIndex of (stock_symbol, data_type) 
                             where data_type includes 'High', 'Low', 'Close', and 'Volume'.
        period (int, optional): The number of periods to calculate the VWAP. Defaults to 20.
        threshold (float, optional): A threshold to determine significant deviation 
                                     from the VWAP for generating signals. Defaults to 0.05.

    Attributes:
        algorithm_name (str): Name of the algorithm, set to "VWAP".
        period (int): The number of periods over which to calculate the VWAP.
        threshold (float): Threshold for determining significant deviation for signals.
        price_type (str): Data type for price, set to 'Close'.
        signal_data (dict): A dictionary to store signal data for each stock.
    """
  
  def __init__(self,data,period = 20, threshold =0.05):
    super().__init__(data)
    self.algorithm_name = "VWAP"
    self.period = period
    self.threshold = threshold
    stocks_in_data = self.data.columns.get_level_values(0).unique()

    for stock in stocks_in_data:
      self.generate_signal_strategy(stock)

    self.calculate_returns()

  def generate_signal_strategy(self,stock):
    """
      Generates trading signals for the specified stock based on the VWAP indicator. 
      The agent enters a long position when the closing price is significantly above 
      the VWAP and a short position when significantly below.

      Args:
          stock (str): The stock symbol for which to generate signals.

      The method updates the `signal_data` attribute with signals for the given stock.
    """
    signals = pd.DataFrame(index = self.data.index)
    price = self.data[(stock,'Close')]
    volume = self.data[(stock,'Volume')]
    high = self.data[(stock,'High')]
    low = self.data[(stock,'Low')]


    # Calculate VMAP
    signals['TP'] = (high+low+price)/3
    signals['VP'] = signals['TP']*volume
    signals['Cumulative_VP'] = signals['VP'].rolling(window = self.period).sum()
    signals['Cumulative_volume'] = volume.rolling(window = self.period).sum()
    signals['VWAP'] = signals['Cumulative_VP']/signals['Cumulative_volume']

    # Generate Signals
    signals['Position'] = np.nan
    signals.loc[price>signals['VWAP']*(1+self.threshold),'Position'] = 1
    signals.loc[price<signals['VWAP']*(1+self.threshold),'Position'] = -1

    # Neutral position when the price is near the VWAP
    signals.loc[(price <= signals['VWAP'] * (1 + self.threshold)) & 
                (price >= signals['VWAP'] * (1 - self.threshold)), 'Position'] = 0

    #forward fill to make sure that once position is entered it is kept
    signals['Position'] = signals['Position'].ffill().fillna(0)

    #Ensure signal is only -1, 0 or 1
    signals['Signal']=0
    signals.loc[signals['Position']>signals['Position'].shift(1),'Signal'] = 1
    signals.loc[signals['Position']<signals['Position'].shift(1),'Signal'] = -1
    signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))

    self.signal_data[stock] = signals







    
    