from agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MovingAverageAgent(TradingAgent):
    """
    A trading agent that generates trading signals based on when the stock price crosses 
    the long-term moving average from below. The agent only takes long positions.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock prices or relevant trading data.
        back_length (int, optional): The number of periods to calculate the moving average. Defaults to 20.
    """

    def __init__(self, data, back_length=20):
        super().__init__(data)
        self.algorithm_name = "MovingAverage"
        self.back_length = back_length
        self.price_type = 'Close'

        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        
        # Handle multiple stocks
        
        for stock in self.stocks_in_data:
          self.generate_signal_strategy(stock)

        self.calculate_returns()

    def generate_signal_strategy(self, stock):
        """
        Generates trading signals for the specified stock based on when the stock price crosses
        the long-term moving average from below. Only long positions are considered.

        Args:
            stock (str): The stock symbol for which to generate signals.
        """
        signals = pd.DataFrame(index=self.data.index)
        close_price = self.data[(stock,'Close')]
        signals['price'] = close_price
        signals['SMA'] = close_price.rolling(window=self.back_length).mean()

        
        # Generate signals for only long positions
        signals['Position'] = 0
        signals.loc[(signals['price'] > signals['SMA']), 'Position'] = 1  # Long position
        signals.loc[(signals['price'] < signals['SMA']), 'Position'] = 0  # Exit position
        
        signals['Signal'] = signals['Position'].diff()
        signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))

        # Forward fill positions; maintain position until explicitly changed
        signals['Position'] = signals['Position'].ffill()
        
        
        signals['buy'] = (signals['Signal']==1) & (signals['Signal'].shift(1)==0)
        signals['sell'] = (signals['Signal']==0) & (signals['Signal'].shift(1)==1)

        signals['hold_start'] = signals['buy'].cumsum()
        signals['hold_end'] = signals['sell'].cumsum()

        signals['position'] = (signals['hold_start']>signals['hold_end']).astype(int)
        
        
        # Determine when the price crosses the SMA
        #condition = (self.data[stock].iloc[self.back_length:] > signals['SMA'].iloc[self.back_length:]).values
        
        
        self.signal_data[stock] = signals

    def plot(self,stock):
      fig,ax = super().plot(stock)
      sma = self.data[(stock, 'Close')].rolling(window=self.back_length).mean()
      ax.plot(self.data.index, sma, label=f'{self.back_length}-Day SMA', color='orange')
      #plt.plot(self.data.index,self.signal_data[(stock,'Close']['position'])
      plt.legend()
      return fig,ax



