from agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np

class MomentumAgent(TradingAgent):
    """
    A trading agent that generates trading signals based on momentum indicators.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock prices or relevant trading data.
        back_length (int, optional): The number of periods to average returns for the momentum signal. Defaults to 1.
    """

    def __init__(self,data,back_length = 1):
        super().__init__(data)
        self.algorithm_name = "Momentum"
        self.back_length = back_length
        self.price_type = 'Close'
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
    

      
        # Handle multiple stocks
        for stock in self.stocks_in_data:
          self.generate_signal_strategy(stock)
        
        self.calculate_returns()


    def generate_signal_strategy(self,stock):
        """
        Generates trading signals for the specified stock based on momentum indicators.

        Args:
            stock (str): The stock symbol for which to generate signals.
        """

        signals = pd.DataFrame(index = self.data.index)
        # Log return calculation
        signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))
        
        # Momentum as a rolling average of returns
        signals['Momentum'] = signals['return'].rolling(self.back_length).mean()

        # Position 1 for positive mometum, -1 for negative , 0 otherwise
        signals['Position'] = np.where(signals['Momentum'] > 0, 1, np.where(signals['Momentum'] < 0, -1, 0))
        # Signals change isn position buy /sell signals
        signals['Signal'] = signals['Position'].diff().fillna(0).astype(int)
        signals['Signal'] = signals['Signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # Store the signals 
        self.signal_data[stock] = signals

   
        
