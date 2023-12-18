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
        signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))
        signals['Position'] = np.sign(signals['return'].rolling(self.back_length).mean())
        signals['Signal'] = signals['Position'].diff()

        
        self.signal_data[stock] = signals



        