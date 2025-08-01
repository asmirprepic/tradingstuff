from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np

class VolumePriceDivergenceAgent(TradingAgent):
    """
    A trading agent that generates signals based on volume-price divergence.

    Buys when price drops with decreasing volume (weak selling pressure).
    Sells when price rises with decreasing volume (weak buying pressure).

    Args:
        data (pd.DataFrame): MultiIndex (stock, 'Close'/'Volume') price data.
        window (int): Number of periods for computing change.
        threshold (float): Minimum absolute price change to trigger a signal.
    """

    def __init__(self, data, window=2, threshold=0.005):
        super().__init__(data)
        self.algorithm_name = 'VolumePriceDivergence'
        self.window = window
        self.threshold = threshold
        self.price_type = 'Close'
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock)

        self.calculate_returns()

    def generate_signal_strategy(self, stock):
        """
        Generate trading signals for the specified stock based on volume-price divergence.
        """
        signals = pd.DataFrame(index=self.data.index)

        price = self.data[(stock, 'Close')]
        volume = self.data[(stock, 'Volume')]

        price_change = price.pct_change(self.window)
        volume_change = volume.pct_change(self.window)

        signals['price_change'] = price_change
        signals['volume_change'] = volume_change

        signals['Position'] = np.nan

        # Buy signal: price falls + volume falls
        buy_cond = (price_change < -self.threshold) & (volume_change < 0)
        signals.loc[buy_cond, 'Position'] = -1

        # Sell signal: price rises + volume falls
        sell_cond = (price_change > self.threshold) & (volume_change < 0)
        signals.loc[sell_cond, 'Position'] = 1

        # Entry signals
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        # Daily log return
        signals['return'] = np.log(price / price.shift(1))

        self.signal_data[stock] = signals
