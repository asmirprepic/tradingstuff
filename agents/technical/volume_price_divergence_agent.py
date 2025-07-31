import pandas as pd
from agents.base_agents.trading_agent import TradingAgent


class VolumePriceDivergenceAgent(TradingAgent):
    def __init__(self, window: int = 2, threshold: float = 0.0):
        """
        Volume-Price Divergence strategy.
        - window: number of days over which to compute price and volume change.
        - threshold: minimum absolute divergence to act.
        """
        super().__init__()
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on volume-price divergence.

        Parameters:
        - data: DataFrame with columns ['Close', 'Volume']

        Returns:
        - pd.Series with signals: 1 for buy, -1 for sell, 0 for hold
        """
        df = data.copy()

        # Compute returns and volume change
        df['price_return'] = df['Close'].pct_change(self.window)
        df['volume_change'] = df['Volume'].pct_change(self.window)

        # Signal: look for opposite signs (divergence)
        df['signal'] = 0

        # Buy: price down, volume down → possible oversold
        buy_cond = (df['price_return'] < -self.threshold) & (df['volume_change'] < 0)
        df.loc[buy_cond, 'signal'] = 1

        # Sell: price up, volume down → possible weak breakout
        sell_cond = (df['price_return'] > self.threshold) & (df['volume_change'] < 0)
        df.loc[sell_cond, 'signal'] = -1

        return df['signal']
