from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np

class MomentumAgent(TradingAgent):
    """
    A trading agent that generates trading signals based on momentum indicators.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock prices or relevant trading data.
        back_length (int, optional): The number of periods to average returns for the momentum signal. Defaults to 1.
    """

    def __init__(self, data, back_length=1, lookbacks=None, long_only: bool = False, score_mode: str = "z"):
        super().__init__(data)
        self.algorithm_name = "Momentum"

        if lookbacks is None:
            if isinstance(back_length, (list, tuple, set, np.ndarray)):
                lookbacks = list(back_length)
            else:
                lookbacks = [back_length]

        lookbacks = [int(x) for x in lookbacks if int(x) > 0]
        if not lookbacks:
            lookbacks = [1]

        self.lookbacks = sorted(set(lookbacks))
        self.back_length = int(self.lookbacks[0])  # backward compatibility / single-window alias
        self.long_only = bool(long_only)

        score_mode = str(score_mode).lower().strip()
        if score_mode not in {"z", "raw"}:
            raise ValueError("score_mode must be 'z' or 'raw'")
        self.score_mode = score_mode

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
        
        mom_cols = []
        z_cols = []
        for lb in self.lookbacks:
            mom_col = f"Momentum_{lb}"
            z_col = f"MomentumZ_{lb}"

            mom = signals['return'].rolling(lb).mean()
            vol = signals['return'].rolling(lb).std(ddof=0).replace(0, np.nan)
            z = mom / vol

            signals[mom_col] = mom
            signals[z_col] = z
            mom_cols.append(mom_col)
            z_cols.append(z_col)

        # Keep a stable 'Momentum' column for downstream code (use equal-weight ensemble mean)
        if len(mom_cols) == 1:
            signals['Momentum'] = signals[mom_cols[0]]
        else:
            signals['Momentum'] = signals[mom_cols].mean(axis=1)

        # SignalStrength is what you should sort/filter by for "today" recommendations
        if self.score_mode == "raw":
            signals['SignalStrength'] = signals['Momentum']
        else:
            signals['SignalStrength'] = signals[z_cols].mean(axis=1) if z_cols else signals['Momentum']

        # Position 1 for positive momentum, -1 for negative, 0 otherwise (or long-only clamp)
        if self.long_only:
            signals['Position'] = np.where(signals['SignalStrength'] > 0, 1, 0)
        else:
            signals['Position'] = np.where(signals['SignalStrength'] > 0, 1, np.where(signals['SignalStrength'] < 0, -1, 0))
        # Signals change isn position buy /sell signals
        signals['Signal'] = signals['Position'].diff().fillna(0).astype(int)
        signals['Signal'] = signals['Signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # Store the signals 
        self.signal_data[stock] = signals

   
        
