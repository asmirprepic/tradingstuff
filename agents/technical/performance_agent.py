from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PerformanceBasedAgent(TradingAgent):
  """
  A trading agent that buys the 5 stocks with the highest returns over the last 5 time periods
  and holds them for a specified holding period.

  Args:
      data (pd.DataFrame): A DataFrame containing the stock prices or relevant trading data.
      period_length (int, optional): The number of periods to consider for calculating returns. Defaults to 5.
      top_n (int, optional): The number of top stocks to select based on returns. Defaults to 5.
      holding_period (int, optional): The number of periods to hold the selected stocks. Defaults to 20.
  """

  def __init__(self, data, period_length=5, top_n=5, holding_period=20):
    super().__init__(data)
    self.algorithm_name = "PerformanceBasedAgent"
    # Rank "today" candidates by the same lookback return used for selection.
    self.score_column = "LookbackReturn"
    self.period_length = period_length
    self.top_n = top_n
    self.holding_period = holding_period
    self.price_type = 'Close'
    self.stocks_in_data = self.data.columns.get_level_values(0).unique()

    self.generate_signal_strategy()
    #self.calculate_returns()

  def generate_signal_strategy(self):
    close = self.data.xs('Close', level=1, axis=1)

    # Returns used for selection/ranking (lookback performance)
    lookback_returns = np.log(close / close.shift(self.period_length))

    # 1-period log returns (used by the base TradingAgent return helpers / latest_row NaN filtering)
    one_period_returns = np.log(close / close.shift(1))

    signals = pd.DataFrame(index=self.data.index, columns=self.stocks_in_data, dtype=np.int8)

    # Rank stocks based on returns in descending order for each period
    ranked_returns = lookback_returns.rank(axis=1, method='first', ascending=False)

    # Generate signals based on top N stocks
    for date in ranked_returns.index:
        r = lookback_returns.loc[date].dropna()
        if len(r) < self.top_n:
            signals.loc[date,:] = 0
        else:
            top = r.nlargest(self.top_n).index
            signals.loc[date,:] = 0
            signals.loc[date, top] = 1
        # if pd.notna(ranked_returns.loc[date]).all():
        #     top_stocks = ranked_returns.loc[date][ranked_returns.loc[date] <= self.top_n].index
        #     signals.loc[date, top_stocks] = 1
        #     signals.loc[date, ~signals.columns.isin(top_stocks)] = 0
        # else:
        #     signals.loc[date, :] = 0

    # Apply holding period to signals
    holding_signals = signals.copy()
    for i in range(0, len(signals), self.holding_period):
        period_signals = signals.iloc[i:i + self.holding_period]
        if period_signals.empty:
            continue
        top_stocks = period_signals.iloc[0][period_signals.iloc[0] == 1].index
        for j in range(self.holding_period):
            if i + j < len(signals):
                holding_signals.iloc[i + j, holding_signals.columns.isin(top_stocks)] = 1
                holding_signals.iloc[i + j, ~holding_signals.columns.isin(top_stocks)] = 0

    # Keep the original "matrix" form for plotting / portfolio-return helpers in this class.
    self.holdings_matrix = holding_signals

    # Also expose per-stock signal DataFrames to match the TradingAgent interface used by
    # `recommendations_from_agent(...)` and `TradingAgent.action_now(...)`.
    self.signal_data = {}
    for stock in self.stocks_in_data:
        df = pd.DataFrame(index=self.data.index)
        df["return"] = one_period_returns[stock]
        df["LookbackReturn"] = lookback_returns[stock]
        df["Rank"] = ranked_returns[stock]

        pos = holding_signals[stock].astype(int)
        df["Position"] = pos
        df["Signal"] = pos.diff().fillna(0).astype(int)
        df["Signal"] = df["Signal"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        self.signal_data[stock] = df

  def calculate_returns(self):
    # Prices (Date x Stock)
    prices = self.data.xs('Close', level=1, axis=1).copy()

    # Daily *log* returns per stock (Date x Stock)
    stock_logret = np.log(prices / prices.shift(1))

    # Holdings matrix (Date x Stock), 0/1. Align to prices index.
    hold = self.holdings_matrix.reindex(prices.index).fillna(0).astype(float)

    # Use yesterday's holdings for today's return (avoids same-bar lookahead)
    hold_lag = hold.shift(1).fillna(0)

    # Equal-weight portfolio log return each day:
    # sum(logret * hold_lag) / number_of_held_stocks
    n_held = hold_lag.sum(axis=1).replace(0, np.nan)
    port_logret = (stock_logret * hold_lag).sum(axis=1) / n_held
    port_logret = port_logret.fillna(0.0)

    # Store outputs
    self.portfolio_log_returns = port_logret
    self.cumulative_returns = port_logret.cumsum()

    # Optional: selection log (kept similar to what you had, but now meaningful)
    self.selection_log = pd.DataFrame(
        {
            "Selected_Stocks": hold.apply(lambda row: list(row.index[row.values == 1]), axis=1),
            "Log_Return": port_logret,
            "Cum_Log_Return": self.cumulative_returns,
            "N_Held": hold.sum(axis=1).astype(int),
        },
        index=prices.index,
    )




  def plot_signals(self):
    """
    Plot the stock prices and trading signals.
    """
    plt.figure(figsize=(14, 7))
    for stock in self.stocks_in_data:
        plt.plot(self.data.index, self.data[(stock, self.price_type)], label=f'{stock} Price')
        buy_signals = self.holdings_matrix[self.holdings_matrix[stock] == 1].index
        plt.scatter(buy_signals, self.data[(stock, self.price_type)].loc[buy_signals], marker='^', color='g', label=f'{stock} Buy Signal')

    plt.title('Stock Prices and Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

  def plot_returns(self):
    fig, ax = plt.subplots(figsize=(14, 7))
    strategy_return = self.cumulative_returns.iloc[-1] * 100  # Convert to percentage
    buy_and_hold_return = (self.data.xs(self.price_type, level=1, axis=1).iloc[-1] / self.data.xs(self.price_type, level=1, axis=1).iloc[0] - 1).mean() * 100  # Convert to percentage

    bars1 = ax.bar(['Strategy Return'], [strategy_return], width=0.4, label='Strategy Return')
    bars2 = ax.bar(['Buy and Hold Return'], [buy_and_hold_return], width=0.4, label='Buy and Hold Return')

    ax.set_ylabel('Returns (%)')
    ax.set_title('Strategy Returns vs Buy and Hold Returns')
    ax.legend()

    self._autolabel(ax, bars1)
    self._autolabel(ax, bars2)

    fig.tight_layout()
    plt.show()

  def _autolabel(self, ax, bars):
    """
    Attach a text label above each bar in *bars*, displaying its height.
    """
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

  def plot_selected_stocks(self):
    selected_stocks = self.holdings_matrix[self.holdings_matrix == 1]
    plt.figure(figsize=(14, 7))
    for stock in self.stocks_in_data:
        stock_selection = selected_stocks[stock][selected_stocks[stock] == 1].index
        plt.scatter(stock_selection, [stock] * len(stock_selection), marker='o', label=f'{stock} Selected')

    plt.title('Selected Stocks Over Time')
    plt.xlabel('Date')
    plt.ylabel('Stock')
    plt.grid(True)
    plt.show()

  def plot_returns_time(self):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot cumulative returns over time
    self.cumulative_returns.plot(ax=ax, label='Strategy Cumulative Returns')

    # Annotate selected stocks at each timepoint
    for date in self.cumulative_returns.index:
        if pd.isna(self.cumulative_returns.loc[date]):
            continue
        selected_stocks = self.holdings_matrix.loc[date][self.holdings_matrix.loc[date] == 1].index
        selected_text = ', '.join(selected_stocks)
        ax.annotate(selected_text, (date, self.cumulative_returns.loc[date]),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.title('Strategy Cumulative Returns with Selected Stocks')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

  def get_top_stocks_for_date(self, date, top_n=3):
      """
      Get the top N stocks based on returns over the last 5 days for a given date.

      Args:
          date (pd.Timestamp): The date for which to select the top stocks.
          top_n (int, optional): The number of top stocks to select. Defaults to 3.

      Returns:
          list: List of top N stocks based on performance over the last 5 days.
      """
      # Ensure the date is valid (not in the first 5 days of data)
      if date not in self.data.index:
          raise ValueError(f"Date {date} not found in the dataset.")
      date_idx = self.data.index.get_loc(date)
      if date_idx < self.period_length:
          raise ValueError(f"Not enough historical data to calculate returns for {date}.")

      # Calculate returns over the last 'period_length' days
      returns = pd.Series(index=self.stocks_in_data, dtype=np.float64)
      for stock in self.stocks_in_data:
          prev_price = self.data.loc[self.data.index[date_idx - self.period_length], (stock, 'Close')]
          current_price = self.data.loc[date, (stock, 'Close')]
          returns[stock] = np.log(current_price / prev_price)

      # Rank the stocks by returns and select the top N
      top_stocks = returns.nlargest(top_n).index.tolist()
      return top_stocks
