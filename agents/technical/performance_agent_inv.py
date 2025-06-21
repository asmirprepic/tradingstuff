from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PerformanceBasedAgentInv(TradingAgent):
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
    self.algorithm_name = "PerformanceBasedInvAgent"
    self.period_length = period_length
    self.top_n = top_n
    self.holding_period = holding_period
    self.price_type = 'Close'
    self.stocks_in_data = self.data.columns.get_level_values(0).unique()

    self.generate_signal_strategy()
    #self.calculate_returns()

  def generate_signal_strategy(self):
    # Pre-allocate DataFrames
    returns = pd.DataFrame(index=self.data.index, columns=self.stocks_in_data, dtype=np.float64)
    signals = pd.DataFrame(index=self.data.index, columns=self.stocks_in_data, dtype=np.int8)

    # Calculate returns for all stocks
    returns = np.log(self.data.xs('Close', level=1, axis=1).shift(-self.period_length) /
                     self.data.xs('Close', level=1, axis=1))

    # Rank stocks based on returns in descending order for each period
    ranked_returns = returns.rank(axis=1, method='first')

    # Generate signals based on top N stocks
    for date in ranked_returns.index:
        if pd.notna(ranked_returns.loc[date]).all():
            top_stocks = ranked_returns.loc[date][ranked_returns.loc[date] <= self.top_n].index
            signals.loc[date, top_stocks] = 1
            signals.loc[date, ~signals.columns.isin(top_stocks)] = 0
        else:
            signals.loc[date, :] = 0

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

    self.signal_data = holding_signals

  def calculate_returns(self):
    # Defragment self.data and its index
    self.data = self.data.copy()
    self.data.index = self.data.index.copy()

    # Initialize the portfolio value with a default value
    portfolio_value = pd.Series(1.0, index=self.data.index, dtype=np.float64)

    # Pre-allocate selection_log with default values
    self.selection_log = pd.DataFrame(
        {
            'Selected_Stocks': pd.Series([[]] * len(self.data.index), dtype='object'),  # Ensure object type
            'Cumulative_Prices': [0.0] * len(self.data.index),
            'Log_Return': [0.0] * len(self.data.index),
        },
        index=self.data.index,
    )

    # Pre-allocate daily_returns
    close_prices = self.data.xs('Close', level=1, axis=1)  # Extract Close prices
    daily_returns = close_prices.pct_change().fillna(0)  # Compute daily returns

    # Pre-allocate portfolio_log_returns
    portfolio_log_returns = pd.Series(0.0, index=self.data.index, dtype=np.float64)
    selected_stocks = []

    for i in range(1, len(daily_returns)):
        date = daily_returns.index[i]
        prev_date = daily_returns.index[i - 1]

        if i % self.holding_period == 0 or i == 1:
            selected_stocks = list(self.signal_data.loc[prev_date][self.signal_data.loc[prev_date] == 1].index)

        if len(selected_stocks) > 0:
            # Validate stock selections
            initial_prices = self.data.loc[prev_date, [(stock, 'Close') for stock in selected_stocks]].values
            current_prices = self.data.loc[date, [(stock, 'Close') for stock in selected_stocks]].values

            if len(initial_prices) != len(current_prices):
                raise ValueError("Initial and current prices lengths do not match!")

            portfolio_return = (current_prices.sum() - initial_prices.sum()) / initial_prices.sum()
            portfolio_log_returns.loc[date] = np.log(1 + portfolio_return)

            # Update selection_log
            self.selection_log.at[date, 'Selected_Stocks'] = selected_stocks  # Store list as object
            self.selection_log.at[date, 'Cumulative_Prices'] = current_prices.sum()
            self.selection_log.at[date, 'Log_Return'] = portfolio_log_returns.loc[date]
        else:
            portfolio_log_returns.loc[date] = 0
            self.selection_log.at[date, 'Selected_Stocks'] = []  # Empty list as object
            self.selection_log.at[date, 'Cumulative_Prices'] = 0
            self.selection_log.at[date, 'Log_Return'] = 0

    # Calculate cumulative log returns
    cumulative_log_returns = portfolio_log_returns.cumsum()
    self.cumulative_returns = cumulative_log_returns




  def plot_signals(self):
    """
    Plot the stock prices and trading signals.
    """
    plt.figure(figsize=(14, 7))
    for stock in self.stocks_in_data:
        plt.plot(self.data.index, self.data[(stock, self.price_type)], label=f'{stock} Price')
        buy_signals = self.signal_data[self.signal_data[stock] == 1].index
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
    selected_stocks = self.signal_data[self.signal_data == 1]
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
        selected_stocks = self.signal_data.loc[date][self.signal_data.loc[date] == 1].index
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
