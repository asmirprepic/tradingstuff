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
    self.period_length = period_length
    self.top_n = top_n
    self.holding_period = holding_period
    self.price_type = 'Close'
    self.stocks_in_data = self.data.columns.get_level_values(0).unique()

    self.generate_signal_strategy()
    self.calculate_returns()

  def generate_signal_strategy(self):
    # Calculate returns over the specified period length
    returns = pd.DataFrame(index=self.data.index)
    for stock in self.stocks_in_data:
        returns[stock] = np.log(self.data[(stock, 'Close')].shift(-self.period_length) / self.data[(stock, 'Close')])

    # Shift returns to avoid lookahead bias
    returns = returns.shift(1)

    # Rank stocks based on returns in descending order for each period
    ranked_returns = returns.rank(axis=1, method='first', ascending=False)

    # Generate signals based on top N stocks
    signals = pd.DataFrame(index=self.data.index)
    for date in ranked_returns.index:
        if pd.notna(ranked_returns.loc[date]).all():
            top_stocks = ranked_returns.loc[date][ranked_returns.loc[date] <= self.top_n].index
            for stock in self.stocks_in_data:
                signals.loc[date, stock] = 1 if stock in top_stocks else 0
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
                holding_signals.iloc[i + j] = 0
                holding_signals.iloc[i + j][top_stocks] = 1

    self.signal_data = holding_signals

  def calculate_returns(self):
    # Initialize the portfolio value
    portfolio_value = pd.Series(index=self.data.index, dtype=np.float64)
    portfolio_value.iloc[0] = 1  # Start with an initial value of 1

    # DataFrame to store selected stocks, cumulative prices, and log returns
    self.selection_log = pd.DataFrame(index=self.data.index, columns=['Selected_Stocks', 'Cumulative_Prices', 'Log_Return'])

    # Calculate the daily returns of each stock
    daily_returns = pd.DataFrame(index=self.data.index)
    for stock in self.stocks_in_data:
        daily_returns[stock] = self.data[(stock, 'Close')].pct_change().fillna(0)

    # Calculate portfolio log returns based on daily signals
    portfolio_log_returns = pd.Series(index=self.data.index, dtype=np.float64)
    selected_stocks = []

    for i in range(1, len(daily_returns)):
        date = daily_returns.index[i]
        prev_date = daily_returns.index[i - 1]

        if i % self.holding_period == 0 or i == 1:
          selected_stocks = self.signal_data.loc[prev_date][self.signal_data.loc[prev_date] == 1].index
        
        if len(selected_stocks) > 0:
            # Calculate the combined return for the selected stocks
            initial_prices = self.data.loc[prev_date, [(stock, 'Close') for stock in selected_stocks]].values
            current_prices = self.data.loc[date, [(stock, 'Close') for stock in selected_stocks]].values
            portfolio_return = (current_prices.sum() - initial_prices.sum()) / initial_prices.sum()
            portfolio_log_returns[date] = np.log(1 + portfolio_return)

            # Save selection log details
            self.selection_log.at[date, 'Selected_Stocks'] = selected_stocks.tolist()
            self.selection_log.at[date, 'Cumulative_Prices'] = current_prices.sum()
            self.selection_log.at[date, 'Log_Return'] = portfolio_log_returns[date]
        else:
            portfolio_log_returns[date] = 0
            self.selection_log.at[date, 'Selected_Stocks'] = []
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
