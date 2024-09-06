from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PerformanceBasedAgent(TradingAgent):
  """
    A trading agent that implements a simple momentum-based strategy. The agent selects
    the top N stocks with the highest returns over the past `period_length` time periods,
    generates buy signals, and holds the selected stocks for a fixed holding period. 
    The agent tracks portfolio performance and provides a variety of plotting functions.
  
    The primary focus of this agent is to exploit momentum by buying top-performing 
    stocks and holding them for a short period, with the expectation that the trend 
    will continue.

  Attributes:
      data (pd.DataFrame): Multi-index DataFrame of stock prices with at least 'Close' prices.
      period_length (int): Number of periods to calculate stock returns.
      top_n (int): Number of top-performing stocks to select based on returns.
      holding_period (int): Number of periods to hold the selected stocks before rebalancin.
      price_type (str): Type of stock price to be used for calculations (default is 'Close').
      stocks_in_data (pd.Index): Unique stock symbols from the data.
      signal_data (pd.DataFrame): DataFrame storing buy signals for each stock at each time period.
      selection_log (pd.DataFrame): Logs of selected stocks, cumulative prices, and returns.
      cumulative_returns (pd.Series): Cumulative log returns of the portfolio over time.
    """


  def __init__(self, data, period_length=5, top_n=5, holding_period=20):
    """
    Initializes the PerformanceBasedAgent with stock price data and trading strategy parameters.

    Args:
        data (pd.DataFrame): A multi-index DataFrame with columns representing stock symbols and price types (e.g., 'Close'). 
        The DataFrame should contain price data indexed by time periods.
        period_length (int, optional): The number of periods to use when calculating stock returns.Defaults to 5.
        top_n (int, optional): The number of top stocks to select based on returns. Defaults to 5.
        holding_period (int, optional): The number of periods to hold the selected stocks before rebalancing. Defaults to 20.

    Raises:
        ValueError: If the provided DataFrame does not contain a 'Close' column for stocks.
        
    """
    super().__init__(data)
    self.algorithm_name = "PerformanceBasedAgent"
    self.period_length = period_length
    self.top_n = top_n
    self.holding_period = holding_period
    self.price_type = 'Close'
    # Get stock symbols from the data
    self.stocks_in_data = self.data.columns.get_level_values(0).unique()

    # Generate buy sell signals for the strategy
    self.generate_signal_strategy()
    # Calculate portfolio returns based on the signals
    self.calculate_returns()

  def generate_signal_strategy(self):
    """
    Generates trading signals for the strategy. The strategy selects the top N stocks
    with the highest log returns over the past `period_length` periods, generates a
    buy signal for these stocks, and holds them for `holding_period` periods.

    The method applies the following steps:
    1. Calculate log returns for each stock.
    2. Rank the stocks based on the calculated returns.
    3. Select the top N stocks and generate a buy signal.
    4. Adjust the signals to ensure stocks are held for the designated holding period.

    The resulting buy signals are stored in `self.signal_data`, where each row corresponds 
    to a time period, and each column corresponds to a stock. A value of 1 indicates a buy
    signal for that stock at that time.
    """

    
    # Initialize returns DataFrame to store log returns
    returns = pd.DataFrame(index=self.data.index)
    
    for stock in self.stocks_in_data:
        # Log return: ln(P_t/P_(t-period_length))
        returns[stock] = np.log(self.data[(stock, 'Close')] / self.data[(stock, 'Close')].shift(self.period_length))

    # Shift returns to avoid lookahead bias
    returns = returns.shift(1)
    
    # Rank stocks based on returns in descending order for each period
    ranked_returns = returns.rank(axis=1, method='average', ascending=False)

    # Initalize DataFrame to store buy signals (1 = buy, 0 = no action)
    signals = pd.DataFrame(index=self.data.index)

    # Generate buy signals for top N stocks on ranked returns
    #for date in ranked_returns.index:
    #    if pd.notna(ranked_returns.loc[date]).all():
    #        top_stocks = ranked_returns.loc[date][ranked_returns.loc[date] <= self.top_n].index
    #        for stock in self.stocks_in_data:
    #            signals.loc[date, stock] = 1 if stock in top_stocks else 0
    #    else:
    #        signals.loc[date, :] = 0
    
    # Adjust signals to reflect the holding period
    holding_signals = signals.copy()
    for i in range(0, len(signals), self.holding_period):
        period_signals = signals.iloc[i:i + self.holding_period]
        if period_signals.empty:
            continue
          
        # Select top N stocks at the start of the period (i.e., on day i)  
        top_stocks = period_signals.iloc[0][period_signals.iloc[0] <= self.top_n].index
        for j in range(self.holding_period):
            if i + j < len(signals):
                holding_signals.iloc[i + j] = 0 # Reset all signals to 0 for the day
                holding_signals.iloc[i + j][top_stocks] = 1 # Set buy signals for top stocks
    # Store selected stocks in the beginning of the period
    self.signal_data = holding_signals

  def calculate_returns(self):
    """
    Calculates the portfolio returns based on the trading signals generated by the
    strategy. The portfolio is rebalanced at the beginning of each holding period,
    and the selected stocks are held for the duration of the holding period.

    Portfolio performance is measured using cumulative log returns, which are
    stored in `self.cumulative_returns`. Additionally, a log of the selected
    stocks and their corresponding returns is maintained in `self.selection_log`.

    Steps:
    1. Calculate daily returns for each stock.
    2. For each holding period, calculate portfolio returns based on the selected stocks.
    3. Update the portfolio value based on the selected stocks' performance.
    """
    # Initialize the portfolio value
    portfolio_value = pd.Series(index=self.data.index, dtype=np.float64)
    portfolio_value.iloc[0] = 1  # Start with an initial value of 1

    # Initialize the log for the selected stocks and cumulative prices
    self.selection_log = pd.DataFrame(index=self.data.index, columns=['Selected_Stocks', 'Cumulative_Prices', 'Log_Return'])

    # Calculate the daily returns of each stock
    daily_returns = pd.DataFrame(index=self.data.index)
    for stock in self.stocks_in_data:
        daily_returns[stock] = self.data[(stock, 'Close')].pct_change().fillna(0)

    # Initialize series for portfolio log returns
    portfolio_log_returns = pd.Series(index=self.data.index, dtype=np.float64)
    selected_stocks = []

    # Iterate through the data and calculate portfolio returns based on selected stocks
    for i in range(1, len(daily_returns)):
        date = daily_returns.index[i]
        prev_date = daily_returns.index[i - 1]

        # Rebalance at the start of a holding period or after the first period
        if i % self.holding_period == 0 or i == 1:
          selected_stocks = self.signal_data.loc[prev_date][self.signal_data.loc[prev_date] == 1].index

        # Calculate portfolio returns based on selected stocks
        if len(selected_stocks) > 0:
            # Calculate the combined return for the selected stocks
            initial_prices = self.data.loc[prev_date, [(stock, 'Close') for stock in selected_stocks]].values
            current_prices = self.data.loc[date, [(stock, 'Close') for stock in selected_stocks]].values
            if initial_prices.sum() != 0
            
              portfolio_return = (current_prices.sum() - initial_prices.sum()) / initial_prices.sum()
            else:
              portfolio_return = 0
            portfolio_log_returns[date] = np.log(1 + portfolio_return)

            # Log selected stocks and cumulative prices
            self.selection_log.at[date, 'Selected_Stocks'] = selected_stocks.tolist()
            self.selection_log.at[date, 'Cumulative_Prices'] = current_prices.sum()
            self.selection_log.at[date, 'Log_Return'] = portfolio_log_returns[date]
        else:
            portfolio_log_returns[date] = 0
            self.selection_log.at[date, 'Selected_Stocks'] = []
            self.selection_log.at[date, 'Cumulative_Prices'] = 0
            self.selection_log.at[date, 'Log_Return'] = 0
    
    # Compute cumulative log returns as the cumulative sum of log returns
    cumulative_log_returns = portfolio_log_returns.cumsum()
    self.cumulative_returns = cumulative_log_returns

  def plot_signals(self):
    """
    Visualizes the stock prices and the trading signals generated by the strategy.
    A buy signal is represented as a green triangle placed on the corresponding stock price.

    The plot displays:
    - Stock price movement for each stock over time.
    - Buy signals for each stock.

    Example:
        agent.plot_signals()
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
    """
    Plots a bar chart comparing the strategy's cumulative returns to a simple 
    buy-and-hold strategy. The buy-and-hold strategy assumes an equal allocation
    across all stocks and no rebalancing.

    The plot displays:
    - Cumulative return of the trading strategy (in percentage).
    - Cumulative return of a buy-and-hold strategy (in percentage).

    Example:
        agent.plot_returns()
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    # Calculate the returns for the strategy and buy_and hold. 
    strategy_return = self.cumulative_returns.iloc[-1] * 100  # Convert to percentage

    # Get total portfolio value at the start (initial) and the end (final)
    initial_portfolio_value = self.data.xs(self.price_type, level=1, axis=1).iloc[0].sum()
    final_portfolio_value = self.data.xs(self.price_type, level=1, axis=1).iloc[-1].sum()

    # Calculate the cumulative return for the entire buy_and_hold portfolio
    buy_and_hold_return = (final_portfolio_value / initial_portfolio_value - 1) * 100

    # Plot bars for strategy and buy and hold return 
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
    Attach a text label above each bar in the bar plot, displaying its height (value).

    Args:
        ax (matplotlib.axes.Axes): The axis object on which to annotate the bars.
        bars (matplotlib.patches.Rectangle): A list of bars to annotate.
    """
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

  def plot_selected_stocks(self):
    """
    Creates a scatter plot showing the stocks selected by the strategy over time.

    Each point represents a stock selected at a specific time period. This provides a 
    visual representation of when and which stocks were selected by the agent.

    Example:
        agent.plot_selected_stocks()
    """
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
    """
    Plots the cumulative returns of the strategy over time and annotates the stocks selected
    at each time point. This visualization helps track how portfolio returns evolved over time
    and highlights which stocks contributed to the performance.

    Example:
        agent.plot_returns_time()
    """
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
