
import numpy as np
import pandas as pd


class MultiStockTradingEnvironment:
  def __init__(self,stock_data,max_positions = 5, window_size = 10,transaction_fee = 0.001):
    """
      Initialize the multi-stock trading environment.

      Args:
          stock_data (dict): A dictionary of stock price DataFrames, each with a 'Close' column.
          max_positions (int): Maximum number of stocks the agent can hold at any given time.
          window_size (int): Number of historical time steps to consider for the state.
          initial_balance (int): Inital portfolio balance
    """
    self.stock_data = stock_data
    self.max_positions = max_positions
    self.window_size = window_size 
    self.transaction_fee = transaction_fee
    
    self.stocks = stock_data.index.get_level_values(0).unique()

    self.current_step = 0
    self.open_positions = {}
    self.balance = 10000000
    self.initial_balance = self.balance
    self.trade_log = []

    self.normalized_data = self._normalize_data
    
    
  
  def _normalize_data(self):
    """ 
    Normalize stock prices by min/max scaling within each stock
    """
    normalized_data ={}
    for stock in self.stocks:
      stock_prices = self.stock_data.loc[stock]['Close']
      normalized_data[stock] = (stock_prices - stock_prices.min())/(stock_prices.max() - stock_prices.min())

    return normalized_data

  def reset(self):
    """
      Reset the environment for a new episode.

      Returns:
          np.array: The initial state of the environment.
    """
    self.current_step = 0
    self.open_positions = {}  # Empty portfolio
    self.balance = self.initial_balance
    self.trade_log = []

    return self._get_state()

  def _get_state(self):
    """
    Construct the current state vector.

    Returns:
        np.array: Flattened state representation combining:
            - Historical normalized prices (window_size days)
            - Cash balance (normalized)
            - Number of held stocks (normalized)
    """
    state = []
    for stock in self.stocks:
      price_series = self.normalized_data[stock]
      window_start = max(0,self.current_step-self.window_size)
      stock_prices = price_series.iloc[window_start:self.current_step].values
      state.extend(stock_prices)
    
    # Portfolio info
    state.append(self.balance/self.initial_balance)
    state.append(len(self.open_positions) / self.max_positions)

    return np.array(state,dtype= np.float32)
  
  def _rank_stocks(self):
    """ Ranking stocks based on momentum and price actions to prioritize better buy selection """
    momentum = {}

    

    for stock in self.stocks: 
      if self.current_step < self.window_size:
        momentum[stock] = 0 # Not enough data adjustment
      
      else: 
        current_price = self.stock_data.loc[stock,'Close'].iloc[self.current_step]
        past_price = self.stock_data.loc[stock,'Close'].iloc[self.current_step- self.window_size]
        momentum[stock] = (current_price - past_price)/(past_price)
    return sorted(momentum.keys(), key = lambda x: momentum[x],reverse = True) # Rank stock by momentum

  def step(self):
    """
    Execute the actions chosen by the agent.

    Args:
        actions (dict): A dictionary where:
            - Key: Stock ticker
            - Value: Action (0=Hold, 1=Buy, 2=Sell)

    Returns:
        tuple: (next_state, total_reward, done)
    """
    rewards = {}
    # Rannk stocks to priortize
    ranked_stocks = self._rank_stocks()

    for stock,action in action.items():
      current_price = self.stock_data.loc[stock]['Close'].iloc[self.current_step]

      if action == 1:
        if len(self.open_positions) < self.max_positions and self.balance >= self.current_price:
          if stock in ranked_stocks[:self.max_positions]:
            self.open_positions[stock] = current_price
            self.balance -= current_price *( 1 + self.transaction_fee)
            rewards[stock] = -self.transaction_fee * current_price
            self.trade_log.append(f"BUY {stock} at {current_price}")
          
          else: 
            rewards[stock] = -2
      elif action == 2:
        if stock in self.open_positions:
          buy_price = self.open_positions[stock]
          profit = current_price - buy_price
          rewards[stock] = profit - self.transaction_fee * current_price
          self.balance += current_price*(1+self.transaction_fee)
          del self.open_positions[stock]
          self.trade_log.append(f"SELL {stock} at {current_price}, Profit: {profit}")
        
        else: 
          rewards[stock] = -5
      
      elif action == 0:
        if stock in self.open_positions:
          rewards[stock] = current_price - self.open_positions[stock]
        else:
          rewards[stock] = 0

    self.current_step += 1
    done = self.current_step >= len(self.stock_data.loc[self.stocks[0]]) -1
    next_state = self._get_state() if not done else None
    total_reward = sum(rewards.values())

    return next_state,total_reward,done





  


  
