
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
    self.balance = 10_000_000
    self.initial_balance = self.balance
    self.trade_log = []
    self.trade_history = []
    self.holdings = {}
    self.normalized_data = self._normalize_data()
    
  def reset(self):
    """
      Reset the environment for a new episode.

      Returns:
          np.array: The initial state of the environment.
    """
    self.current_step = 0
    self.open_positions = {}  # {stock: holding_duration}
    self.balance = self.initial_balance
    self.trade_log = []
    self.trade_history = []
    self.holdings = {}

    return self._get_state()

  
  def _normalize_data(self):
    """ 
    Normalize stock prices by min/max scaling within each stock
    """
    normalized_data ={}
    for stock in self.stocks:
      stock_prices = self.stock_data.loc[stock]['Close'].ffill()
      min_price = stock_prices.min()
      max_price = stock_prices.max()

      if max_price - min_price == 0: 
        normalized_data[stock] = stock_prices*0
      
      else: 
        normalized_data[stock] = (stock_prices - min_price)/(max_price - min_price)

    return normalized_data

  
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
        current_price = self.stock_data.loc[stock,'Close'].iloc[self.current_step-1]
        past_price = self.stock_data.loc[stock,'Close'].iloc[self.current_step- self.window_size]
        momentum[stock] = (current_price - past_price) / past_price if past_price != 0 else 0
    return sorted(momentum.keys(), key = lambda x: momentum[x],reverse = True) # Rank stock by momentum

  def step(self,actions):
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
    ranked_stocks = self._rank_stocks() # Rank stocks to priortize

    for stock,action in actions.items():
      
      if stock not in self.stock_data.index.get_level_values(0):
        continue # Ignore unknown stock

      if self.current_step >= len(self.stock_data.loc[stock])-1:
        continue # Preventing out of bounds

      current_price = self.stock_data.loc[stock]['Close'].iloc[self.current_step]

      if action == 1:
        if len(self.open_positions) < self.max_positions and self.balance >= self.current_price:
          if stock in ranked_stocks[:self.max_positions]:
            self.open_positions[stock] = current_price
            self.balance -= current_price *( 1 + self.transaction_fee)
            rewards[stock] = -self.transaction_fee * current_price
            self.trade_log.append(f"BUY {stock} at {current_price}")
            self.holdings[stock] = 0
            self.trade_history.append({
                "Stock": stock, "Action": "BUY", "Price": current_price, "Step": self.current_step
            })
        
          else: 
            rewards[stock] = -2
            
      elif action == 2:
        if stock in self.open_positions:
          buy_price,buy_step = self.open_positions[stock]
          profit = current_price - buy_price
          holding_period = self.current_step - buy_step
          rewards[stock] = profit - self.transaction_fee * current_price
          self.balance += current_price*(1-self.transaction_fee)
          del self.open_positions[stock]
          del self.holdings[stock]
          self.trade_log.append(f"SELL {stock} at {current_price}, Profit: {profit}")
          self.trade_history.append({
                          "Stock": stock, "Action": "SELL", "Price": current_price, "Step": self.current_step,
                          "Held_For": holding_period, "Profit": profit
                      })
        else: 
          rewards[stock] = -5
      
      elif action == 0:
        if stock in self.open_positions:
          rewards[stock] = current_price - self.open_positions[stock]
        else:
          rewards[stock] = 0

    for held_stock in self.holdings.keys():
      self.holdings[held_stock] += 1

    self.current_step += 1
    done = self.current_step >= len(self.stock_data.loc[self.stocks[0]]) -1
    next_state = self._get_state() if not done else None
    total_reward = sum(rewards.values())

    return next_state,total_reward,done

  
    





  


  
