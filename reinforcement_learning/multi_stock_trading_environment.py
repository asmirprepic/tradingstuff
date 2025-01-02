
import numpy as np
import pandas as pd


class MultiStockTradingEnvironment:
  def __init__(self,stock_data,max_positions = 5, window_size = 10):
    """
      Initialize the multi-stock trading environment.

      Args:
          stock_data (dict): A dictionary of stock price DataFrames, each with a 'Close' column.
          max_positions (int): Maximum number of stocks the agent can hold at any given time.
          window_size (int): Number of historical time steps to consider for the state.
    """
    self.stock_data = stock_data
    self.max_positions = max_positions
    self.window_size = window_size 
    self.current_step = 0
    self.open_positions = []
    self.total_reward = 0

    self.data = {stock: df['Close'] for stock, df in stock_data.items()}
    self.normalized_data = {stock: (df['Close'] - df['Close'].mean()) / df['Close'].std() for stock, df in stock_data.items()}

  
  def reset(self):
    """
      Reset the environment for a new episode.

      Returns:
          np.array: The initial state of the environment.
    """
    self.current_step = 0
    self.open_positions = []
    self.total_reward = 0
    return self._get_state()

  def _get_state(self):
    """
    Get the current state of the environment.

    Returns:
        np.array: Flattened array of historical prices for all stocks.
    """
    state = []
    for stock in self.normalized_data.items():
      window_start = max(self.current_step-self.window_size+1,0)
      state.extend(self.data.iloc[window_start:self.current_step+1])
      return np.array(state)

  
  def step(self):
    """
      Execute actions for all stocks and advance to the next time step.

      Args:
          actions (dict): A dictionary of actions (0=Hold, 1=Buy, 2=Sell) for each stock.

      Returns:
          tuple: (next_state, total_reward, done)
      """
    rewards = {}
    for stock,action in action.items():
      if action == 1:
        if len(self.open_positions) < self.max_positions and stock not in self.open_positions:
          self.open_positions.append(stock)
          rewards[stock] = -1

        else: 
          rewards[stock] =-5

      elif action == 2:
        if stock in self.open_positions:
          rewards[stock]= self.data[stock].iloc[self.current_step]-self.data[stock].iloc[self.current_step-1]
          self.open_positions.remove(stock)
        else: 
          rewards[stock] = -5
        
      elif action == 0:
        if stock in self.open_positions:
          rewards[stock] = self.data[stock].iloc[self.current_step]-self.data[stock].iloc[self.current_step-1]
        else: 
          rewards[stock] = 0

    self.current_step += 1
    done = self.current_step >= len(self.data[list(self.data.keys())[[0]]])-1  
    next_state = self._get_state() if not done else None
    total_reward = sum(rewards.values())
    return next_state, total_reward, done


  


  
