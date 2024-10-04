import pandas as pd
import numpy as np

class StockTradingEnviron:
  def __init__(self,data,window_size=5,n_bins = 100):
    """
    Initializes the stock trading environment.

    Args:
        data (pd.DataFrame): DataFrame containing stock price data.
        window_size (int): The number of recent time steps to consider for the state.
    """
    self.data = data['Close']
    self.window_size = window_size
    self.num_state_bins = n_bins
    self.min_price = data['Close'].min()
    self.max_price = data['Close'].max()
    self.bin_width = (self.max_price - self.min_price) / n_bins
    self.n_steps = len(data) - window_size + 1
    self.current_step = 0
    self.last_action_price = None

  def reset(self):
    """
    Resets the environment to the initial state.

    Returns:
        np.array: The initial state of the environment.
    """
    self.current_step = 0
    return self._next_observation()

  def _next_observation(self):
    """
    Returns current state of the environment using specified features in this code. 

    """
    
    window_start = max(self.current_step - self.window_size + 1, 0)
    window_end = self.current_step + 1
    window_data = self.data.iloc[window_start:window_end]
    
    avg_price = window_data.mean()
    price_volatility = window_data.std()

    moving_average_10 = self.data.rolling(window=5).mean().iloc[self.current_step]
    relative_changes = window_data.pct_change().fillna(0)
    state = np.array([avg_price,price_volatility,moving_average_5,relative_changes])
    state_index = self._discretize_price(avg_price)
    
    return state_index
    
    #return self.data.iloc[window_start:window_end].values
  
  def _discretize_price(self, price):
    bin_index = int((price - self.min_price) / self.bin_width)
    return min(bin_index, self.num_state_bins - 1)

  def _get_state_index(self,price):
    bin_index = int((price-self.data.min())/self.bin_width)
    return min(bin_index,self.num_state_bins-1)

  def step(self, action):
    """
    Takes an action in the environment.

    Args:
        action (int): The action taken by the agent.

    Returns:
        tuple: (next_state, reward, done)
    """
    self.current_step += 1
    done = self.current_step >= self.n_steps - 1
    next_state = self._next_observation() if not done else None
    reward = self._calculate_reward(action)
    

    return next_state, reward, done

  def _get_state(self):
    """
    Retrieves the current state of the environment.

    Returns:
        np.array: The current state.
    """
    #start = max(self.current_step - self.window_size, 0)
    #state = self.data.iloc[start:self.current_step + 1]
    # Flatten the state to a 1D array
    #return state.values.flatten()
    start = max(self.current_step-self.window_size,0)
    end  = self.current_step + 1
    
    relative_changes = self.data.pct_change().iloc[start:end]
    relative_changes.fillna(0,inplace = True)
    discretisized_state = pd.cut(relative_changes,self.bins,labels = False)
    print(discretisized_state,relative_changes)
    return discretisized_state.values.flatten()

  def _calculate_reward(self, action):
    """<s
    Calculates the reward based on the action taken.

    Args:
        action (int): The action taken by the agent.

    Returns:
        float: The calculated reward.
    """
    # Example reward function; can be modified based on strategy
    # Assuming: 0 = hold, 1 = buy, 2 = sell
    # This is a placeholder; reward logic should be aligned with trading strategy
    reward = 0

    if self.current_step < len(self.data):
      current_price = self.data.iloc[self.current_step]
      
      if action == 1:  # Buy
          reward = 0 #-self.data.iloc[self.current_step]  # Cost of buying, or no reward since profits are not realized
          self.last_action_price = current_price

      elif action == 2 and self.current_step > 0: # Sell
        if self.last_action_price is not None: 
          # Profit from selling (current price - last price)
          last_buy_price = self.data.iloc[self.current_step-1]
          reward = current_price - last_buy_price
          self.last_action_price = None
          
      elif action == 0 : #Hold 
          if self.last_action_price is not None:
            reward = current_price-self.last_action_price # Reward is based on non-realized profits
            

    return reward

