from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
from reinforcement_learning.q_learning_agent import QLearningAgent
from reinforcement_learning.stock_trading_environment import StockTradingEnviron

class QLearningTradingAgent(TradingAgent):
  """
    A trading agent that uses Q-learning to generate trading signals based on stock price data.

    This agent is designed to work with stock trading environments where it learns
    to make decisions (buy, sell, hold) based on past price data.

    Attributes:
        data (pd.DataFrame): A DataFrame containing the stock prices or relevant trading data.
        state_size (int): The size of the state representation for the Q-learning model.
        action_size (int): The number of possible actions in the Q-learning model.
        q_learning_parameters (dict): Parameters for the Q-learning agent, such as learning rate and discount factor.
        algorithm_name (str): Name of the algorithm (set to "QLearning").
        stocks_in_data (pd.Index): Unique stock symbols present in the data.
        environments (dict): Dictionary mapping stock symbols to their respective trading environments.
        agents (dict): Dictionary mapping stock symbols to their respective QLearningAgent instances.
        signal_data (dict): Dictionary to store generated trading signals for each stock.
    """

  def __init__(self, data, state_size, action_size, q_learning_parameters):
      """
      Initializes the QLearningTradingAgent with the given data, state size, action size, and Q-learning parameters.

      Args:
          data (pd.DataFrame): DataFrame containing the stock price data.
          state_size (int): Size of the state space for the Q-learning model.
          action_size (int): Size of the action space for the Q-learning model.
          q_learning_parameters (dict): Parameters for the Q-learning model, including learning rate and discount factor.
      """
      super().__init__(data)
      self.algorithm_name = "QLearning"
      self.state_size = state_size
      self.action_size = action_size
      self.q_learning_parameters = q_learning_parameters
      self.stocks_in_data = self.data.columns.get_level_values(0).unique()
      
      # Initialize Q-learning components
      self.init_q_learning()

      # Generate trading signals
      self.generate_signal_strategy()
      self.calculate_returns()

  def init_q_learning(self):
    """
      Initializes the Q-learning environments and agents for each stock in the data.

      This method creates a StockTradingEnviron and a QLearningAgent for each stock symbol.
      It stores these in the environments and agents dictionaries, keyed by stock symbol.
    """
    # Initialize the Q-learning environment and agent
    self.environments = {}
    self.agents = {}
    
    for stock in self.stocks_in_data:
      self.environments[stock] = StockTradingEnviron(self.data[stock])
      self.agents[stock] = QLearningAgent(self.state_size, self.action_size, **self.q_learning_parameters)
     
    for stock in self.stocks_in_data:
      print(stock)
      self.train(stock,num_episodes=1000)

  def generate_signal_strategy(self):
    """
        Generates trading signals for each stock using the trained Q-learning agents.

        This method iterates through each stock, uses the corresponding Q-learning agent
        to decide actions based on the state, and generates trading signals based on these actions.
        The signals indicate buy, sell, or hold decisions. The method also calculates returns.
    """
    self.signal_data = {}

    # Use the trained Q-learning agent to generate trading signals for each stock
    for stock in self.stocks_in_data:
      
      environment = self.environments[stock]
      agent = self.agents[stock]
      
      # Prepare a dataframe for the stock
      signals = pd.DataFrame(index = self.data[stock].index)
      signals['Position'] = 0
      signals['Signal'] = 0
      last_position = None
      hold_days = 0
      min_hold_period = 5


      # Define how the state is represented for each stock
      state = environment.reset()  # State representation logic
      last_action = 0

      for step in range(len(self.data[stock])):
        #fromatted_state = self._format_state(state,step,stock)
        
        action = agent.choose_action(state)
        position = 0 if action == 0 else (1 if action == 1 else -1)
        signals.loc[step,'Position'].iloc[step] = position

        # Generate signal based on change in action
        if last_position is not None and position != last_position:
          signals.loc[step,'Signal'].iloc[step] = 1 if position != 0 else -1

        last_position = position


        last_action = action
        #Update the next prediction
        next_state,_,_ = environment.step(action)
        state = next_state

      # Calculate returns
      signals['return'] = np.log(self.data[stock]['Close'] / self.data[stock]['Close'].shift(1))
      self.signal_data[stock] = signals

  def generate_signal_on_new_data(self,new_data):
    """
    Uses the trained Q-learning agent to generate trading signals on new data. 
    Args: 
      new_data (pd.DataFrame): New stock price data tp generate signals for. 

    Returns: 
      dict: A dictionary of trading signals for each stock
    """
    self.new_signal_data = {}

    for stock in new_data.columns.get_level_values(0).unique():
      # Create environment with new data for each stock 
      environment = StockTradingEnviron(new_data[stock])
      agent = self.agents[stock]
      signals = pd.DataFrame(index = new_data[stock].index)
      signals['Position'] = 0
      signals['Signal'] = 0
      last_position = None
      hold_days = 0
      min_hold_period = 5

      # Initialize state for new data 
      state = environment.reset()
      last_action = 0

      for step in range(len(new_data[stock])):
        if hold_days >= min_hold_period:
          action = agent.choose_action(state,hold_days,last_action)
          hold_days = 0
        else: 
          action = last_action
          hold_days += 1

        position = 0 if action == 0 else (1 if action == 1 else -1)
        signals['Position'].iloc[step] = position

        # Generate buy sell signals based on change in action 
        if last_position is not None and position != last_position: 
          signals['Signal'].iloc[step] = 1 if position > last_position else -1

        last_position = position
        last_action = action

        # Update state 
        next_state,_,done = environment.step(action)
        if done: 
          break
        state  = next_state

        self.new_signal_data[stock] = signals

        return self.new_signal_data
  
  def train(self,stock, num_episodes=1000):
    """
        Trains the Q-learning agent for a specific stock over a given number of episodes.

        Args:
            stock (str): The stock symbol to train the agent on.
            num_episodes (int): The number of episodes to run the training for.

        Each episode involves the agent making decisions over the entire data length,
        and the agent learns from these decisions to improve its policy.
        """

    print(f"Training on stock: {stock}")
    environment = self.environments[stock]
    agent = self.agents[stock]
      
    for episode in range(num_episodes):
      state = environment.reset()
      total_reward = 0
      done = False

      while not done:
        action = agent.choose_action(state)
        next_state, reward, done = environment.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        

      if episode % 100 == 0:
          print(f"Stock: {stock}, Episode: {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
          

  def _format_state(self,state,step,stock):
    """
        Formats the state for the Q-learning model based on the last N closing prices.

        Args:
            state: The current state representation.
            step (int): The current time step in the data.
            stock (str): The stock symbol.

        Returns:
            np.array: An array of the last N closing prices for the given stock.
        """
    N = self.window_size
    start_index = max(step-N,0)
    end_index = step + 1
    state_data = self.data[stock]['Close'][start_index:end_index]

    return state_data.values
