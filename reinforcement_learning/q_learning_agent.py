import numpy as np
import pandas as pd


class QLearningAgent:
  def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,min_hold_period = 5):
    """
    Initializes the Q-learning agent.

    Args:
        state_size (int): The size of the state space.
        action_size (int): The size of the action space.
        alpha (float): The learning rate.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The exploration rate for choosing random actions.
        epsilon_decay (float): The rate at which epsilon decreases.
    """
    self.state_size = state_size
    self.action_size = action_size
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.q_table = np.zeros((state_size, action_size))

  def choose_action(self, state,bins = 10):
    """
    Chooses an action based on the current state and Q-table.

    Args:
        state (int): The current state.

    Returns:
        int: The chosen action.
    """
    
    # Enforcing minimum holding period
    if hold_days < self.min_holding_period:
      action = last_action
    else:
      # Exploration or exploitation
      if np.random.rand() < self.epsilon:
        action  = np.random.choice(self.action_size)
      else: 
        action = np.argmax(self.q_table[state])
        
    #print(f"Current State: {state}")
    if np.random.rand() < self.epsilon:
        action =  np.random.choice(self.action_size)
    else:
        action= np.argmax(self.q_table[state])
        
    #print(f"Chosen Action: {action}")
    return action

  
  def learn(self, state, action, reward, next_state, done):
    """
    Updates the Q-table based on the action taken and the reward received.

    Args:
        state (int): The current state.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (int): The next state.
        done (bool): Whether the episode is finished.
    """
    
    predict = self.q_table[state, action]
    
    
    
    target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
   
    self.q_table[state, action] += self.alpha * (target - predict)

    if not done and self.epsilon > 0.01:
        self.epsilon *= self.epsilon_decay

  def reset(self):
    """ Resetting the agents q_table"""
    self.q_table = np.zeros((self.state_size,self.action_size))
    self.epsilon = 1.0
    



