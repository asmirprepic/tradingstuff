import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQNAgentTest:
  def __init__(self,state_size: int,action_size: int, gamma: float = 0.99,epsilon:float = 1.0,epsilon_decay:float = 0.995,epsilon_min:float = 0.01,lr:float = 0.001,batch_size:int = 32,memory_size:int = 2000,target_update_freq:int = 10):
    """
    Initialize the DQN agent with a neural network for Q-values

    Params: 
    state_size (int): Number features in the state (e.g. stock price, moving average)
    action_size (int): Number of possible actions. For this implementation a buy, hold, sell is used
    gamma (float): Discount factor for future rewards
    epsilon (float): Exploration rate 
    epsilon_decay (float): Decay rate for epsilon
    epsilon_min (float): Minimum value for epsilon
    lr (float): Learning rate for the neural network
    batch_size (int): Size of the batches used for experience replay
    memory_size (int): Maximum size of the replay buffer. 
    target_update_freq (int): Frequency of target network updates
    """
