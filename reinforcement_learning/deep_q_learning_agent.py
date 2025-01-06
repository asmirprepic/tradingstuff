import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from collections import deque  # For memory management
import random

class DQNAgent:
  def __init__(self,state_size,action_size,learning_rate = 0.001,gamma=0.99,epsilon = 1.0,epsilon_decay = 0.995,epsilon_min = 0.01,memory_size = 1_000_000):
    """
      Initialize the DQN Agent.

      Args:
          state_size (int): Dimension of the state space.
          action_size (int): Number of possible actions.
          learning_rate (float): Learning rate for the optimizer.
          gamma (float): Discount factor for future rewards.
          epsilon (float): Initial exploration rate.
          epsilon_decay (float): Decay rate for exploration.
          epsilon_min (float): Minimum exploration rate.
          memory_size (int): Maximum size of memory
    """ 

    self.state_size = state_size
    self.action_size = action_size
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.batch_size = 64
    self.memory = deque(maxlen = memory_size)

    self.model = self._build_model()
    

  def _build_model(self):
    """
    Build the neural network for Q-value approximation.

    Returns:
        Sequential: Compiled Keras model.
    """

    model = Sequential([
      Dense(128,input_dim = self.state_size,activation = 'relu'),
      Dense(128,activation = 'relu'),
      Dense(self.action_size,activation = 'linear' )
    ])

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate),loss ='mse')
    return model

  def act(self,state):
    """
      Choose an action using an epsilon-greedy policy.

      Args:
          state (np.array): Current state.

      Returns:
          int: Chosen action.
      """

    if np.random.rand() <= self.epsilon:
      return np.random.randint(0,self.action_size)
    q_values = self.model.predict(np.array([state]), verbose=0)
    return np.argmax(q_values[0])  # Action with the highest Q-value

  def remember(self, state, action, reward, next_state, done):
    
    if isinstance(reward, (list, np.ndarray)):  
        if len(reward) == 1:
            reward = reward[0]  
        else:
            print("ERROR: Expected scalar reward but got multi-dimensional reward")
            return  
    
    fixed_state = np.array(state, dtype=np.float32).flatten()
    fixed_next_state = np.array(next_state, dtype=np.float32).flatten()

    self.memory.append((
        fixed_state,
        int(action),
        float(np.array(reward).item()),
        fixed_next_state,
        bool(done)
    ))

    

  def replay(self,batch_size =32):
    """
    Trains the model using experience replay

    Args:
      batch_size (int): Number of experiences per training step
    """

    if len(self.memory) < batch_size: 
      return # Dont train until memory has enough
    
    try:
      if not isinstance(self.memory, list):
        self.memory = list(self.memory)
      minibatch = random.sample(self.memory, batch_size)  # Use Python's random module


    except ValueError as e: 
      print("ERROR: Memory sampling issue", e)
      return  # Exit replay function safely

    for state,action, reward,next_state, done in minibatch:
      target = reward
      if not done:
        target += self.gamma * np.max(self.model.predict(np.array([next_state]),verbose = 0)[0])
      
      # Update Q-value
      target_q_values = self.model.predict(np.array([state]),verbose = 0)
      target_q_values[0][action] = target 
      
      self.model.fit(np.array([state]), target_q_values, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min: 
      self.epsilon *= self.epsilon_decay

  def load(self,name):
    """ Load trained model """
    self.model.load_weights(name)
  
  def save(self,name):
    self.model.save_weights(name)
