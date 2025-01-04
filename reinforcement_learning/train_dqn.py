import random
from q_learning_agent import DQNAgent

def train_dqn(self,env,episodes = 500, batch_size = 32):
  """
  Trains the DQN agent on the training environment. 

  Args: 
    env: MultiStockTradingEnvironment object
    episodes (int): Number of episodes for training
    batch_size (int): Size of replay batch
  """

  state_size = env.get_state().shape[0]
  action_size = 3
  agent = DQNAgent(state_size,action_size)

  for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
      action_index = agent._choose_action(state)
      actions = {stock: action_index for stock in env.stocks}
      next_state,reward,done = env.step(actions)
      agent.remember(state,action_index,reward,next_state,done)
      agent.replay(batch_size)
      state = next_state
      total_reward += reward
      if done: break

    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
  return agent
