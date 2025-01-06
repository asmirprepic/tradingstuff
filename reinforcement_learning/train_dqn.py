import random
import numpy as np


def train_dqn_agent(agent,env,episodes = 500, batch_size = 32):
  """
    Train the DQN agent in the trading environment.

    Args:
        agent: The DQNAgent instance.
        env: The MultiStockTradingEnvironment.
        episodes (int): Number of training episodes.
        batch_size (int): Batch size for experience replay.

    Returns:
        agent (DQNAgent): The trained agent.
        training_rewards (list): Total rewards per episode.
        portfolio_values (list): Portfolio value over time.
    """

  
  training_rewards = []
  portfolio_values = []

  for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    episode_portfolio = []

    if env.current_step < env.window_size:
      print(f" Skipping early training steps (waiting for window size)...")
      env.current_step = env.window_size

    while True:
      action_index = agent.act(state)
      actions = {stock: action_index for stock in env.stocks}
      next_state,reward,done = env.step(actions)
      reward = np.sum(reward) if isinstance(reward, (list, np.ndarray)) else float(reward)
      agent.remember(state,action_index,reward,next_state,done)
      agent.replay(batch_size)
      state = next_state
      total_reward += reward
      episode_portfolio.append(env.balance)
      if done: break

    training_rewards.append(total_reward)
    portfolio_values.append(episode_portfolio[-1])
    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
  return agent,training_rewards,portfolio_values
