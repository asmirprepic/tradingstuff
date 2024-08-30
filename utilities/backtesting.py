

class BackTesting: 

  """
  A class for backtesting trading strategies. 

  Attributes: 
  --------------
  data (pd.DataFrame): Historical price data
  initial_capital (float): The starting capital for the backtest
  signals (dict): A dictionary of signals from different trading agents
  portfolio (pd.DataFrame): DataFrame containing portfolio performance

  """

  def __init__(self,data,initial_capital = 10000.0):
    self.data = data
    self.initial_capital = initial_capital
    self.signals = {}
    self.portfolio = pd.DataFrame(index = self.data)

  def add_strategy(self,agent,stock):
    """
    Adds a strategy's trading signals

    Args: 
    ------------------
      agent (TradingAgent): The trading agent to add
      stock (str): The stock symbol for which to add the trading strategy
    """
    if (stock,'Close') not in self.data.columns:
      raise  ValueError(f"Stock {stock} not found in data")
    self.signals[agent.algorithm_name] = agent.signal_data[stock]

  def run_backtest(self):
    """
    Runs the backtest using the provided signals and calculates the portfolio value
    """

    for (name,stock),signals in self.signals.items():
      signals['Holdings'] = signals['Position'] * self.data['Close']*100
      signals['Cash']  = self.initial_capital - (signals['Signal'] * signals['Position']*self.data['Close']*100).cumsum()
      signals['Total']= signals['Holdings'] + signals['Cash']
      signals['Return'] = signals['Total'].pct_change().fillna(0)
      self.portfolio[(name,stock)] = singals['Total']

  def calculate_performance(self):
    performance = pd.DataFrame(index = pd.MultiIndex.from_tuples(self.signals.keys(),names = ['Strategy','Stock']))

    for (name,stock),_ in self.singals.items():
      total_return = (self.portfolio[(name,stock)].iloc[-1]/self.portfolio[(name,stock)].iloc[0]) - 1
      daily_returns= self.portfolio[(name,stock)].pct_change().fillna(0)
      compounded_returns = (1+daily_returns).mean()
      annualized_return = compounded_return**(252/len(daily_returns)) -1

      annualized_volatility = daily_returns.std()*np.sqrt(252)

      sharpe_ratio = annualized_return/annualized_volatility

      performance.loc[(name,stock), 'Total returns'] = total_return*100

  return performance

  def plot_performance(self):
    """
    Plots the portfolio performance of the strategies over time. 
    Each strategy and stock combination is plotted separetely
  
    """
    plt.figure(figsize = (12,8))
    for (name,stock) in self.portfolio.columns:
      plt.plot(self.portfolio.index, self.portfolio[(name,stock)],label = f"{name} ({stock})")
      plt.title("Portfolio performance over time")
      plt.xlabel('Date')
      plt.ylabel('Portfolio Value')
      plt.legend()
      plt.grid(True)
      plt.show()

      
                               
                                                            
  

    
      


