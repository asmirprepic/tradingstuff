
class BacktestingML(self,window = 252):

  """
  Backtesting framework that periodically refits models using a rolling or walk-forward approach.

  """
  def __init__(self,data, initial_capital = 100000.0,window =252):
    self.data =data
    self.initial_capital = initial_capital
    self.window = window
    self.portfolio = pd.DataFrame(index = self.data.index)
    self.singals = {}

  def add_strategy(self,agent,stock):
    """
    Add strategy for the signal backtest

    Args: 
    --------------
    agent (TradingAgent): The trading agent to add
    stock (str): The stock symbol for which to add the strategy

    """

    if (stock,'Close') not in self.data.columns:
      raise ValueError(f"Stock {stock} not found in the data")


  def run_backtest(self):
    """
    Runs the backtest using a walk-forward approach, avoiding look-ahead bias
    """

    for (name,stock), _ in self.signals.items():
      stock_close = self.data[(stock,'Close')]
      holdings, cash,total = [],[],[]


      for i in range(self,window,len(stock_close)):
        # Train the data on the window up to day i
        X_train,Y_train = self.create_train_data(stock,start = i-self.window,end = i) 
        agent_model = self.train_model(X_train,Y_train,strategy=name) 

        # Store positions, cash and portfolio value
        position= signal*100 # Assuming 100 shares
        holdings.append(position*stock_close.iloc[i])
        if len(cash) > 0: 
          cash.append(cash[-1] - signals*stock_close.iloc[i]*100)
        else: 
          cash.append(self.initial_capital - signal*stock_close.iloc[i]*100)

        total.append(holdings[-1] + cash[-1])

      # Save results to portfolio 
      signals_df = pd.DataFrame(index = stock_close.index[self.window:])
      signals_df['Holdings'] = holdings
      singals_df['Cash'] = cash
      signals_df['Total'] = total
      signals_df['Returns'] = signals_df['Total'].pct_change().fillna()
      self.portfolio[(name,stock)]  = signals_df['Total']

  def create_train_data(self,stock,current_index):
    """
    Creates training data for the model using a slice of historical data. 
    Args: 
    ---------------
    stock(str): Symbol
    start(int): The start of the training window 
    end(int): The end index of the training window

    Returns: 
    --------------
    X (np.ndarray): The feature set for the training data
    Y (np.ndarray): The target variable for the training data
    """
    


      
