
from agents.base_agents.trading_agent import TradingAgent
from agents.helper_classes.TransformerClassifier import TransformerClassifier
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class TransformerTradingAgent(TradingAgent):
  """  
  A trading agent that utilizes a Transformer-based classifier to generate trading signals. 

  It constructs sequences of price derived features (such as "Open-Close" and "High-Low")
  over a specified lookback period, trans a transformer model to predict the direction
  of the next time instance price movement (up or down) and generates trading signals accordingly.
  """

  def __init__(self, data, lookback = 10, model_params = None, train_split = 0.8, epochs = 20):
    super().__init__(data)
    self.algorithm_name = 'Transformer'
    self.lookback = lookback
    self.stocks_in_data = self.data.columns.get_level_values(0).unique()
    self.signal_data = {}

    if model_params is None:
      model_params = {
        'input_dim': 2, 
        'model_dim': 32, 
        'num_heads': 4,
        'num_layers': 2, 
        'num_classes': 2, 
        'max_seq_len': lookback
      }
      self.model_params = model_params
      self.epochs = epochs

  def create_classification_trading_condition(self,stock):
    """
    Constructs sequences of features and corresponding targets for the given stock. 

    Features:
      'Open-Close': Difference between open and close prices
      'High-Low': Difference between high and low prices
    A sliding window of length 'lookback' is used to build each sample

    Targets:
      1 if next days Close > previous days close else 0

    Returns: 
      X (np.array): Array of shape (n_samples,lookback,2).
      Y (np.array): Array of shape (n_samples,) with labels 0 or 1.
      index (pd.index): Date index corresponding to the first prediction in each sequence
    """
    data_copy = self.data[stock].copy()
    data_copy['Open-Close'] = data_copy['Open'] - data_copy['Close']
    data['High-Low'] = data_copy['High'] - data_copy['Low']
    data_copy.dropna(inplace = True)

    X,Y = [],[]

    for i in range(len(data_copy)-self.lookback):
      seq = data_copy[['Open-Close','High-Low']].iloc[i:i+self.lookback].values
      X.append(seq)
      # Setting the target
      target = 1 if data_copy['Close'].iloc[i+self.lookback] > data_copy['Close'].iloc[i+self.lookback-1] else 0
      Y.append(target)

    X = np.array(X)
    Y = np.array(Y)
    index = data_copy.index[self.lookack:]

  def create_train_split_group(self, X,Y, split_ratio):
    """
    Split the dataset into trading and testing sets without shuffling
    """
    split_point = int(len(X)*split_ratio)
    return X[:split_ratio], X[split_ratio:], Y[:split_ratio], Y[split_ratio:]

  def transfomer_model(self,stock):
    """
    Training a transformer classifier for the stock using the training portion of the split

    Returns: 
      model (TransformerClassifier): The trained model
      index (pd.Index): Index for the test set predictions

    """
    



      

        




