
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

      # Note to self. Implement this so that its not immediately trained. Also implement such that the model is trained beforehand.
      #for stock in self.stocks_in_data: 
      #  self.generate_signal_strategy(stock):
        

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
    X,Y,index = self.create_classification_trading_condition(stock)
    X_train,Y_train,X_test,Y_test = self.create_train_split_group(X,Y,split_ratio = 0.9)
    X_train_tensor = torch.tensor(X_train,dtype = torch.float32)
    Y_train_tensor = torch.tensor(Y_train,dtype = torch.long)

    model = TranformerClassifier(
      input_dim = self.model_params['input_dim'],
      model_dim = self.model_params['model_dim'],
      num_heads = self.model_params['num_heads'],
      num_layers = self.model_params['num_layers'],
      num_classes = self.model
    )

    optimizer = optim.Adam(model.parameters(),lr =1e-3)
    criterion = nn.CrossEntropy()
    model.train()

    batch_size = 32
    dataset = torch.utils.data.TensorDataSet(X_train_tensor,Y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size, shuffle = True)

    for epoch in range(self.epochs):
      for batch_X, batch_Y in dataLoader:
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits,batch_Y)
        loss.backward()
        optimizer.step()
    return model,index

def generate_signal_strategy(self,stock):
  """
  Generates trading signals for the specified stock using the trained Transformer classifer
  Predictions are converted to positions and then signals are then computed based on changes in positions.
  The generated signal dataframe is stored in self.signal_data

  """

  X,_,index = self.create_classification_trading_condition(stock)
  model,index = self.transformer_model(stock)
  model.eval()
  X_tensor = torch.tensor(X,dtype = torch.float32)
  with torch.no_grad():
    logits = model(X_tensor)
    predictions = torch.argmax(logits,dim =1).cpu().numpy()

  signals = pd.DataFrame(index = index)
  signals['Predictions'] = predictions
  signals['Position'] = signals['Predictions'].apply(lambda x: 1 if x == 1 else 0)
  signals['Signal'] = 0
  signals.loc[signals['Position'] > signals['Position'].shift(1),'Signal'] = 1
  signals.loc[signals['Position'] > signals['Position'].shift(1),'Signal'] = -1

  signals['return'] = np.log(self.data[(stock, 'Close')] / self.data[(stock, 'Close')].shift(1))
  self.signal_data[stock] = signals

  



      

        




