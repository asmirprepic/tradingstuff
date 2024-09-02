from agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split

import tensorflow_probability as tfp

tfd = tfp.distributions

class BNNclassification(TradingAgent):
  """
    A trading agent that generates trading signals based on a Bayesian Neural Network (NN) classification of historical 
    stock information. The NN model predicts the direction of stock price movement and generates trading signals 
    accordingly.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock prices or relevant trading data.
        lags (int, optional): The number of lags to include for some features. Defaults to 5.
  """


  def __init__(self,data,lags =5):
    """
      Initializes the NNClassification agent with provided data and lag features.

      Args:
          data (pd.DataFrame): The dataset containing stock prices or relevant trading data.
          lags (int): The number of lags to include for creating features.
    """
    super().__init__(data)
    self.algorithm_name = "NNClassification"
    self.lags = lags
    self.stocks_in_data = self.data.columns.get_level_values(0).unique()
    self.models = {}  


    for stock in  self.stocks_in_data:
      self.generate_signal_strategy(stock,lags)
    
    self.calculate_returns()
    

    
  def create_train_split_group(self,X,Y,split_ratio):
    """
      Splits the dataset into training and testing sets.

      Args:
          X (pd.DataFrame): The feature set.
          Y (pd.Series): The target variable.
          split_ratio (float): The proportion of the dataset to include in the train split.

      Returns:
          tuple: The training and testing sets (X_train, X_test, Y_train, Y_test).
    """
    return train_test_split(X,Y,shuffle = False,test_size =1- split_ratio)
  
  def create_classification_trading_condition(self,stock,lags):
    """
      Creates trading conditions and features for the NN classification model.

      Args:
          stock (str): The stock symbol for which to create features.
          lags (int): The number of lags to include for creating features.

      Returns:
          pd.DataFrame: The feature set.
          pd.Series: The target variable, where 1 indicates an upward price movement 
                      and 0 indicates a downward price movement.
          list: The list of feature column names.
    """

    data_copy = self.data[stock].copy()
    data_copy['return'] = np.log(self.data[stock]['Close']/self.data[stock]['Close'].shift(1))
    data_copy['direction'] = np.where(data_copy['return']>0,1,0)
    data_copy['Open-Close'] = self.data[stock]['Open'] - self.data[stock]['Close']
    data_copy['High-Low'] = self.data[stock]['High'] - self.data[stock]['Low']
    data_copy['long_MA'] = self.data[stock]['Close'].rolling(window=100).mean()
    data_copy['short_MA'] = self.data[stock]['Close'].rolling(window=50).mean()
    #data_copy['MA_crossover'] = np.where(data_copy['short_MA']>data_copy['long_MA'],1,-1)
    #data_copy['volume_MA'] = data_copy['Volume'].rolling(window=10).mean()
    epsilon = 1e-10 
    #data_copy['volume_change'] = data_copy['Volume'].diff() / (data_copy['Volume'].shift(1) + epsilon)
    lags = lags
    cols  = []

    """
    for lag in range(1,lags+1):
      col = f'lag_{lag}'
      data_copy[col] = data_copy['return'].shift(lag)
      cols.append(col)
      data_copy.dropna(inplace=True)
    """
    
    data_copy['momentum'] = data_copy['return'].rolling(10).mean()
    #data_copy['volatility'] = data_copy['return'].rolling(20).std().shift(1)
    data_copy['distance'] = (self.data[stock]['Close']-self.data[stock]['Close'].rolling(10).mean())
    #cols.extend(['momentum','volatility','distance'])
    #data_copy.dropna(inplace=True)

    data_copy.dropna(inplace=True)

    cols.extend(['Open-Close','High-Low','distance','momentum'])
    X = data_copy.drop(columns=['direction','High','Low','Volume','Open','Close','short_MA','long_MA'])
    Y = data_copy['direction']
    
    return X,Y,cols
  
  def NNmodelFit(self,stock,lags):
    """
      Defines and fits the neural network model for a given stock.

      Args:
          stock (str): The stock symbol for which to train the model.
          lags (int): The number of lags to include for creating features.

      Returns:
          tuple: The fitted model, training history, training set, and other relevant data.
    """
    
    ## Prepare data ###
    X,Y,cols = self.create_classification_trading_condition(stock,lags)
    
    X_train,X_test,Y_train,Y_test = self.create_train_split_group(X,Y,split_ratio=0.8)
    
    
    ## Normalize the data
    mu_train,std_train = X_train.mean(), X_train.std()
    X_train_ = (X_train - mu_train)/std_train
    X_test_ = (X_test - mu_train)/std_train

    # Prior weight disitribution (Gaussian)

    def prior(kernel_size,bias_size,dtype = None):
      n = kernel_size + bias_size
      prior_m  = Sequential([
        tfp.layers.DistributionLambda(
          lambda t: tfd.MultivariateNormalDiag(
            loc = tf.zeros(n),scale_diag = tf.ones(n)
          )
        )
      ])
      return prior_m

    def posterior(kernel_size,bias_size,dtype = None):
      n = kernel_size + bias_size
      posterior_m = Sequential([
        tfp.layers.VariableLayer(
          tfp.layers.MultivariateNormalTriL.params_size(n),dtype =dtype),
          tfp.layers.MultivariateNormalTriL(n)
        
      ])
      return posterior_m



    ### Define model ###
    model = Sequential([
        tfp.layers.DenseVariational(input_shape=(len(cols),),
                                    units=64,
                                    make_prior_fn=prior,
                                    make_posterior_fn=posterior,
                                    kl_weight=1/X_train.shape[0],
                                    activation='relu'),
        tfp.layers.DenseVariational(units=64,
                                    make_prior_fn=prior,
                                    make_posterior_fn=posterior,
                                    kl_weight=1/X_train.shape[0],
                                    activation='relu'),
        tfp.layers.DenseVariational(units=1,
                                    make_prior_fn=prior,
                                    make_posterior_fn=posterior,
                                    kl_weight=1/X_train.shape[0],
                                    activation='sigmoid')
    ])

    model.compile(optimizer = Adam(learning_rate = 0.0001),loss = 'binary_crossentropy',metrics = ['accuracy'])

    model.fit(X_train[cols],Y_train,epochs = 10, batch_size = 1)

    res = pd.DataFrame(model.history.history)

    return model,res,X_train,Y_train,cols,X,Y,X_test
    
  def generate_signal_strategy(self,stock,lags):
    """
      Generates trading signals for the specified stock using a trained neural network model.

      Args:
          stock (str): The stock symbol for which to generate signals.
          lags (int): The number of lags to include for creating features.

      The method updates the `signal_data` attribute with signals for the given stock.
    """
    signals = pd.DataFrame(index= self.data.index)
    model,res,X_train,Y_train,cols,X,Y,X_test = self.NNmodelFit(stock,lags)
    signal = model.predict(X[cols])
    
    signals.loc[X.index,'Position'] = np.where(signal > 0.5,1,-1)
    signals.loc[X.index,'Prediction'] = signal
    #signals['buy_signal'] = (signals['Signal'] == 1).astype(int)
    #signals['sell_signal'] = (signals['Signal'] == -1).astype(int)
    signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))
    
    # Calculate Signal as the change in position
    signals['Signal'] = 0
    signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
    signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1
    
    position_chg = signals['Position'].diff().abs() != 0
    signals.loc[X.index,'buy_signal'] = signals[(signals['Position'] ==1) & position_chg]
    signals.loc[X.index,'sell_signal'] = signals[(signals['Position'] ==-1) & position_chg]
    
    self.signal_data[stock] = signals
    self.models[stock] = model
  
  
    
 
