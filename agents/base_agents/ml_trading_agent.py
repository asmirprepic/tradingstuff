from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from agents.base_agents.trading_agent import TradingAgent

class MLBasedAgent(TradingAgent):
  """
  Abstract base class for machine-learning based trading algorithms.

  Inherits from TradingAgent and provides machine learning training, signal generation,
  and evaluation functionality. Subclasses must define the model, features, and
  optionally override feature_engineering() and generate_signal_strategy().

  Attributes:
  model (object): The ML model (must implement fit/predict API).
  features (list[str]): Names of feature columns used for training.
  trained (bool): Flag indicating whether the model has been trained.
  """


  def __init__(self,data, model = None,features = None):
    """
    Initialize the MLTradingAgent with provided data and model

    Args:
    -------------
    data (pd.DataFrame): The dataset containing the stock prices or relevant trading data. Should be multilevel index with level 1 the stock symbol and level 2 data
    model (object): The machine learning model to use for predicitions
    features (list): a list of feature column names used for model training
    """

    super().__init__(data)
    self.algorithm_name = 'MLBaseAlgorithm'
    self.model = model
    self.features = features
    self.trained = False

  def default_feature_engineering(self,stock):
    """
    Default feature set: Open - Close and High - Low differences.
    Target: +1 if next close > current, else -1

    Returns:
      X (pd.DataFrame): Engineered features.
      Y (pd.Series): Binary target for classification
    """

    df = self.data[stock]
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    df = df.fill()

    X = df[self.features].copy()
    Y = np.where(df['Close'].shift(-1) > df['Close'],1,-1)
    Y = pd.Series(Y, index = df.index)

    return X,Y

  @abstractmethod
  def feature_engineering(self,stock):
    """
    Abstract method for custom feature engineering.
    Returns (X,Y) with X features and Y target.
    """
    pass


  def create_classification_trading_condition(self,stock):
    """
    Creates the feature set for classification model.

    Args:
    -----------
      stock (str): The stock symbol for which to create features.

    Returns:
    ----------
      pd.DataFrame: The features set
      pd.Series: The target variable where 1 indicates an upward price movement and -1 indicates a downward price movement.
    """

    data_copy = self.data[stock].copy()
    data_copy['High-Low'] = self.data[stock]['High'] - self.data[stock]['Low']
    data_copy['Open-Close'] = self.data[stock]['Open'] - self.data[stock]['Close']
    data_copy.ffill()

    X = data_copy[['High-Low','Open-Close']]
    Y = np.where(self.data[stock]['Close'].shift(-1) > self.data[stock]['Close'],1,-1)
    Y_series = pd.Series(Y,index = self.data[stock].index)
    Y = Y_series.loc[X.index]

    return X,Y

  def create_train_split_group(self,X,Y,split_ratio):
    """
    Splits the dataset into training and testing sets.

    Args:
    -------------
      X (pd.DataFrame): The feature set
      Y (pd.DataFrame): The target variable
      split_ratio (float): The proportion of the dataset to include in the train split

    """

    return train_test_split(X,Y,shuffle = False, test_size = 1-split_ratio)



  @abstractmethod
  def feature_engineering(self,stock):
    """
    Abstract method for feature engineering.

    This method should be implemented by subclasses to define the specific features to be used
    by the model.

    Args:
    -----------
    stock (str): The stock symbol for which to train the model
    test_size (float): The proportion of data to use as a test set

    Returns:
    ------------
    tuple: A tuple containng:
      - pd.DataFrame: A dataframe of engineering features
      - pd.Series: A Series containing the target variable for training.
    """

    return self.default_feature_engineering(stock)


  def train_model(self,stock,test_size = 0.2):
    """
    Trains the model on the specific stocks data.

    Args:
    -----------------
    stock (str): The stock symbol for which to train the model.
    test_size (float): The proportion of data to use as a test set.

    Returns:
    -----------------
    dict: A dictionary containing model performance metrics
    """


    if not self.model or not self.features:
      raise ValueError("Model and features must be defined to train the model")

    X,y = self.feature_engineering(stock)
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size =test_size)

    self.model.fit(X_train,y_train)
    y_pred = self.model.predict(X_test)

    metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred)
  }
    print(f"Model Performance metrics for {stock} ({self.algorithm_name}):")
    for metric,value in metrics.items():
      print(f"{metric}: {value:.4f}")
    self.trained = True
    return metrics



  @abstractmethod
  def generate_signal_strategy(self,stock,*args):
    """
    Abstract method to generate trading signals for a given stock using the ML model.

    This method should be implemented by subclasses to define the specific strategy for generating sigals using
    the trained ML model.

    Args:
    ----------
    stock (str): The stock symbol for which to generate the signals.
    """
  pass

  def predict_signals(self,stock):
    """
    Use the trained ML to predict trading signals.

    Args:
    -----------
    stock (str): The stock symbol for which to predict signals.

    Returns:
    ----------
    pd.DataFrame: A dataframe containing the predicted signals

    """
    if not self.trained:
      raise ValueError("Model needs to be trained before predicting signals")

    X,_ = self.feature_engineering(stock)
    predictions = self.model.predict(X)

    signals = pd.DataFrame(index=X.index)
    signals['Prediction'] = predictions
    signals['Position'] = signals['Prediction'].apply(lambda x: 1 if x > 0.55 else 0)

    # Calculate Signal as the change in position
    signals['Signal'] = 0
    signals.loc[signals['Position'] > signals['Position'].shift(1),'Signal'] = 1
    signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1
    signals['return'] = np.log(self.data[(stock,'Close')].loc[X.index] / self.data[(stock,'Close')].shift(1).loc[X.index])

    return signals





