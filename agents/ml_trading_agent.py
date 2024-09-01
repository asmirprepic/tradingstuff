from abc import ABC, abstracthmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from agents.trading_agent import TradingAgent

class MLTradingAgent(TradingAgent): 
  """
  Abstract base class for machine-learning based trading algorithms.

  This class extends the TradingAgent class to include methods specific to machine learning
  models. 

  Attributes: 
  ---------------
  model (object): The machine learning model to be used for trading
  features (list): A list of feature column names used for model training.

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
    self.feature = features
    self.trained = False

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

    pass 


  def train_model(self,stock,test_size = 0.2)
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

    features_df = self.feature_engineering(stock)
    X = features_df[self.features]
    y = 
