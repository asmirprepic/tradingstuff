from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class KNNAgent(TradingAgent):
  """
  A trading agent that uses the K-Nearest Neighbors (KNN) classifier to generate trading signals.
  This agent employs a simple machine learning model based on price movement features to predict
  the direction of stock price movement.

  Attributes:
      algorithm_name (str): Name of the algorithm, set to "KNN".
      stocks_in_data (pd.Index): Unique stock symbols present in the data.
      signal_data (dict): A dictionary to store signal data for each stock.
  """

  def __init__(self, data):
      super().__init__(data)
      self.algorithm_name = 'KNN'
      self.stocks_in_data = self.data.columns.get_level_values(0).unique()

      for stock in self.stocks_in_data:
          self.generate_signal_strategy(stock)
      self.calculate_returns()

  def create_classification_trading_condition(self, stock):
      """
      Creates the feature set for the KNN classification model. The features include 
      'Open-Close' (difference between opening and closing prices) and 'High-Low' 
      (difference between high and low prices).

      Args:
          stock (str): The stock symbol for which to create features.

      Returns:
          pd.DataFrame: The feature set.
          pd.Series: The target variable, where 1 indicates an upward price movement 
                      and -1 indicates a downward price movement.
      """
      data_copy = self.data[stock].copy()
      data_copy['Open-Close'] = self.data[stock]['Open'] - self.data[stock]['Close']
      data_copy['High-Low'] = self.data[stock]['High'] - self.data[stock]['Low']
      data_copy.dropna(inplace=True)

      X = data_copy[['Open-Close', 'High-Low']]
      Y = np.where(self.data[stock]['Close'].shift(-1) > self.data[stock]['Close'], 1, -1)
      Y_series = pd.Series(Y, index=self.data[stock].index)
      Y = Y_series.loc[X.index]

      return X, Y

  def create_train_split_group(self, X, Y, split_ratio):
      """
      Splits the dataset into training and testing sets.

      Args:
          X (pd.DataFrame): The feature set.
          Y (pd.Series): The target variable.
          split_ratio (float): The proportion of the dataset to include in the train split.

      Returns:
          tuple: The training and testing sets (X_train, X_test, Y_train, Y_test).
      """
      return train_test_split(X, Y, shuffle=False, test_size=1 - split_ratio)

  def KNN_model(self, stock):
      """
      Trains the KNN model for the specified stock.

      Args:
          stock (str): The stock symbol for which to train the model.

      Returns:
          KNeighborsClassifier: The trained KNN model.
      """
      X, Y = self.create_classification_trading_condition(stock)
      X_train, X_test, Y_train, Y_test = self.create_train_split_group(X, Y, split_ratio=0.8)
      knn = KNeighborsClassifier(n_neighbors=15)
      knn.fit(X_train, Y_train)

      return knn

  def generate_signal_strategy(self, stock):
      """
      Generates trading signals for the specified stock using the trained KNN model.
      A signal is generated for each day based on the predicted direction of the stock price.

      Args:
          stock (str): The stock symbol for which to generate signals.

      The method updates the `signal_data` attribute with signals for the given stock.
      """
      signals = pd.DataFrame(index=self.data.index)
      X, _ = self.create_classification_trading_condition(stock)
      knn = self.KNN_model(stock)
      prediction = knn.predict(X)
      signals = signals.loc[X.index]
      signals['Prediction'] = prediction
      signals['Position'] = signals['Prediction'].apply(lambda x: 1 if x == 1 else 0)
      
      # Calculate Signal as the change in position
      signals['Signal'] = 0
      signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
      signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

      
      signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))
      
      self.signal_data[stock] = signals








    




