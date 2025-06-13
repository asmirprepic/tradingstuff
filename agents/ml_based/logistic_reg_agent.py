from agents.base_agents.trading_agent import TradingAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class LRAgent(TradingAgent):
	"""
	A trading agent that uses the Logistic Regression classifier to generate trading signals.
	This agent employs a simple machine learning model based on price movement features to predict
	the direction of stock price movement.

	Attributes:
		algorithm_name (str): Name of the algorithm, set to "KNN".
		stocks_in_data (pd.Index): Unique stock symbols present in the data.
		signal_data (dict): A dictionary to store signal data for each stock.
	"""

	def __init__(self, data):
		super().__init__(data)
		self.algorithm_name = 'Logistic_Regression'
		self.stocks_in_data = self.data.columns.get_level_values(0).unique()

		self.models = {}
		self.signal_data = {}

	def create_classification_trading_condition(self, stock):
		"""
		Creates the feature set for the Logistic Regression model. The features include
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


	def train_model(self, stock, split_ratio=0.8):
		X, Y = self.create_classification_trading_condition(stock)
		X_train, X_test, Y_train, Y_test = self.create_train_split_group(X, Y, split_ratio)

		lr = LogisticRegression(max_iter=1000)
		lr.fit(X_train, Y_train)

		self.models[stock] = lr

		return lr, X_train, X_test, Y_train, Y_test


	def generate_signal_strategy(self, stock, mode = 'backtest'):
		"""
		Generates trading signals for the specified stock using the trained LR model.
		A signal is generated for each day based on the predicted direction of the stock price.

		Args:
			stock (str): The stock symbol for which to generate signals.

		The method updates the `signal_data` attribute with signals for the given stock.
		"""

		if stock not in self.models:
			lr, X_train, X_test, Y_train, Y_test = self.train_model(stock)
		else:
			lr = self.models[stock]
			X, Y = self.create_classification_trading_condition(stock)
			X_train, X_test, Y_train, Y_test = self.create_train_split_group(X, Y, split_ratio=0.8)

		if mode == 'backtest':
			X_pred = X_test
			index_used = X_test.index
			print(f"[{stock}] Generating BACKTEST signals on TEST set ({len(X_test)} samples).")
		elif mode == 'live':
			X, _ = self.create_classification_trading_condition(stock)
			X_pred = X
			index_used = X.index
			print(f"[{stock}] Generating LIVE signals on FULL set ({len(X)} samples).")
		else:
			raise ValueError("mode must be 'backtest' or 'live'.")

		predictions = lr.predict(X_pred)

		signals = pd.DataFrame(index=index_used)
		signals['Prediction'] = predictions
		signals['Position'] = signals['Prediction'].apply(lambda x: 1 if x == 1 else 0)

		signals['Signal'] = 0
		signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
		signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

		signals['return'] = np.log(self.data[(stock, 'Close')].loc[index_used] /
                                   self.data[(stock, 'Close')].shift(1).loc[index_used])

		self.signal_data[stock] = signals

	def generate_signals(self, stocks=None, mode='backtest'):
		"""
		Generates signals for selected stocks or all stocks.

		Args:
			stocks (list or str or None): Stocks to generate signals for.
											If None → all stocks.
			mode (str): 'backtest' or 'live'.
		"""
		if stocks is None:
			stocks = self.stocks_in_data
		elif isinstance(stocks, str):
			stocks = [stocks]

		for stock in stocks:
			if stock not in self.stocks_in_data:
				print(f"[Warning] Stock '{stock}' not found in data. Skipping.")
				continue
			print(f"Generating signals for {stock} ...")
			self.generate_signal_strategy(stock, mode=mode)






