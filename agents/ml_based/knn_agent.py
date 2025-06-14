from agents.base_agents.ml_trading_agent import MLBasedAgent
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

class KNNAgent(MLBasedAgent):
    """
    A trading agent using K-Nearest Neighbors (KNN) to generate trading signals.
    """

    def __init__(self, data, n_neighbors=15):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        features = ['Open-Close', 'High-Low']
        super().__init__(data, model=model, features=features)
        self.algorithm_name = 'KNN'

    def feature_engineering(self, stock):
        """
        Feature engineering using 'Open-Close' and 'High-Low' features.
        """
        return self.default_feature_engineering(stock)

    def generate_signal_strategy(self, stock, mode='backtest'):
        """
        Trains model (if needed) and generates signals for a single stock.
        """
        print(f"[{stock}] Running generate_signal_strategy in {mode} mode.")

        self.train_model(stock)

        signals = self.predict_signals(stock)
        self.signal_data[stock] = signals

    def generate_signals(self, stocks=None):
        """
        Generates signals for a list of stocks or all if not specified.
        """
        if stocks is None:
            stocks = self.stocks_in_data
        elif isinstance(stocks, str):
            stocks = [stocks]

        for stock in stocks:
            if stock not in self.stocks_in_data:
                print(f"[Warning] Stock {stock} not found in data. Skipping.")
                continue
            self.generate_signal_strategy(stock)
