from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from agents.base_agents.trading_agent import TradingAgent


class MLBasedAgent(TradingAgent, ABC):
    """
    Abstract base class for machine-learning based trading algorithms.
    """

    def __init__(self, data, model=None, features=None):
        """
        Initialize the MLBasedAgent with data and ML model.

        Args:
            data (pd.DataFrame): MultiIndex DataFrame with level 0 = stock, level 1 = price columns.
            model (object): Any sklearn-compatible ML model.
            features (list): List of column names to use as features.
        """
        super().__init__(data)
        self.algorithm_name = 'MLBaseAlgorithm'
        self.model = model
        self.features = features
        self.trained = False

    def default_feature_engineering(self, stock):
        """
        Basic feature engineering with Open-Close and High-Low differences.
        """
        df = self.data[stock].copy()
        df['Open-Close'] = df['Open'] - df['Close']
        df['High-Low'] = df['High'] - df['Low']
        df = df.ffill()

        X = df[self.features].copy()
        Y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
        Y = pd.Series(Y, index=df.index)

        return X, Y

    @abstractmethod
    def feature_engineering(self, stock):
        """
        Subclasses must override this method or rely on default.
        """
        return self.default_feature_engineering(stock)

    def create_train_split_group(self, X, Y, split_ratio):
        """
        Splits dataset chronologically (no shuffling).
        """
        return train_test_split(X, Y, shuffle=False, test_size=1 - split_ratio)

    def train_model(self, stock, test_size=0.2):
        """
        Trains the ML model on one stock.
        """
        if not self.model or not self.features:
            raise ValueError("Both model and features must be provided.")

        X, y = self.feature_engineering(stock)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }

        print(f"\nModel Performance for {stock} ({self.algorithm_name}):")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        self.trained = True
        return metrics

    def predict_signals(self, stock, threshold=0.5):
        """
        Uses the trained model to predict signals.

        Args:
            stock (str): Stock symbol
            threshold (float): For probabilistic models (default = 0.5)

        Returns:
            pd.DataFrame: Signals with 'Prediction', 'Position', 'Signal', 'return'
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction.")

        X, _ = self.feature_engineering(stock)

        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X)[:, 1]
            predictions = np.where(prob > threshold, 1, -1)
        else:
            predictions = self.model.predict(X)

        signals = pd.DataFrame(index=X.index)
        signals['Prediction'] = predictions
        signals['Position'] = (predictions == 1).astype(int)

        # Entry signals
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        close = self.data[(stock, 'Close')]
        signals['return'] = np.log(close / close.shift(1)).reindex(X.index)

        return signals

    @abstractmethod
    def generate_signal_strategy(self, stock, *args):
        """
        Populate self.signal_data[stock] with predicted signals.
        """
        pass
