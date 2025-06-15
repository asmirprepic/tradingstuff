from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from agents.base_agents.trading_agent import TradingAgent

class MLBasedAgent(TradingAgent, ABC):
    def __init__(self, data, model=None, features=None):
        super().__init__(data)
        self.model = model
        self.features = features
        self.algorithm_name = 'MLBaseAlgorithm'
        self.models = {}  # Per-stock models
        self.train_data = {}  # Per-stock train/test splits
        self.signal_data = {}  # Per-stock signals
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

    def default_feature_engineering(self, stock):
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
        pass

    @abstractmethod
    def generate_signal_strategy(self, stock, *args, **kwargs):
        pass

    def create_train_split_group(self, X, Y, split_ratio):
        return train_test_split(X, Y, shuffle=False, test_size=1 - split_ratio)

    def train_model(self, stock, split_ratio=0.8):
        if not self.model or not self.features:
            raise ValueError("Model and features must be defined.")

        X, Y = self.feature_engineering(stock)
        X_train, X_test, Y_train, Y_test = self.create_train_split_group(X, Y, split_ratio)

        model = clone(self.model)
        model.fit(X_train, Y_train)
        self.models[stock] = model
        self.train_data[stock] = (X_train, X_test, Y_train, Y_test)

        Y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(Y_test, Y_pred),
            'precision': precision_score(Y_test, Y_pred, zero_division=0),
            'recall': recall_score(Y_test, Y_pred, zero_division=0),
            'f1_score': f1_score(Y_test, Y_pred, zero_division=0)
        }

        print(f"\nModel Performance for {stock} ({self.algorithm_name}):")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        return metrics

    def predict_signals(self, stock, mode='backtest', threshold=0.5):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} has not been trained.")

        model = self.models[stock]
        X, _ = self.feature_engineering(stock)

        if mode == 'backtest':
            if stock not in self.train_data:
                raise ValueError(f"Training data for {stock} is missing.")
            X_pred = self.train_data[stock][1]  # X_test
            index_used = X_pred.index
        elif mode == 'live':
            X_pred = X
            index_used = X.index
        else:
            raise ValueError("mode must be 'backtest' or 'live'")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_pred)[:, 1]
            predictions = np.where(prob > threshold, 1, -1)
        else:
            predictions = model.predict(X_pred)

        signals = pd.DataFrame(index=index_used)
        signals['Prediction'] = predictions
        signals['Position'] = (signals['Prediction'] == 1).astype(int)
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        close = self.data[(stock, 'Close')]
        signals['return'] = np.log(close / close.shift(1)).reindex(index_used)
        return signals
