from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from agents.base_agents.trading_agent import TradingAgent


class SequentialNNAgent(TradingAgent, ABC):
    """
    Base class for CNN/LSTM/Transformer agents using 3D Keras models.
    Handles training and signal generation for time-series models.
    Expects input shape: (samples, timesteps or features, channels).
    """

    def __init__(self, data, split_ratio=0.8, batch_size=32, epochs=20, verbose=0):
        super().__init__(data)
        self.algorithm_name = "SequentialNN"
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.models = {}
        self.train_data = {}
        self.signal_data = {}

    @abstractmethod
    def feature_engineering(self, stock):
        """
        Returns:
            - X: np.ndarray (samples, timesteps/features, channels)
            - y: np.ndarray or pd.Series (binary labels)
        """
        pass

    @abstractmethod
    def build_model(self, input_shape):
        """
        Returns a compiled Keras model.
        """
        pass

    def train_model(self, stock):
        X, y = self.feature_engineering(stock)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=False, test_size=1 - self.split_ratio
        )

        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0) + 1e-8  # prevent divide-by-zero

        X_train_norm = (X_train - mu) / sigma
        X_test_norm = (X_test - mu) / sigma

        model = self.build_model(input_shape=X_train.shape[1:])

        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(
            X_train_norm, y_train,
            validation_data=(X_test_norm, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[early_stopping]
        )

        self.models[stock] = model
        self.train_data[stock] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "mu": mu,
            "sigma": sigma
        }

    def predict_signals(self, stock, mode='backtest', threshold=0.5):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} not trained.")

        model = self.models[stock]
        data = self.train_data[stock]
        mu, sigma = data['mu'], data['sigma']

        if mode == 'backtest':
            X_pred = data['X_test']
            index_used = pd.date_range(end=self.data[stock].index[-1], periods=len(X_pred))

        elif mode == 'live':
            X_full, _ = self.feature_engineering(stock)
            X_pred = X_full[-1:].copy()
            index_used = pd.Index([self.data[stock].index[-1]])

        else:
            raise ValueError("mode must be 'backtest' or 'live'")

        X_norm = (X_pred - mu) / sigma

        probs = model.predict(X_norm, verbose=0).flatten()
        predictions = (probs > threshold).astype(int)

        signals = pd.DataFrame(index=index_used)
        signals['Prediction'] = predictions
        signals['Position'] = predictions
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        close = self.data[(stock, 'Close')].reindex(index_used)
        signals['return'] = np.log(close / close.shift(1))
        return signals

    def walk_forward_predict(self, stock, initial_train_size=100, step_size=1):
        X, y = self.feature_engineering(stock)
        predictions = []
        indices = []

        for start in range(initial_train_size, len(X) - step_size + 1, step_size):
            X_train = X[:start]
            y_train = y[:start]
            X_test = X[start:start+step_size]

            # Normalize based on train
            mu = X_train.mean(axis=0)
            sigma = X_train.std(axis=0) + 1e-8
            X_train_norm = (X_train - mu) / sigma
            X_test_norm = (X_test - mu) / sigma

            model = self.build_model(input_shape=X_train.shape[1:])
            model.fit(
                X_train_norm, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                callbacks=[EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)]
            )

            prob = model.predict(X_test_norm, verbose=0).flatten()
            pred = (prob > 0.5).astype(int)

            predictions.extend(pred)
            indices.extend(self.data[stock].index[start:start+step_size])

        # Assemble final signal DataFrame
        signals = pd.DataFrame(index=indices)
        signals['Prediction'] = predictions
        signals['Position'] = predictions
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        close = self.data[(stock, 'Close')].reindex(signals.index)
        signals['return'] = np.log(close / close.shift(1))

        return signals

    def run_all_walk_forward(self, initial_train_size=100, step_size=1):
        for stock in self.stocks_in_data:
            try:
                print(f"[WALK-FORWARD] {stock}")
                self.signal_data[stock] = self.walk_forward_predict(stock, initial_train_size, step_size)
            except Exception as e:
                print(f"[WARNING] {stock} failed: {e}")
        self.calculate_returns()

    def run_all(self, mode='backtest'):
        """
        Trains and generates signals for all stocks in data.
        """
        for stock in self.stocks_in_data:
            self.train_model(stock)
            self.signal_data[stock] = self.predict_signals(stock, mode=mode)
        self.calculate_returns()

    @abstractmethod
    def generate_signal_strategy(self, stock, *args, **kwargs):
        pass
