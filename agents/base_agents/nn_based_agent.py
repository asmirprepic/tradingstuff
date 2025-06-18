from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from agents.base_agents.trading_agent import TradingAgent


class NNBasedAgent(TradingAgent, ABC):
    """
    Abstract base class for neural network-based trading agents.

    Subclasses must implement:
    - feature_engineering(stock): Returns X, y, and list of feature columns.
    - build_model(input_shape): Returns a compiled Keras model.
    """

    def __init__(self, data, split_ratio=0.8, batch_size=32, epochs=20, verbose=0):
        super().__init__(data)
        self.algorithm_name = "NNBaseAlgorithm"
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
        Must return: X (DataFrame), y (Series), feature_cols (list of str)
        """
        pass

    @abstractmethod
    def build_model(self, input_shape):
        """
        Must return a compiled Keras model.
        """
        pass

    def train_model(self, stock):
        X, y, feature_cols = self.feature_engineering(stock)

        X_train, X_test, y_train, y_test = train_test_split(
            X[feature_cols], y, shuffle=False, test_size=1 - self.split_ratio
        )

        mu, sigma = X_train.mean(), X_train.std()
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

        model = self.build_model(input_shape=(len(feature_cols),))

        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[early_stopping]
        )

        self.models[stock] = model
        self.train_data[stock] = (X_train, X_test, y_train, y_test, feature_cols)

        print(f"[{stock}] NN model trained ({self.algorithm_name})")

    def predict_signals(self, stock, mode='backtest', threshold=0.5):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} not trained.")

        model = self.models[stock]
        _, _, _, _, feature_cols = self.train_data[stock]
        X_all, y_all, _ = self.feature_engineering(stock)

        if mode == 'backtest':
            X_pred = self.train_data[stock][1]
        elif mode == 'live':
            X_pred = X_all[feature_cols]
        else:
            raise ValueError("mode must be 'backtest' or 'live'")


        X_train = self.train_data[stock][0]
        mu, sigma = X_train.mean(), X_train.std()
        X_pred_norm = (X_pred - mu) / sigma

        probs = model.predict(X_pred_norm, verbose=0).flatten()
        predictions = (probs > threshold).astype(int)
        signals = pd.DataFrame(index=X_pred.index)
        signals['Prediction'] = predictions
        signals['Position'] = predictions
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        close = self.data[(stock, 'Close')]
        signals['return'] = np.log(close / close.shift(1)).reindex(X_pred.index)

        return signals

    @abstractmethod
    def generate_signal_strategy(self, stock, *args, **kwargs):
        pass
