from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from agents.base_agents.trading_agent import TradingAgent

try:
    from tensorflow.keras.callbacks import EarlyStopping
except ModuleNotFoundError as exc:
    EarlyStopping = None
    _TENSORFLOW_IMPORT_ERROR = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None


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
        self.score_column = "SignalStrength"
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

    def _require_tensorflow(self):
        if EarlyStopping is None:
            raise ModuleNotFoundError(
                "tensorflow is required to train neural-network agents. "
                "Install tensorflow before using NNBasedAgent subclasses."
            ) from _TENSORFLOW_IMPORT_ERROR

    def create_train_split_group(self, X, y, split_ratio=None, **kwargs):
        kwargs.setdefault("shuffle", False)
        if split_ratio is not None and "test_size" not in kwargs:
            kwargs["test_size"] = 1 - split_ratio
        return train_test_split(X, y, **kwargs)

    def train_model(self, stock):
        self._require_tensorflow()
        X, y, feature_cols = self.feature_engineering(stock)

        X_train, X_test, y_train, y_test = self.create_train_split_group(
            X[feature_cols], y, split_ratio=self.split_ratio
        )

        mu = X_train.mean()
        sigma = X_train.std().replace(0, 1.0).fillna(1.0)
        X_train_norm = (X_train - mu) / sigma
        X_test_norm = (X_test - mu) / sigma

        model = self.build_model(input_shape=(len(feature_cols),))

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
            "feature_cols": feature_cols,
            "mu": mu,
            "sigma": sigma,
        }

        print(f"[{stock}] NN model trained ({self.algorithm_name})")

    def predict_signals(self, stock, mode='backtest', threshold=0.5):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} not trained.")

        model = self.models[stock]
        train_data = self.train_data[stock]
        feature_cols = train_data["feature_cols"]
        X_all, y_all, _ = self.feature_engineering(stock)

        if mode == 'backtest':
            X_pred = train_data["X_test"]
        elif mode == 'live':
            X_pred = X_all[feature_cols]
        else:
            raise ValueError("mode must be 'backtest' or 'live'")


        mu = train_data["mu"]
        sigma = train_data["sigma"]
        X_pred_norm = (X_pred - mu) / sigma

        probs = model.predict(X_pred_norm, verbose=0).flatten()
        predictions = (probs > threshold).astype(int)
        signals = pd.DataFrame(index=X_pred.index)
        signals['Prediction'] = predictions
        signals["ProbUp"] = probs
        signals["SignalStrength"] = probs
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


    def walk_forward_predict(self,stock,initial_train_size = 100,step_size = 1, threshold = 0.5):
        self._require_tensorflow()
        X,y,feature_cols = self.feature_engineering(stock)
        X = X[feature_cols]

        predictions = []
        probabilities = []
        indices = []

        for start in range(initial_train_size,len(X) - step_size + 1, step_size):
            X_train = X.iloc[:start]
            y_train = y.iloc[:start]
            X_test = X.iloc[start:start + step_size]

            mu = X_train.mean()
            sigma = X_train.std().replace(0, 1.0).fillna(1.0)
            X_train_norm = (X_train - mu) / sigma
            X_test_norm = (X_test - mu) / sigma

            model = self.build_model(input_shape=(len(feature_cols),))
            model.fit(
                X_train_norm, y_train,
                epochs = self.epochs,
                batch_size = self.batch_size,
                verbose = self.verbose,
                callbacks = [EarlyStopping(monitor = 'loss',patience = 3,restore_best_weights = True)]

            )

            prob = model.predict(X_test_norm, verbose = 0).flatten()
            pred = (prob > threshold).astype(int)
            predictions.extend(pred)
            probabilities.extend(prob)
            indices.extend(X_test.index)

        signals = pd.DataFrame(index = indices)
        signals['Prediction'] = predictions
        signals["ProbUp"] = probabilities
        signals["SignalStrength"] = probabilities
        signals['Position'] = predictions
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1),'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1),'Signal'] = -1

        close = self.data[(stock,'Close')].reindex(signals.index)
        signals['return'] = np.log(close/close.shift(1))

        return signals

    def run_all_walk_forward(self, initial_train_size=100, step_size=1, **kwargs):
        if "intial_train_size" in kwargs:
            initial_train_size = kwargs.pop("intial_train_size")
        for stock in self.stocks_in_data:
            try:
                print(f"[WALK FORWARD] {stock}")
                self.signal_data[stock] = self.walk_forward_predict(stock, initial_train_size, step_size, **kwargs)
            except Exception as e:
                print(f"[WARNING] {stock} failed: {e}")
        self.calculate_returns()
