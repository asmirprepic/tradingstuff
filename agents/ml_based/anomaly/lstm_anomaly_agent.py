import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agents.base_agents.trading_agent import TradingAgent

try:
    from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed
    from tensorflow.keras.models import Model
except ModuleNotFoundError as exc:
    Dense = Input = LSTM = RepeatVector = TimeDistributed = Model = None
    _TENSORFLOW_IMPORT_ERROR = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None


class LSTMAnomalyAgent(TradingAgent):
    def __init__(
        self,
        data,
        sequence_length=10,
        split_ratio=0.8,
        batch_size=32,
        epochs=20,
        latent_dim=32,
        verbose=0,
        anomaly_threshold_percentile=95,
        position_on_anomaly=-1,
    ):
        super().__init__(data)
        self.algorithm_name = "LSTMAnomaly"
        self.score_column = "SignalStrength"
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        self.sequence_length = sequence_length
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.verbose = verbose
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        self.position_on_anomaly = position_on_anomaly

        self.models = {}
        self.thresholds = {}
        self.train_data = {}

    def _require_tensorflow(self):
        if Model is None:
            raise ModuleNotFoundError(
                "tensorflow is required to train LSTMAnomalyAgent. "
                "Install tensorflow before using this agent."
            ) from _TENSORFLOW_IMPORT_ERROR

    def feature_engineering(self, stock):
        df = self.data[stock].copy()
        df["Open-Close"] = df["Open"] - df["Close"]
        df["High-Low"] = df["High"] - df["Low"]
        df["Volume_Change"] = df["Volume"].pct_change()
        df = df.dropna(subset=["Open-Close", "High-Low", "Volume_Change"])
        return df[["Open-Close", "High-Low", "Volume_Change"]]

    def build_sequence_dataset(self, features):
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
        if len(features) < self.sequence_length:
            raise ValueError(
                f"Not enough rows ({len(features)}) to build sequences of length {self.sequence_length}."
            )

        values = features.to_numpy(dtype=float)
        feature_index = features.index
        X = []
        index = []

        for end in range(self.sequence_length - 1, len(features)):
            start = end - self.sequence_length + 1
            X.append(values[start:end + 1])
            index.append(feature_index[end])

        return np.asarray(X, dtype=float), pd.Index(index)

    def create_train_split_group(self, X, index, split_ratio=None, **kwargs):
        kwargs.setdefault("shuffle", False)
        if split_ratio is not None and "test_size" not in kwargs:
            kwargs["test_size"] = 1 - split_ratio
        return train_test_split(X, index, **kwargs)

    def build_model(self, input_shape):
        timesteps, n_features = input_shape

        inputs = Input(shape=input_shape)
        encoded = LSTM(self.latent_dim, activation="tanh")(inputs)
        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(self.latent_dim, activation="tanh", return_sequences=True)(decoded)
        outputs = TimeDistributed(Dense(n_features))(decoded)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def train_model(self, stock, anomaly_threshold_percentile=None):
        self._require_tensorflow()
        percentile = (
            self.anomaly_threshold_percentile
            if anomaly_threshold_percentile is None
            else anomaly_threshold_percentile
        )

        features = self.feature_engineering(stock)
        X, index = self.build_sequence_dataset(features)
        X_train, X_test, index_train, index_test = self.create_train_split_group(
            X,
            index,
            split_ratio=self.split_ratio,
        )

        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - mu) / sigma
        X_test_norm = (X_test - mu) / sigma

        model = self.build_model(input_shape=X_train.shape[1:])
        fit_kwargs = {
            "x": X_train_norm,
            "y": X_train_norm,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "shuffle": False,
            "verbose": self.verbose,
        }
        if len(X_test_norm) > 0:
            fit_kwargs["validation_data"] = (X_test_norm, X_test_norm)
        model.fit(**fit_kwargs)

        train_pred = model.predict(X_train_norm, verbose=0)
        train_mse = np.mean(np.power(X_train_norm - train_pred, 2), axis=(1, 2))
        threshold = float(np.percentile(train_mse, percentile))

        self.models[stock] = model
        self.thresholds[stock] = threshold
        self.train_data[stock] = {
            "X_train": X_train,
            "X_test": X_test,
            "index_train": pd.Index(index_train),
            "index_test": pd.Index(index_test),
            "mu": mu,
            "sigma": sigma,
            "percentile": percentile,
        }
        return threshold

    def predict_signals(self, stock, mode="backtest"):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} has not been trained.")
        if stock not in self.train_data:
            raise ValueError(f"Training data for {stock} is missing.")

        model = self.models[stock]
        threshold = self.thresholds[stock]
        train_data = self.train_data[stock]
        mu = train_data["mu"]
        sigma = train_data["sigma"]

        if mode == "backtest":
            X_pred = train_data["X_test"]
            index_used = train_data["index_test"]
        elif mode == "live":
            features = self.feature_engineering(stock)
            X_full, feature_index = self.build_sequence_dataset(features)
            X_pred = X_full[-1:].copy()
            index_used = pd.Index([feature_index[-1]])
        else:
            raise ValueError("mode must be 'backtest' or 'live'")

        if len(X_pred) == 0:
            return pd.DataFrame(
                columns=[
                    "Anomaly",
                    "ReconstructionError",
                    "SignalStrength",
                    "Position",
                    "Signal",
                    "return",
                ]
            )

        X_norm = (X_pred - mu) / sigma
        X_reconstructed = model.predict(X_norm, verbose=0)
        reconstruction_error = np.mean(np.power(X_norm - X_reconstructed, 2), axis=(1, 2))
        anomaly = reconstruction_error > threshold
        strength = reconstruction_error / threshold if threshold > 0 else reconstruction_error
        position = np.where(anomaly, self.position_on_anomaly, 0)

        signals = pd.DataFrame(index=index_used)
        signals["Anomaly"] = anomaly
        signals["ReconstructionError"] = reconstruction_error
        signals["SignalStrength"] = strength
        signals["Position"] = position
        signals["Signal"] = 0
        signals.loc[signals["Position"] > signals["Position"].shift(1), "Signal"] = 1
        signals.loc[signals["Position"] < signals["Position"].shift(1), "Signal"] = -1

        close = self.data[(stock, "Close")].reindex(index_used)
        signals["return"] = np.log(close / close.shift(1))
        return signals

    def generate_signal_strategy(self, stock, mode="backtest", anomaly_threshold_percentile=None):
        target_percentile = (
            self.anomaly_threshold_percentile
            if anomaly_threshold_percentile is None
            else anomaly_threshold_percentile
        )

        needs_retrain = (
            stock not in self.models
            or stock not in self.train_data
            or self.train_data[stock].get("percentile") != target_percentile
        )

        if needs_retrain:
            self.train_model(stock, anomaly_threshold_percentile=target_percentile)

        signals = self.predict_signals(stock, mode=mode)
        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest", anomaly_threshold_percentile=None):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.signal_data[stock] = self.generate_signal_strategy(
                stock,
                mode=mode,
                anomaly_threshold_percentile=anomaly_threshold_percentile,
            )
        self.calculate_returns()

    def generate_signals_for_all(self, anomaly_threshold_percentile=95, mode="backtest"):
        self.run_all(mode=mode, anomaly_threshold_percentile=anomaly_threshold_percentile)
