import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agents.base_agents.trading_agent import TradingAgent

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Dense, Input, Lambda
    from tensorflow.keras.models import Model
except ModuleNotFoundError as exc:
    tf = K = Dense = Input = Lambda = Model = None
    _TENSORFLOW_IMPORT_ERROR = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None


class VAEAgent(TradingAgent):
    def __init__(
        self,
        data,
        split_ratio=0.8,
        batch_size=32,
        epochs=50,
        hidden_dim=32,
        latent_dim=2,
        verbose=0,
        anomaly_threshold_percentile=95,
        position_on_anomaly=-1,
    ):
        super().__init__(data)
        self.algorithm_name = "VAE"
        self.score_column = "SignalStrength"
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.verbose = verbose
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        self.position_on_anomaly = position_on_anomaly

        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.thresholds = {}
        self.train_data = {}

    def _require_tensorflow(self):
        if Model is None:
            raise ModuleNotFoundError(
                "tensorflow is required to train VAEAgent. "
                "Install tensorflow before using this agent."
            ) from _TENSORFLOW_IMPORT_ERROR

    def feature_engineering(self, stock):
        df = self.data[stock].copy()
        df["Open-Close"] = df["Open"] - df["Close"]
        df["High-Low"] = df["High"] - df["Low"]
        df["Close"] = df["Close"]
        df = df.dropna(subset=["Open-Close", "High-Low", "Close"])
        return df[["Open-Close", "High-Low", "Close"]]

    def create_train_split_group(self, X, index, split_ratio=None, **kwargs):
        kwargs.setdefault("shuffle", False)
        if split_ratio is not None and "test_size" not in kwargs:
            kwargs["test_size"] = 1 - split_ratio
        return train_test_split(X, index, **kwargs)

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build_model(self, input_dim):
        inputs = Input(shape=(input_dim,))
        hidden = Dense(self.hidden_dim, activation="relu")(inputs)

        z_mean = Dense(self.latent_dim, name="z_mean")(hidden)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(hidden)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name="sampling")([z_mean, z_log_var])

        decoder_hidden = Dense(self.hidden_dim, activation="relu")(z)
        outputs = Dense(input_dim, activation="linear")(decoder_hidden)

        vae = Model(inputs=inputs, outputs=outputs)
        encoder = Model(inputs=inputs, outputs=z_mean)

        def vae_loss(x_true, x_pred):
            reconstruction_loss = tf.reduce_sum(tf.square(x_true - x_pred), axis=1)
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
            return reconstruction_loss + kl_loss

        vae.compile(optimizer="adam", loss=vae_loss)
        return vae, encoder

    def train_model(self, stock, anomaly_threshold_percentile=None):
        self._require_tensorflow()
        percentile = (
            self.anomaly_threshold_percentile
            if anomaly_threshold_percentile is None
            else anomaly_threshold_percentile
        )

        X = self.feature_engineering(stock)
        X_train, X_test, index_train, index_test = self.create_train_split_group(
            X,
            X.index,
            split_ratio=self.split_ratio,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model, encoder = self.build_model(input_dim=X_train_scaled.shape[1])
        fit_kwargs = {
            "x": X_train_scaled,
            "y": X_train_scaled,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "shuffle": False,
            "verbose": self.verbose,
        }
        if len(X_test_scaled) > 0:
            fit_kwargs["validation_data"] = (X_test_scaled, X_test_scaled)
        model.fit(**fit_kwargs)

        train_pred = model.predict(X_train_scaled, verbose=0)
        train_mse = np.mean(np.power(X_train_scaled - train_pred, 2), axis=1)
        threshold = float(np.percentile(train_mse, percentile))

        self.models[stock] = model
        self.encoders[stock] = encoder
        self.scalers[stock] = scaler
        self.thresholds[stock] = threshold
        self.train_data[stock] = {
            "X_train": X_train,
            "X_test": X_test,
            "index_train": pd.Index(index_train),
            "index_test": pd.Index(index_test),
            "percentile": percentile,
        }
        return threshold

    def predict_signals(self, stock, mode="backtest"):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} has not been trained.")
        if stock not in self.train_data:
            raise ValueError(f"Training data for {stock} is missing.")

        scaler = self.scalers[stock]
        model = self.models[stock]
        threshold = self.thresholds[stock]
        train_data = self.train_data[stock]

        if mode == "backtest":
            X_pred = train_data["X_test"]
            index_used = train_data["index_test"]
        elif mode == "live":
            X_full = self.feature_engineering(stock)
            X_pred = X_full.iloc[[-1]]
            index_used = pd.Index([X_full.index[-1]])
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

        X_scaled = scaler.transform(X_pred)
        reconstructed = model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        anomaly = mse > threshold
        strength = mse / threshold if threshold > 0 else mse
        position = np.where(anomaly, self.position_on_anomaly, 0)

        signals = pd.DataFrame(index=index_used)
        signals["Anomaly"] = anomaly
        signals["ReconstructionError"] = mse
        signals["SignalStrength"] = strength
        signals["Position"] = position
        signals["Signal"] = 0
        signals.loc[signals["Position"] > signals["Position"].shift(1), "Signal"] = 1
        signals.loc[signals["Position"] < signals["Position"].shift(1), "Signal"] = -1
        close = self.data[(stock, "Close")]
        signals["return"] = np.log(close / close.shift(1)).reindex(index_used)
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
