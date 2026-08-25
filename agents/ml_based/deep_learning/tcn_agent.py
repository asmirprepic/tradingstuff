import numpy as np
import pandas as pd

from agents.base_agents.sequential_based import SequentialNNAgent

try:
    import tensorflow as tf
    from tensorflow.keras import layers
except ModuleNotFoundError:
    tf = None
    layers = None


class TCNAgent(SequentialNNAgent):
    """
    Temporal Convolutional Network trading agent built on the shared sequential pipeline.
    """

    def __init__(
        self,
        data,
        sequence_length=20,
        kernel_size=3,
        filters=32,
        num_layers=2,
        dense_units=32,
        dropout=0.1,
        epochs=20,
        batch_size=32,
        verbose=0,
    ):
        super().__init__(data, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.algorithm_name = "TCN"
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.filters = filters
        self.num_layers = num_layers
        self.dense_units = dense_units
        self.dropout = dropout

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain.div(loss.replace(0, np.nan))
        return 100 - (100 / (1 + rs))

    def _build_feature_frame(self, stock):
        df = self.data[stock].copy()
        df["Open-Close"] = df["Open"] - df["Close"]
        df["High-Low"] = df["High"] - df["Low"]
        df["SMA-10"] = df["Close"].rolling(window=10).mean()
        df["SMA-50"] = df["Close"].rolling(window=50).mean()
        df["Momentum"] = df["Close"] - df["Close"].shift(10)
        df["RSI"] = self._calculate_rsi(df["Close"])
        df["Volume_Change"] = df["Volume"].pct_change()
        df = df.replace([np.inf, -np.inf], np.nan).ffill()

        next_close = df["Close"].shift(-1)
        df["Target"] = np.where(next_close.isna(), np.nan, np.where(next_close > df["Close"], 1, 0))
        return df

    def feature_engineering(self, stock):
        df = self._build_feature_frame(stock).dropna(
            subset=["Open-Close", "High-Low", "SMA-10", "SMA-50", "Momentum", "RSI", "Volume_Change", "Target"]
        )
        features = df[["Open-Close", "High-Low", "SMA-10", "SMA-50", "Momentum", "RSI", "Volume_Change"]]
        target = df["Target"].astype(int)
        return self.build_sequence_dataset(features, target, self.sequence_length)

    def feature_engineering_live(self, stock):
        df = self._build_feature_frame(stock).dropna(
            subset=["Open-Close", "High-Low", "SMA-10", "SMA-50", "Momentum", "RSI", "Volume_Change"]
        )
        features = df[["Open-Close", "High-Low", "SMA-10", "SMA-50", "Momentum", "RSI", "Volume_Change"]]
        placeholder_target = pd.Series(0, index=features.index, dtype=int)
        X, _, index = self.build_sequence_dataset(features, placeholder_target, self.sequence_length)
        return X, index

    def _tcn_block(self, x, dilation_rate):
        residual = x
        y = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
        )(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(self.dropout)(y)
        y = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
        )(y)
        y = layers.BatchNormalization()(y)

        if residual.shape[-1] != self.filters:
            residual = layers.Conv1D(self.filters, kernel_size=1, padding="same")(residual)

        y = layers.Add()([y, residual])
        y = layers.ReLU()(y)
        return y

    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for i in range(self.num_layers):
            x = self._tcn_block(x, dilation_rate=2 ** i)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(self.dense_units, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
