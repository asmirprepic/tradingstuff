import numpy as np
import pandas as pd

from agents.base_agents.sequential_based import SequentialNNAgent

try:
    import tensorflow as tf
    from tensorflow.keras import layers
except ModuleNotFoundError:
    tf = None
    layers = None


class LSTMAttentionAgent(SequentialNNAgent):
    """
    LSTM agent with a simple attention pooling head built on the shared sequential pipeline.
    """

    def __init__(
        self,
        data,
        sequence_length=10,
        lstm_units=64,
        dense_units=50,
        dropout=0.1,
        epochs=20,
        batch_size=32,
        verbose=0,
    ):
        super().__init__(data, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.algorithm_name = "LSTM Attention"
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout

    def _build_feature_frame(self, stock):
        df = self.data[stock].copy()
        df["Open-Close"] = df["Open"] - df["Close"]
        df["High-Low"] = df["High"] - df["Low"]
        df["Close_Level"] = df["Close"]
        df["Volume"] = df["Volume"]
        df = df.replace([np.inf, -np.inf], np.nan).ffill()

        next_close = df["Close"].shift(-1)
        df["Target"] = np.where(next_close.isna(), np.nan, np.where(next_close > df["Close"], 1, 0))
        return df

    def feature_engineering(self, stock):
        df = self._build_feature_frame(stock).dropna(
            subset=["Open-Close", "High-Low", "Close_Level", "Volume", "Target"]
        )
        features = df[["Open-Close", "High-Low", "Close_Level", "Volume"]]
        target = df["Target"].astype(int)
        return self.build_sequence_dataset(features, target, self.sequence_length)

    def feature_engineering_live(self, stock):
        df = self._build_feature_frame(stock).dropna(subset=["Open-Close", "High-Low", "Close_Level", "Volume"])
        features = df[["Open-Close", "High-Low", "Close_Level", "Volume"]]
        placeholder_target = pd.Series(0, index=features.index, dtype=int)
        X, _, index = self.build_sequence_dataset(features, placeholder_target, self.sequence_length)
        return X, index

    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
        attention_scores = layers.Dense(1, activation="tanh")(x)
        attention_weights = layers.Softmax(axis=1)(attention_scores)
        context = layers.Multiply()([x, attention_weights])
        context = layers.Lambda(lambda tensor: tf.reduce_sum(tensor, axis=1))(context)
        context = layers.Dense(self.dense_units, activation="relu")(context)
        context = layers.Dropout(self.dropout)(context)
        outputs = layers.Dense(1, activation="sigmoid")(context)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
