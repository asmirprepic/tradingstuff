import numpy as np
import pandas as pd

from agents.base_agents.sequential_based import SequentialNNAgent

try:
    import tensorflow as tf
    from tensorflow.keras import layers
except ModuleNotFoundError:
    tf = None
    layers = None


class TransformerAgent(SequentialNNAgent):
    """
    Transformer-based trading agent built on the shared sequential agent pipeline.
    """

    def __init__(
        self,
        data,
        sequence_length=10,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=None,
        dropout=0.25,
        mlp_dropout=0.4,
        epochs=20,
        batch_size=32,
        verbose=0,
    ):
        super().__init__(data, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.algorithm_name = "Transformer"
        self.sequence_length = sequence_length
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units or [128]
        self.dropout = dropout
        self.mlp_dropout = mlp_dropout

    def feature_engineering(self, stock):
        df = self.data[stock].copy()
        df["Open-Close"] = df["Open"] - df["Close"]
        df["High-Low"] = df["High"] - df["Low"]
        df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        df = df.iloc[:-1].dropna(subset=["Open-Close", "High-Low"])

        features = df[["Open-Close", "High-Low"]]
        target = df["Target"]
        return self.build_sequence_dataset(features, target, self.sequence_length)

    def transformer_encoder(self, inputs):
        x = layers.MultiHeadAttention(
            key_dim=self.head_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )(inputs, inputs)
        x = layers.Dropout(self.dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)

        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
