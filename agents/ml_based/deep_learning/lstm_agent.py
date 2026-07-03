import numpy as np
import pandas as pd

from agents.base_agents.sequential_based import SequentialNNAgent

try:
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.models import Sequential
except ModuleNotFoundError:
    Dense = LSTM = Sequential = None


class LSTMAgent(SequentialNNAgent):
    """
    LSTM-based trading agent built on the shared sequential agent pipeline.
    """

    def __init__(self, data, sequence_length=10, lstm_units=50, epochs=20, batch_size=32, verbose=0):
        super().__init__(data, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.algorithm_name = "LSTM"
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units

    def feature_engineering(self, stock):
        df = self.data[stock].copy()
        df["Open-Close"] = df["Open"] - df["Close"]
        df["High-Low"] = df["High"] - df["Low"]
        df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        df = df.iloc[:-1].dropna(subset=["Open-Close", "High-Low"])

        features = df[["Open-Close", "High-Low"]]
        target = df["Target"]
        return self.build_sequence_dataset(features, target, self.sequence_length)

    def build_model(self, input_shape):
        model = Sequential(
            [
                LSTM(units=self.lstm_units, return_sequences=True, input_shape=input_shape),
                LSTM(units=self.lstm_units),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
