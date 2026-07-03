import numpy as np
import pandas as pd

from agents.base_agents.sequential_based import SequentialNNAgent

try:
    from tensorflow.keras.layers import Conv1D, Dense, Flatten
    from tensorflow.keras.models import Sequential
except ModuleNotFoundError:
    Conv1D = Dense = Flatten = Sequential = None


class CNNAgent(SequentialNNAgent):
    def __init__(self, data, sequence_length=5, epochs=25, batch_size=32, verbose=0):
        super().__init__(data, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.algorithm_name = "CNN"
        self.sequence_length = sequence_length

    def feature_engineering(self, stock):
        df = self.data[stock].copy()
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Volatility"] = df["Return"].rolling(window=5).std()
        df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        df = df.iloc[:-1].dropna(subset=["Return", "Volatility"])

        features = df[["Return", "Volatility"]]
        target = df["Target"]
        return self.build_sequence_dataset(features, target, self.sequence_length)

    def build_model(self, input_shape):
        model = Sequential(
            [
                Conv1D(16, kernel_size=2, activation="relu", input_shape=input_shape),
                Flatten(),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
