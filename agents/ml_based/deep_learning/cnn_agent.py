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

    def _build_feature_frame(self, stock):
        df = self.data[stock].copy()
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Volatility"] = df["Return"].rolling(window=5).std()
        next_close = df["Close"].shift(-1)
        df["Target"] = np.where(next_close.isna(), np.nan, np.where(next_close > df["Close"], 1, 0))
        return df

    def feature_engineering(self, stock):
        df = self._build_feature_frame(stock).dropna(subset=["Return", "Volatility", "Target"])

        features = df[["Return", "Volatility"]]
        target = df["Target"].astype(int)
        return self.build_sequence_dataset(features, target, self.sequence_length)

    def feature_engineering_live(self, stock):
        df = self._build_feature_frame(stock).dropna(subset=["Return", "Volatility"])
        features = df[["Return", "Volatility"]]
        placeholder_target = pd.Series(0, index=features.index, dtype=int)
        X, _, index = self.build_sequence_dataset(features, placeholder_target, self.sequence_length)
        return X, index

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
