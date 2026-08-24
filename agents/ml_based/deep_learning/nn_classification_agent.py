import numpy as np
import pandas as pd

from agents.base_agents.nn_based_agent import NNBasedAgent

try:
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
except ModuleNotFoundError:
    Dense = Sequential = None


class DenseNNAgent(NNBasedAgent):
    def __init__(self, data):
        super().__init__(data, epochs=30)
        self.algorithm_name = "DenseNN"

    def _build_feature_frame(self, stock):
        df = self.data[stock].copy()
        df["Open-Close"] = df["Open"] - df["Close"]
        df["High-Low"] = df["High"] - df["Low"]
        df = df.ffill()
        next_close = df["Close"].shift(-1)
        df["Target"] = np.where(next_close.isna(), np.nan, np.where(next_close > df["Close"], 1, 0))
        return df

    def feature_engineering(self, stock):
        df = self._build_feature_frame(stock).dropna(subset=["Open-Close", "High-Low", "Target"])
        X = df[["Open-Close", "High-Low"]]
        y = df["Target"].astype(int)
        return X, y, ["Open-Close", "High-Low"]

    def live_feature_engineering(self, stock):
        df = self._build_feature_frame(stock).dropna(subset=["Open-Close", "High-Low"])
        return df[["Open-Close", "High-Low"]]

    def build_model(self, input_shape):
        model = Sequential([
            Dense(64, activation='relu', input_shape=input_shape),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def generate_signal_strategy(self, stock, mode='backtest'):
        self.train_model(stock)
        signals = self.predict_signals(stock, mode)
        self.signal_data[stock] = signals
        return signals
