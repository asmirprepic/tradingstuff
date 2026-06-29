import unittest

import numpy as np
import pandas as pd

from agents.base_agents.nn_based_agent import NNBasedAgent
from agents.base_agents.sequential_based import SequentialNNAgent
from agents.ml_based.logistic_reg_agent import LRAgent


def make_market_data(stock="AAA", periods=60):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    close = np.linspace(100, 130, periods) + np.sin(np.arange(periods))
    columns = pd.MultiIndex.from_product([[stock], ["Open", "High", "Low", "Close", "Volume"]])
    data = pd.DataFrame(index=index, columns=columns, dtype=float)
    data[(stock, "Close")] = close
    data[(stock, "Open")] = close + 0.5
    data[(stock, "High")] = close + 1.0
    data[(stock, "Low")] = close - 1.0
    data[(stock, "Volume")] = np.arange(periods) + 1000
    return data


class DummyPredictModel:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict(self, X, verbose=0):
        return self._probs.reshape(-1, 1)


class DummyNNAgent(NNBasedAgent):
    def feature_engineering(self, stock):
        df = self.data[stock].copy()
        x = pd.DataFrame(
            {
                "f1": np.linspace(0.0, 1.0, len(df)),
                "f2": np.linspace(1.0, 2.0, len(df)),
            },
            index=df.index,
        )
        y = pd.Series((np.arange(len(df)) % 2).astype(int), index=df.index)
        return x, y, ["f1", "f2"]

    def build_model(self, input_shape):
        raise NotImplementedError

    def generate_signal_strategy(self, stock, *args, **kwargs):
        raise NotImplementedError


class DummySequentialAgent(SequentialNNAgent):
    def feature_engineering(self, stock):
        feature_index = self.data[stock].index[2:]
        x = np.arange(len(feature_index) * 4, dtype=float).reshape(len(feature_index), 2, 2)
        y = np.arange(len(feature_index)) % 2
        return x, y, feature_index

    def build_model(self, input_shape):
        raise NotImplementedError

    def generate_signal_strategy(self, stock, *args, **kwargs):
        raise NotImplementedError


class BaseAgentsTests(unittest.TestCase):
    def test_ml_agent_smoke_produces_returns(self):
        data = make_market_data(periods=80)
        agent = LRAgent(data)

        agent.run_all(mode="backtest")

        self.assertIn("AAA", agent.signal_data)
        self.assertIn("AAA", agent.returns_data)
        self.assertFalse(agent.signal_data["AAA"].empty)
        self.assertIn("SignalStrength", agent.signal_data["AAA"].columns)

    def test_nn_predict_signals_uses_training_stats_and_emits_score(self):
        data = make_market_data(periods=8)
        stock = "AAA"
        agent = DummyNNAgent(data)
        x, _, feature_cols = agent.feature_engineering(stock)
        x_train = x.iloc[:5]
        x_test = x.iloc[5:]

        agent.models[stock] = DummyPredictModel([0.2, 0.8, 0.6])
        agent.train_data[stock] = {
            "X_train": x_train,
            "X_test": x_test,
            "y_train": pd.Series([0, 1, 0, 1, 0], index=x_train.index),
            "y_test": pd.Series([1, 0, 1], index=x_test.index),
            "feature_cols": feature_cols,
            "mu": x_train.mean(),
            "sigma": x_train.std().replace(0, 1.0).fillna(1.0),
        }

        signals = agent.predict_signals(stock, mode="backtest", threshold=0.5)

        self.assertListEqual(signals.index.tolist(), x_test.index.tolist())
        self.assertListEqual(signals["Prediction"].tolist(), [0, 1, 1])
        self.assertIn("SignalStrength", signals.columns)
        self.assertAlmostEqual(float(signals["SignalStrength"].iloc[1]), 0.8)

    def test_sequential_agent_preserves_actual_test_index(self):
        data = make_market_data(periods=10)
        stock = "AAA"
        agent = DummySequentialAgent(data)
        _, _, feature_index = agent.feature_engineering(stock)
        test_index = pd.Index(feature_index[-3:])

        agent.models[stock] = DummyPredictModel([0.1, 0.7, 0.9])
        agent.train_data[stock] = {
            "X_train": np.zeros((5, 2, 2), dtype=float),
            "X_test": np.ones((3, 2, 2), dtype=float),
            "y_train": np.array([0, 1, 0, 1, 0]),
            "y_test": np.array([1, 0, 1]),
            "index_train": pd.Index(feature_index[:5]),
            "index_test": test_index,
            "mu": np.zeros((2, 2), dtype=float),
            "sigma": np.ones((2, 2), dtype=float),
        }

        signals = agent.predict_signals(stock, mode="backtest", threshold=0.5)

        self.assertListEqual(signals.index.tolist(), test_index.tolist())
        self.assertListEqual(signals["Prediction"].tolist(), [0, 1, 1])
        self.assertIn("SignalStrength", signals.columns)


if __name__ == "__main__":
    unittest.main()
