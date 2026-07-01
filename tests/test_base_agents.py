import unittest

import numpy as np
import pandas as pd

from agents.base_agents.nn_based_agent import NNBasedAgent
from agents.base_agents.sequential_based import SequentialNNAgent
from agents.ml_based.autoencoder_agent import AutoencoderAgent
from agents.ml_based.cnn_agent import CNNAgent
from agents.ml_based.logistic_reg_agent import LRAgent
from agents.ml_based.lstm_agent import LSTMAgent
from agents.ml_based.transformer_agent import TransformerAgent
from agents.technical.moving_average_agent import MovingAverageAgent
from agents.technical.momentum_agent import MomentumAgent


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


class ProbeLSTMAgent(LSTMAgent):
    def __init__(self, data):
        self.train_calls = []
        self.predict_calls = []
        super().__init__(data, sequence_length=3)

    def train_model(self, stock):
        self.train_calls.append(stock)

    def predict_signals(self, stock, mode="backtest", threshold=0.5):
        self.predict_calls.append((stock, mode, threshold))
        index = self.data[stock].index[-3:]
        return pd.DataFrame(
            {
                "Prediction": [0, 1, 1],
                "ProbUp": [0.2, 0.8, 0.9],
                "SignalStrength": [0.2, 0.8, 0.9],
                "Position": [0, 1, 1],
                "Signal": [0, 1, 0],
                "return": [0.01, 0.02, -0.01],
            },
            index=index,
        )


class ProbeTransformerAgent(TransformerAgent):
    def __init__(self, data):
        self.train_calls = []
        self.predict_calls = []
        super().__init__(data, sequence_length=3)

    def train_model(self, stock):
        self.train_calls.append(stock)

    def predict_signals(self, stock, mode="backtest", threshold=0.5):
        self.predict_calls.append((stock, mode, threshold))
        index = self.data[stock].index[-2:]
        return pd.DataFrame(
            {
                "Prediction": [1, 0],
                "ProbUp": [0.7, 0.3],
                "SignalStrength": [0.7, 0.3],
                "Position": [1, 0],
                "Signal": [1, -1],
                "return": [0.03, -0.02],
            },
            index=index,
        )


class ProbeAutoencoderAgent(AutoencoderAgent):
    def __init__(self, data):
        self.train_calls = []
        self.predict_calls = []
        super().__init__(data, anomaly_threshold_percentile=90)

    def train_model(self, stock, anomaly_threshold_percentile=None):
        percentile = self.anomaly_threshold_percentile if anomaly_threshold_percentile is None else anomaly_threshold_percentile
        self.train_calls.append((stock, percentile))
        self.models[stock] = object()
        self.scalers[stock] = object()
        self.thresholds[stock] = 1.0
        self.train_data[stock] = {
            "X_train": pd.DataFrame(),
            "X_test": pd.DataFrame(),
            "index_train": pd.Index([]),
            "index_test": pd.Index([]),
            "percentile": percentile,
        }
        return 1.0

    def predict_signals(self, stock, mode="backtest"):
        self.predict_calls.append((stock, mode))
        index = self.data[stock].index[-3:]
        return pd.DataFrame(
            {
                "Anomaly": [False, True, True],
                "ReconstructionError": [0.2, 1.2, 1.4],
                "SignalStrength": [0.2, 1.2, 1.4],
                "Position": [0, 1, 1],
                "Signal": [0, 1, 0],
                "return": [0.01, 0.03, -0.01],
            },
            index=index,
        )


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

    def test_sequence_agents_share_base_contract(self):
        self.assertTrue(issubclass(CNNAgent, SequentialNNAgent))
        self.assertTrue(issubclass(LSTMAgent, SequentialNNAgent))
        self.assertTrue(issubclass(TransformerAgent, SequentialNNAgent))

    def test_cnn_feature_engineering_returns_aligned_sequences(self):
        data = make_market_data(periods=20)
        agent = CNNAgent(data, sequence_length=4)

        X, y, index = agent.feature_engineering("AAA")

        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[1], 4)
        self.assertEqual(X.shape[2], 2)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(index))

    def test_lstm_generate_signal_strategy_uses_shared_pipeline_without_ctor_training(self):
        data = make_market_data(periods=12)
        agent = ProbeLSTMAgent(data)

        self.assertEqual(agent.train_calls, [])
        self.assertEqual(agent.predict_calls, [])

        signals = agent.generate_signal_strategy("AAA", mode="backtest", threshold=0.7)

        self.assertEqual(agent.train_calls, ["AAA"])
        self.assertEqual(agent.predict_calls, [("AAA", "backtest", 0.7)])
        self.assertTrue(signals.equals(agent.signal_data["AAA"]))

    def test_transformer_generate_signal_strategy_uses_shared_pipeline_without_ctor_training(self):
        data = make_market_data(periods=12)
        agent = ProbeTransformerAgent(data)

        self.assertEqual(agent.train_calls, [])
        self.assertEqual(agent.predict_calls, [])

        signals = agent.generate_signal_strategy("AAA", mode="backtest", threshold=0.6)

        self.assertEqual(agent.train_calls, ["AAA"])
        self.assertEqual(agent.predict_calls, [("AAA", "backtest", 0.6)])
        self.assertTrue(signals.equals(agent.signal_data["AAA"]))

    def test_autoencoder_feature_engineering_returns_dataframe(self):
        data = make_market_data(periods=12)
        agent = AutoencoderAgent(data)

        features = agent.feature_engineering("AAA")

        self.assertIsInstance(features, pd.DataFrame)
        self.assertListEqual(features.columns.tolist(), ["Open-Close", "High-Low"])
        self.assertEqual(len(features), 12)

    def test_autoencoder_generate_signal_strategy_uses_explicit_train_predict_flow(self):
        data = make_market_data(periods=12)
        agent = ProbeAutoencoderAgent(data)

        self.assertEqual(agent.train_calls, [])
        self.assertEqual(agent.predict_calls, [])

        signals = agent.generate_signal_strategy("AAA", mode="backtest", anomaly_threshold_percentile=92)

        self.assertEqual(agent.train_calls, [("AAA", 92)])
        self.assertEqual(agent.predict_calls, [("AAA", "backtest")])
        self.assertTrue(signals.equals(agent.signal_data["AAA"]))

    def test_autoencoder_run_all_populates_returns(self):
        data = make_market_data(periods=12)
        agent = ProbeAutoencoderAgent(data)

        agent.run_all(mode="backtest", anomaly_threshold_percentile=90)

        self.assertIn("AAA", agent.signal_data)
        self.assertIn("AAA", agent.returns_data)
        self.assertIn("SignalStrength", agent.signal_data["AAA"].columns)

    def test_momentum_agent_can_skip_constructor_generation(self):
        data = make_market_data(periods=12)
        agent = MomentumAgent(data, lookbacks=[2, 4], auto_generate=False)

        self.assertEqual(agent.signal_data, {})
        self.assertEqual(agent.returns_data, {})

    def test_momentum_agent_requires_all_lookbacks_before_emitting_score(self):
        data = make_market_data(periods=12)
        agent = MomentumAgent(data, lookbacks=[2, 4], score_mode="z", auto_generate=False)

        signals = agent.generate_signal_strategy("AAA")

        self.assertTrue(signals["SignalStrength"].iloc[:3].isna().all())
        self.assertTrue(signals["Momentum"].iloc[:3].isna().all())
        self.assertTrue((signals["Position"].iloc[:3] == 0).all())
        self.assertTrue(signals["SignalStrength"].iloc[4:].notna().all())

    def test_momentum_agent_run_all_populates_returns(self):
        data = make_market_data(periods=20)
        agent = MomentumAgent(data, lookbacks=[3, 5], score_mode="raw", auto_generate=False)

        agent.run_all()

        self.assertIn("AAA", agent.signal_data)
        self.assertIn("AAA", agent.returns_data)
        self.assertIn("SignalStrength", agent.signal_data["AAA"].columns)
        self.assertIn("MomentumDaily_3", agent.signal_data["AAA"].columns)

    def test_moving_average_agent_validates_windows(self):
        data = make_market_data(periods=20)

        with self.assertRaises(ValueError):
            MovingAverageAgent(data, short_window=0, long_window=5, auto_generate=False)

        with self.assertRaises(ValueError):
            MovingAverageAgent(data, short_window=5, long_window=5, auto_generate=False)

    def test_moving_average_agent_can_skip_constructor_generation(self):
        data = make_market_data(periods=20)
        agent = MovingAverageAgent(data, short_window=3, long_window=5, auto_generate=False)

        self.assertEqual(agent.signal_data, {})
        self.assertEqual(agent.returns_data, {})

    def test_moving_average_agent_uses_explicit_warmup_gating(self):
        data = make_market_data(periods=20)
        agent = MovingAverageAgent(data, short_window=3, long_window=5, auto_generate=False)

        signals = agent.generate_signal_strategy("AAA")

        self.assertTrue(signals["SignalStrength"].iloc[:4].isna().all())
        self.assertTrue((signals["Position"].iloc[:4] == 0).all())
        self.assertTrue(signals["Valid"].iloc[:4].eq(False).all())
        self.assertTrue(signals["Valid"].iloc[4:].eq(True).all())

    def test_moving_average_agent_run_all_populates_returns(self):
        data = make_market_data(periods=20)
        agent = MovingAverageAgent(data, short_window=3, long_window=5, auto_generate=False)

        agent.run_all()

        self.assertIn("AAA", agent.signal_data)
        self.assertIn("AAA", agent.returns_data)
        self.assertIn("SignalStrength", agent.signal_data["AAA"].columns)


if __name__ == "__main__":
    unittest.main()
