import unittest

import numpy as np
import pandas as pd

from agents.base_agents.ml_trading_agent import MLBasedAgent
from agents.base_agents.nn_based_agent import NNBasedAgent
from agents.base_agents.sequential_based import SequentialNNAgent
from agents.ml_based.anomaly.autoencoder_agent import AutoencoderAgent
from agents.ml_based.deep_learning.cnn_agent import CNNAgent
from agents.ml_based.regime.hmm_based_agent import HMMRegimeAgent
from agents.ml_based.classical.logistic_reg_agent import LRAgent
from agents.ml_based.deep_learning.lstm_agent import LSTMAgent
from agents.ml_based.classical.naive_bayes_agent import NaiveBayesAgent
from agents.ml_based.deep_learning.transformer_agent import TransformerAgent
from agents.ml_based.classical.svm_agent import SVMAgent
from agents.technical.bollinger_bands_agent import BollingerBandsAgent
from agents.technical.high_low import HighLowAgent
from agents.technical.macd_agent import MACDAgent
from agents.technical.moving_average_agent import MovingAverageAgent
from agents.technical.moving_average_crossover_agent import MovingAverageCrossoverAgent
from agents.technical.mean_reversion_agent import MeanReversionAgent
from agents.technical.momentum_agent import MomentumAgent
from agents.technical.nr7_agent import NR7BreakoutAgent
from agents.technical.on_balance_volume_agent import OBVAgent
from agents.technical.performance_agent import PerformanceBasedAgent
from agents.technical.price_volume_trend_agent import PVTAgent
from agents.technical.rsi_agent import RSIAgent
from agents.technical.supertrend_agent import SupertrendAgent
from agents.technical.volume_price_divergence_agent import VolumePriceDivergenceAgent
from agents.technical.volume_weighted_average_price_agent import VWAPAgent


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


def make_geometric_market_data(stock="AAA", periods=12, daily_log_return=0.01):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    close = 100 * np.exp(np.arange(periods) * daily_log_return)
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


class DummyProbClassifier:
    def __init__(self, prob_up):
        self._prob_up = np.asarray(prob_up, dtype=float)
        self.classes_ = np.array([-1, 1])

    def predict_proba(self, X):
        prob_up = self._prob_up[: len(X)]
        return np.column_stack([1.0 - prob_up, prob_up])


class DummyMLAgent(MLBasedAgent):
    def __init__(self, data):
        super().__init__(data, model=None, features=["f1", "f2"])

    def feature_engineering(self, stock):
        df = self.data[stock].copy()
        x = pd.DataFrame(
            {
                "f1": np.linspace(0.0, 1.0, len(df)),
                "f2": np.linspace(1.0, 2.0, len(df)),
            },
            index=df.index,
        )
        y = pd.Series([1, -1, 1, -1, 1, -1][: len(df)], index=df.index)
        return x, y

    def generate_signal_strategy(self, stock, *args, **kwargs):
        raise NotImplementedError


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


class ProbeHMMRegimeAgent(HMMRegimeAgent):
    def __init__(self, data):
        self.train_calls = []
        self.predict_calls = []
        super().__init__(data, auto_generate=False)

    def train_hmm(self, stock, split_ratio=None):
        self.train_calls.append((stock, split_ratio))
        self.hmm_models[stock] = object()
        self.scalers[stock] = object()
        self.best_regimes[stock] = 1
        self.regime_return_maps[stock] = {0: -0.01, 1: 0.02}
        self.train_data[stock] = {
            "df_train": pd.DataFrame(),
            "df_test": pd.DataFrame(),
            "feature_cols": ["Return_1D", "Volatility", "Volume_Change"],
            "index_train": pd.Index([]),
            "index_test": pd.Index([]),
        }

    def predict_signals(self, stock, mode="backtest"):
        self.predict_calls.append((stock, mode))
        index = self.data[stock].index[-3:]
        return pd.DataFrame(
            {
                "Regime": [0, 1, 1],
                "Good_Regime": [0, 1, 1],
                "SignalStrength": [0.3, 0.8, 0.9],
                "Position": [0, 1, 1],
                "Signal": [0, 1, 0],
                "return": [0.01, 0.02, -0.01],
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

    def test_ml_predict_signals_preserves_short_positions_for_minus_one_labels(self):
        data = make_market_data(periods=6)
        stock = "AAA"
        agent = DummyMLAgent(data)
        x, y = agent.feature_engineering(stock)
        x_train = x.iloc[:3]
        x_test = x.iloc[3:]

        agent.models[stock] = DummyProbClassifier([0.2, 0.8, 0.3])
        agent.train_data[stock] = (x_train, x_test, y.iloc[:3], y.iloc[3:])

        signals = agent.predict_signals(stock, mode="backtest", threshold=0.5)

        self.assertListEqual(signals.index.tolist(), x_test.index.tolist())
        self.assertListEqual(signals["Prediction"].tolist(), [-1, 1, -1])
        self.assertListEqual(signals["Position"].tolist(), [-1, 1, -1])
        self.assertListEqual(signals["Signal"].tolist(), [0, 1, -1])

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

    def test_naive_bayes_agent_feature_contract_matches_base(self):
        data = make_market_data(periods=20)
        agent = NaiveBayesAgent(data)

        X, y = agent.feature_engineering("AAA")

        self.assertListEqual(X.columns.tolist(), ["OC", "HL"])
        self.assertEqual(len(X), len(y))

    def test_svm_agent_feature_contract_matches_base(self):
        data = make_market_data(periods=20)
        agent = SVMAgent(data)

        X, y = agent.feature_engineering("AAA")

        self.assertListEqual(X.columns.tolist(), ["OC", "HL"])
        self.assertEqual(len(X), len(y))

    def test_hmm_agent_uses_explicit_train_predict_flow(self):
        data = make_market_data(periods=12)
        agent = ProbeHMMRegimeAgent(data)

        self.assertEqual(agent.train_calls, [])
        self.assertEqual(agent.predict_calls, [])

        signals = agent.generate_signal_strategy("AAA", mode="backtest")

        self.assertEqual(agent.train_calls, [("AAA", None)])
        self.assertEqual(agent.predict_calls, [("AAA", "backtest")])
        self.assertTrue(signals.equals(agent.signal_data["AAA"]))

    def test_hmm_agent_run_all_populates_returns(self):
        data = make_market_data(periods=12)
        agent = ProbeHMMRegimeAgent(data)

        agent.run_all(mode="backtest")

        self.assertIn("AAA", agent.signal_data)
        self.assertIn("AAA", agent.returns_data)
        self.assertIn("SignalStrength", agent.signal_data["AAA"].columns)

    def test_momentum_agent_can_skip_constructor_generation(self):
        data = make_market_data(periods=12)
        agent = MomentumAgent(data, lookbacks=[2, 4], auto_generate=False)

        self.assertEqual(agent.signal_data, {})
        self.assertEqual(agent.returns_data, {})

    def test_momentum_agent_validates_lookbacks(self):
        data = make_market_data(periods=12)

        with self.assertRaises(ValueError):
            MomentumAgent(data, lookbacks=[0, 4], auto_generate=False)

        with self.assertRaises(ValueError):
            MomentumAgent(data, lookbacks=[], auto_generate=False)

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

    def test_momentum_agent_raw_mode_does_not_depend_on_z_score_availability(self):
        data = make_geometric_market_data(periods=10)
        agent = MomentumAgent(data, lookbacks=[2], score_mode="raw", auto_generate=False)

        signals = agent.generate_signal_strategy("AAA")

        self.assertTrue(signals["MomentumZ_2"].iloc[3:].isna().all())
        self.assertTrue(signals["SignalStrength"].iloc[2:].notna().all())
        self.assertTrue((signals["Position"].iloc[2:] == 1).all())

    def test_momentum_agent_z_score_suppresses_near_zero_volatility(self):
        data = make_geometric_market_data(periods=10)
        agent = MomentumAgent(data, lookbacks=[2], score_mode="z", auto_generate=False)

        signals = agent.generate_signal_strategy("AAA")

        self.assertTrue(signals["Momentum"].iloc[2:].notna().all())
        self.assertTrue(signals["SignalStrength"].iloc[2:].isna().all())
        self.assertTrue((signals["Position"].iloc[2:] == 0).all())

    def test_momentum_agent_exports_dailyized_momentum(self):
        data = make_geometric_market_data(periods=10, daily_log_return=0.02)
        agent = MomentumAgent(data, lookbacks=[2], score_mode="z", auto_generate=False)

        signals = agent.generate_signal_strategy("AAA")

        expected = signals["MomentumDaily_2"]
        pd.testing.assert_series_equal(signals["Momentum"], expected, check_names=False)

    def test_volume_price_divergence_agent_maps_buy_and_sell_to_correct_positions(self):
        index = pd.date_range("2024-01-01", periods=3, freq="B")
        columns = pd.MultiIndex.from_product([["AAA"], ["Close", "Volume"]])

        buy_data = pd.DataFrame(index=index, columns=columns, dtype=float)
        buy_data[("AAA", "Close")] = [100, 90, 80]
        buy_data[("AAA", "Volume")] = [1000, 800, 700]

        buy_agent = VolumePriceDivergenceAgent(buy_data, window=1, threshold=0.05)
        buy_signals = buy_agent.signal_data["AAA"]
        self.assertEqual(int(buy_signals["Position"].iloc[1]), 1)
        self.assertEqual(buy_agent.action_now("AAA")["Action"], "HOLD (LONG)")

        sell_data = pd.DataFrame(index=index, columns=columns, dtype=float)
        sell_data[("AAA", "Close")] = [100, 110, 120]
        sell_data[("AAA", "Volume")] = [1000, 800, 700]

        sell_agent = VolumePriceDivergenceAgent(sell_data, window=1, threshold=0.05)
        sell_signals = sell_agent.signal_data["AAA"]
        self.assertEqual(int(sell_signals["Position"].iloc[1]), -1)
        self.assertEqual(sell_agent.action_now("AAA")["Action"], "HOLD (SHORT)")

    def test_mean_reversion_agent_signal_tracks_forward_filled_position(self):
        index = pd.date_range("2024-01-01", periods=8, freq="B")
        columns = pd.MultiIndex.from_product([["AAA"], ["Close"]])
        data = pd.DataFrame(index=index, columns=columns, dtype=float)
        data[("AAA", "Close")] = [100, 100, 100, 90, 100, 100, 100, 100]

        agent = MeanReversionAgent(data, lookback_period=3, threshold=1.0, auto_generate=False)
        signals = agent.generate_signal_strategy("AAA")

        self.assertEqual(int(signals["Position"].iloc[3]), 1)
        self.assertEqual(int(signals["Signal"].iloc[3]), 1)
        self.assertEqual(int(signals["Position"].iloc[4]), 1)
        self.assertEqual(int(signals["Signal"].iloc[4]), 0)

    def test_moving_average_crossover_agent_populates_returns(self):
        data = make_market_data(periods=20)
        agent = MovingAverageCrossoverAgent(data, short_window=3, long_window=5, auto_generate=False)

        self.assertEqual(agent.signal_data, {})
        self.assertEqual(agent.returns_data, {})

        agent.run_all()

        self.assertIn("AAA", agent.signal_data)
        self.assertIn("AAA", agent.returns_data)
        self.assertIn("return", agent.signal_data["AAA"].columns)

    def test_rsi_agent_uses_flat_position_for_neutral_or_undefined_rsi(self):
        index = pd.date_range("2024-01-01", periods=20, freq="B")
        columns = pd.MultiIndex.from_product([["AAA"], ["Close"]])
        data = pd.DataFrame(index=index, columns=columns, dtype=float)
        data[("AAA", "Close")] = [
            100, 102, 104, 106, 108, 110, 108, 106, 104, 103,
            103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
        ]

        agent = RSIAgent(data, period=5, upper_band=70, lower_band=30, auto_generate=False)
        signals = agent.generate_signal_strategy("AAA")

        self.assertFalse(signals["Position"].iloc[5:].isna().any())
        self.assertEqual(int(signals["Position"].iloc[-1]), 0)
        self.assertEqual(int(signals["Signal"].iloc[-1]), 0)

    def test_vwap_agent_can_skip_constructor_generation(self):
        data = make_market_data(periods=20)
        agent = VWAPAgent(data, period=3, threshold=0.05, auto_generate=False)

        self.assertEqual(agent.signal_data, {})
        self.assertEqual(agent.returns_data, {})

    def test_vwap_agent_validates_parameters(self):
        data = make_market_data(periods=20)

        with self.assertRaises(ValueError):
            VWAPAgent(data, period=0, threshold=0.05, auto_generate=False)

        with self.assertRaises(ValueError):
            VWAPAgent(data, period=3, threshold=0.0, auto_generate=False)

    def test_vwap_agent_uses_symmetric_threshold_bands(self):
        index = pd.date_range("2024-01-01", periods=3, freq="B")
        columns = pd.MultiIndex.from_product([["AAA"], ["High", "Low", "Close", "Volume"]])
        data = pd.DataFrame(index=index, columns=columns, dtype=float)
        data[("AAA", "High")] = [100, 100, 100]
        data[("AAA", "Low")] = [100, 100, 100]
        data[("AAA", "Close")] = [100, 96, 80]
        data[("AAA", "Volume")] = [100, 100, 100]

        agent = VWAPAgent(data, period=2, threshold=0.05, auto_generate=False)
        signals = agent.generate_signal_strategy("AAA")

        self.assertEqual(int(signals["Position"].iloc[1]), 0)
        self.assertEqual(int(signals["Position"].iloc[2]), -1)

    def test_vwap_agent_uses_warmup_gating_and_signal_strength(self):
        data = make_market_data(periods=20)
        agent = VWAPAgent(data, period=3, threshold=0.05, auto_generate=False)

        signals = agent.generate_signal_strategy("AAA")

        self.assertTrue(signals["SignalStrength"].iloc[:2].isna().all())
        self.assertTrue((signals["Position"].iloc[:2] == 0).all())
        self.assertTrue(signals["Valid"].iloc[:2].eq(False).all())
        self.assertTrue(signals["Valid"].iloc[2:].eq(True).all())
        self.assertTrue(signals["SignalStrength"].iloc[2:].notna().all())

    def test_obv_agent_uses_standard_flat_close_obv(self):
        index = pd.date_range("2024-01-01", periods=4, freq="B")
        columns = pd.MultiIndex.from_product([["AAA"], ["Close", "Volume"]])
        data = pd.DataFrame(index=index, columns=columns, dtype=float)
        data[("AAA", "Close")] = [100, 101, 100, 100]
        data[("AAA", "Volume")] = [10, 20, 30, 40]

        agent = OBVAgent(data, auto_generate=False)
        signals = agent.generate_signal_strategy("AAA")

        expected = pd.Series([0.0, 20.0, -10.0, -10.0], index=index)
        pd.testing.assert_series_equal(signals["OBV"], expected, check_names=False)

    def test_pvt_agent_holds_position_across_neutral_bar(self):
        index = pd.date_range("2024-01-01", periods=6, freq="B")
        columns = pd.MultiIndex.from_product([["AAA"], ["Close", "Volume"]])
        data = pd.DataFrame(index=index, columns=columns, dtype=float)
        data[("AAA", "Close")] = [100, 80, 64, 51.2, 56.32, 50.688]
        data[("AAA", "Volume")] = [100, 100, 100, 100, 300, 100]

        agent = PVTAgent(data, threshold=0.05, auto_generate=False)
        signals = agent.generate_signal_strategy("AAA")

        self.assertEqual(int(signals["Position"].iloc[4]), 1)
        self.assertEqual(int(signals["Position"].iloc[5]), 1)
        self.assertEqual(int(signals["Signal"].iloc[5]), 0)

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

    def test_performance_agent_can_skip_constructor_generation(self):
        data = make_market_data(periods=20)
        agent = PerformanceBasedAgent(data, period_length=3, top_n=1, holding_period=4, auto_generate=False)

        self.assertEqual(agent.signal_data, {})
        self.assertEqual(agent.returns_data, {})

    def test_performance_agent_validates_top_n_against_stock_count(self):
        data = pd.concat(
            [
                make_market_data(stock="AAA", periods=10),
                make_market_data(stock="BBB", periods=10),
            ],
            axis=1,
        )

        with self.assertRaises(ValueError):
            PerformanceBasedAgent(data, period_length=3, top_n=3, holding_period=4, auto_generate=False)

    def test_performance_agent_run_all_populates_returns_and_portfolio_metrics(self):
        data = make_market_data(periods=20)
        agent = PerformanceBasedAgent(data, period_length=3, top_n=1, holding_period=4, auto_generate=False)

        agent.run_all()

        self.assertIn("AAA", agent.signal_data)
        self.assertIn("AAA", agent.returns_data)
        self.assertIn("LookbackReturn", agent.signal_data["AAA"].columns)
        self.assertEqual(len(agent.portfolio_log_returns), len(data.index))
        self.assertEqual(len(agent.selection_log), len(data.index))

    def test_performance_agent_rebalances_from_first_valid_date(self):
        data = make_market_data(periods=12)
        agent = PerformanceBasedAgent(data, period_length=3, top_n=1, holding_period=4, auto_generate=False)

        agent.run_all()

        n_held = agent.holdings_matrix.sum(axis=1)
        self.assertTrue((n_held.iloc[:3] == 0).all())
        self.assertEqual(int(n_held.iloc[3]), 1)

    def test_performance_agent_uses_exact_equal_weight_portfolio_log_return(self):
        index = pd.date_range("2024-01-01", periods=4, freq="B")
        columns = pd.MultiIndex.from_product([["AAA", "BBB"], ["Close"]])
        data = pd.DataFrame(index=index, columns=columns, dtype=float)
        data[("AAA", "Close")] = [100, 100, 200, 200]
        data[("BBB", "Close")] = [100, 100, 100, 100]

        agent = PerformanceBasedAgent(
            data,
            period_length=1,
            top_n=2,
            holding_period=2,
            auto_generate=False,
        )

        agent.run_all()

        self.assertAlmostEqual(agent.portfolio_log_returns.iloc[2], np.log1p(0.5), places=9)
        self.assertAlmostEqual(agent.cumulative_returns.iloc[-1], np.log1p(0.5), places=9)

    def test_nr7_agent_run_all_populates_return_contract(self):
        data = make_market_data(periods=12)
        agent = NR7BreakoutAgent(data, hold_days=3, auto_generate=False)

        self.assertEqual(agent.signal_data, {})
        self.assertEqual(agent.returns_data, {})

        agent.run_all()

        self.assertIn("AAA", agent.signal_data)
        self.assertIn("AAA", agent.returns_data)
        self.assertIn("return", agent.signal_data["AAA"].columns)

    def test_high_low_agent_flips_on_opposite_breakout(self):
        data = make_market_data(periods=55)
        close = np.full(55, 100.0)
        close[50] = 101.0
        close[51:54] = 101.0
        close[54] = 99.0
        data[("AAA", "Close")] = close
        data[("AAA", "Open")] = close
        data[("AAA", "High")] = close + 1.0
        data[("AAA", "Low")] = close - 1.0

        agent = HighLowAgent(data, auto_generate=False)
        signals = agent.generate_signal_strategy("AAA")

        self.assertEqual(int(signals["Position"].iloc[50]), 1)
        self.assertEqual(int(signals["Signal"].iloc[50]), 1)
        self.assertEqual(int(signals["Position"].iloc[54]), -1)
        self.assertEqual(int(signals["Signal"].iloc[54]), -1)

    def test_macd_agent_supports_skip_and_validates_windows(self):
        data = make_market_data(periods=20)
        agent = MACDAgent(data, short_window=3, long_window=5, signal_window=2, auto_generate=False)

        self.assertEqual(agent.signal_data, {})
        self.assertEqual(agent.returns_data, {})

        with self.assertRaises(ValueError):
            MACDAgent(data, short_window=5, long_window=5, signal_window=2, auto_generate=False)

    def test_bollinger_agent_supports_skip_and_validation(self):
        data = make_market_data(periods=20)
        agent = BollingerBandsAgent(data, period=5, num_std_dev=2, auto_generate=False)

        self.assertEqual(agent.signal_data, {})
        self.assertEqual(agent.returns_data, {})

        with self.assertRaises(ValueError):
            BollingerBandsAgent(data, period=0, num_std_dev=2, auto_generate=False)

        with self.assertRaises(ValueError):
            BollingerBandsAgent(data, period=5, num_std_dev=0, auto_generate=False)

    def test_supertrend_agent_can_skip_constructor_generation(self):
        data = make_market_data(periods=20)
        agent = SupertrendAgent(data, period=3, multiplier=2.0, auto_generate=False)

        self.assertEqual(agent.signal_data, {})
        self.assertEqual(agent.returns_data, {})

    def test_supertrend_agent_validates_parameters(self):
        data = make_market_data(periods=20)

        with self.assertRaises(ValueError):
            SupertrendAgent(data, period=0, multiplier=2.0, auto_generate=False)

        with self.assertRaises(ValueError):
            SupertrendAgent(data, period=3, multiplier=0, auto_generate=False)

    def test_supertrend_agent_uses_warmup_gating(self):
        data = make_market_data(periods=20)
        agent = SupertrendAgent(data, period=3, multiplier=2.0, auto_generate=False)

        signals = agent.generate_signal_strategy("AAA")

        self.assertTrue(signals["SignalStrength"].iloc[:2].isna().all())
        self.assertTrue((signals["Position"].iloc[:2] == 0).all())
        self.assertTrue(signals["Valid"].iloc[:2].eq(False).all())
        self.assertTrue(signals["Valid"].iloc[2:].eq(True).all())

    def test_supertrend_agent_run_all_populates_returns(self):
        data = make_market_data(periods=20)
        agent = SupertrendAgent(data, period=3, multiplier=2.0, auto_generate=False)

        agent.run_all()

        self.assertIn("AAA", agent.signal_data)
        self.assertIn("AAA", agent.returns_data)
        self.assertIn("SignalStrength", agent.signal_data["AAA"].columns)
        self.assertIn("Supertrend", agent.signal_data["AAA"].columns)


if __name__ == "__main__":
    unittest.main()
