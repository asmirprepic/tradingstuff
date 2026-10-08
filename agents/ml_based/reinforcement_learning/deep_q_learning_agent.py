import time
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agents.base_agents.trading_agent import TradingAgent

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential
except ModuleNotFoundError as exc:
    tf = Dense = Input = Sequential = None
    _TENSORFLOW_IMPORT_ERROR = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None


class DeepQLearningAgent(TradingAgent):
    """
    Deep Q-learning style trading agent with a cleaned explicit lifecycle.

    The agent learns a simple long-vs-flat policy per stock from engineered
    features and next-bar log returns. The implementation is intentionally
    lightweight so it fits the shared TradingAgent contract cleanly.
    """
    DEFAULT_FEATURES = (
        "Return_1D",
        "Return_5D",
        "Intraday_Return",
        "Range_Pct",
        "Volatility_5D",
        "MA_Distance_20",
        "Volume_Change",
    )

    def __init__(
        self,
        data,
        alpha=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        episodes=100,
        split_ratio=0.8,
        hidden_units=24,
        verbose=0,
        random_state=42,
        progress_interval=10,
        features=None,
    ):
        super().__init__(data)
        self._validate_parameters(
            alpha, gamma, epsilon, epsilon_decay, epsilon_min,
            episodes, split_ratio, hidden_units, verbose, random_state, progress_interval,
        )
        self.algorithm_name = "DeepQLearning"
        self.score_column = "SignalStrength"
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.split_ratio = split_ratio
        self.hidden_units = hidden_units
        self.verbose = verbose
        self.random_state = int(random_state)
        self.progress_interval = int(progress_interval)

        if isinstance(features, str):
            raise ValueError("features must be a sequence of feature names, not a string.")
        self.features = list(features if features is not None else self.DEFAULT_FEATURES)
        if not self.features:
            raise ValueError("At least one feature must be selected.")
        if len(self.features) != len(set(self.features)):
            raise ValueError("features must not contain duplicates.")
        unknown = sorted(set(self.features) - set(self.DEFAULT_FEATURES))
        if unknown:
            raise ValueError(f"Unsupported features: {unknown}")

        self.models = {}
        self.train_data = {}

    @staticmethod
    def _validate_parameters(alpha, gamma, epsilon, epsilon_decay, epsilon_min,
                             episodes, split_ratio, hidden_units, verbose, random_state,
                             progress_interval):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not 0 <= gamma < 1:
            raise ValueError("gamma must be between 0 (inclusive) and 1 (exclusive).")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1.")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be between 0 (exclusive) and 1.")
        if not 0 <= epsilon_min <= epsilon:
            raise ValueError("epsilon_min must be between 0 and epsilon.")
        integer_values = {
            "episodes": episodes,
            "hidden_units": hidden_units,
            "progress_interval": progress_interval,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < split_ratio < 1:
            raise ValueError("split_ratio must be between 0 and 1.")
        if isinstance(verbose, bool) or not isinstance(verbose, (int, np.integer)) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer.")
        if (isinstance(random_state, bool)
                or not isinstance(random_state, (int, np.integer))
                or random_state < 0):
            raise ValueError("random_state must be a non-negative integer.")

    def _training_rng(self):
        tf.keras.utils.set_random_seed(self.random_state)
        return np.random.default_rng(self.random_state)

    def _require_tensorflow(self):
        global tf, Dense, Input, Sequential, _TENSORFLOW_IMPORT_ERROR
        if Sequential is not None:
            return
        try:
            tf = import_module("tensorflow")
            layers = import_module("tensorflow.keras.layers")
            models = import_module("tensorflow.keras.models")
            Dense = layers.Dense
            Input = layers.Input
            Sequential = models.Sequential
            _TENSORFLOW_IMPORT_ERROR = None
        except ModuleNotFoundError as exc:
            _TENSORFLOW_IMPORT_ERROR = exc
            raise ModuleNotFoundError(
                "tensorflow is required to train DeepQLearningAgent. "
                "Install it in the active Python environment and restart or reload the notebook."
            ) from exc

    def build_feature_frame(self, stock):
        df = self.data[stock].copy()
        required = {"Close"}
        if "Intraday_Return" in self.features:
            required.add("Open")
        if "Range_Pct" in self.features:
            required.update(("High", "Low"))
        if "Volume_Change" in self.features:
            required.add("Volume")
        missing = [
            column for column in sorted(required)
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{stock} is missing columns: {missing}"
            )

        close = df["Close"]

        if "Return_1D" in self.features or "Volatility_5D" in self.features:
            df["Return_1D"] = close.pct_change(fill_method=None)
        if "Return_5D" in self.features:
            df["Return_5D"] = close.pct_change(periods=5, fill_method=None)
        if "Intraday_Return" in self.features:
            df["Intraday_Return"] = close.div(df["Open"]).sub(1.0)
        if "Range_Pct" in self.features:
            df["Range_Pct"] = df["High"].sub(df["Low"]).div(close)
        if "Volatility_5D" in self.features:
            df["Volatility_5D"] = df["Return_1D"].rolling(5).std()
        if "MA_Distance_20" in self.features:
            moving_average = close.rolling(20).mean()
            df["MA_Distance_20"] = close.div(moving_average).sub(1.0)
        if "Volume_Change" in self.features:
            df["Volume_Change"] = df["Volume"].pct_change(fill_method=None)

        df[self.features] = df[self.features].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        return df.dropna(subset=[*self.features, "Close"])

    def feature_engineering(self, stock):
        df = self.build_feature_frame(stock)
        df["ForwardReturn"] = np.log(df["Close"].shift(-1) / df["Close"])
        df = df.iloc[:-1].dropna(subset=["ForwardReturn"])
        X = df[self.features].copy()
        rewards = df["ForwardReturn"].copy()
        return X, rewards

    def create_train_split_group(self, X, rewards, index, split_ratio=None, **kwargs):
        kwargs.setdefault("shuffle", False)
        if split_ratio is not None and "test_size" not in kwargs:
            kwargs["test_size"] = 1 - split_ratio
        X_train, X_test, rewards_train, rewards_test, index_train, index_test = train_test_split(
            X, rewards, index, **kwargs
        )
        if len(X_train) < 2:
            raise ValueError("Training split is too small to purge the boundary label.")

        # The final training reward uses the following close, which belongs to
        # the test period. Remove that transition without changing the test set.
        return (
            X_train.iloc[:-1],
            X_test,
            rewards_train.iloc[:-1],
            rewards_test,
            index_train[:-1],
            index_test,
        )

    def build_model(self, input_dim):
        model = Sequential(
            [
                Input(shape=(input_dim,)),
                Dense(self.hidden_units, activation="relu"),
                Dense(self.hidden_units, activation="relu"),
                Dense(2, activation="linear"),
            ]
        )
        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
        )
        return model

    def _predict_q_values(self, model, X):
        return model.predict(X, verbose=0)

    def train_model(self, stock):
        self._require_tensorflow()
        rng = self._training_rng()
        X, rewards = self.feature_engineering(stock)

        if len(X) < 3:
            raise ValueError(f"Not enough rows to train DeepQLearningAgent for {stock}.")

        X_train, X_test, rewards_train, rewards_test, index_train, index_test = self.create_train_split_group(
            X,
            rewards,
            X.index,
            split_ratio=self.split_ratio,
        )

        if len(X_train) < 2:
            raise ValueError(f"Training split for {stock} is too small for Q-learning.")

        mu = X_train.mean()
        sigma = X_train.std().replace(0, 1.0).fillna(1.0)
        X_train_norm = ((X_train - mu) / sigma).to_numpy(dtype=float)
        X_test_norm = ((X_test - mu) / sigma).to_numpy(dtype=float)

        model = self.build_model(input_dim=X_train_norm.shape[1])
        epsilon = float(self.epsilon)
        reward_values = rewards_train.to_numpy(dtype=float)
        started = time.perf_counter()
        total_steps = 0
        if self.verbose:
            print(
                f"[DQN] {stock}: {len(X_train_norm)} training rows, "
                f"{self.episodes} episodes",
                flush=True,
            )

        for episode in range(1, self.episodes + 1):
            q_values = self._predict_q_values(model, X_train_norm)
            episode_steps = len(X_train_norm)

            greedy_actions = np.argmax(q_values, axis = 1)
            explore = rng.random(episode_steps) <= epsilon
            random_actions = rng.integers(2, size = episode_steps)
            actions = np.where(
                explore,
                random_actions,
                greedy_actions,
            ).astype(int)


            clipped_returns = np.clip(
                reward_values,
                -0.05,
                0.05
            )

            downside = np.maximum(
                -clipped_returns,
                0.0,
            )

            utility = clipped_returns - 0.5 * downside

            realized_rewards = np.where(
                actions == 1,
                utility,
                0.0
            ) * 100

            targets = realized_rewards.copy()
            targets[:-1] += (
                self.gamma * np.max(q_values[1:], axis = 1)
            )

            training_targets = q_values.copy()
            training_targets[
                np.arange(episode_steps),
                actions
            ] = targets

            model.train_on_batch(
                X_train_norm,
                training_targets,
            )

            episode_reward = float(realized_rewards.sum())
            total_steps += episode_steps

            if epsilon > self.epsilon_min:
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            should_report = (
                self.verbose
                and (episode == 1 or episode == self.episodes or episode % self.progress_interval == 0)
            )
            if should_report:
                elapsed = time.perf_counter() - started
                average_reward = episode_reward / episode_steps if episode_steps else 0.0
                rate = total_steps / elapsed if elapsed > 0 else 0.0
                print(
                    f"[DQN] {stock}: episode {episode}/{self.episodes} | "
                    f"epsilon={epsilon:.4f} | avg_reward={average_reward:.6f} | "
                    f"elapsed={elapsed:.1f}s | {rate:.1f} steps/s",
                    flush=True,
                )

        training_seconds = time.perf_counter() - started

        self.models[stock] = model
        self.train_data[stock] = {
            "X_train": X_train,
            "X_test": X_test,
            "rewards_train": rewards_train,
            "rewards_test": rewards_test,
            "index_train": pd.Index(index_train),
            "index_test": pd.Index(index_test),
            "mu": mu,
            "sigma": sigma,
            "epsilon_final": epsilon,
            "X_train_norm": X_train_norm,
            "X_test_norm": X_test_norm,
        }
        self.training_info[stock] = {
            "SplitRatio": self.split_ratio,
            "Episodes": self.episodes,
            "EpsilonFinal": epsilon,
            "TrainingSeconds": training_seconds,
            "TrainingUpdates": self.episodes,
        }

    def predict_signals(self, stock, mode="backtest", threshold=0.0):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} has not been trained.")
        if stock not in self.train_data:
            raise ValueError(f"Training data for {stock} is missing.")

        model = self.models[stock]
        train_data = self.train_data[stock]
        mu = train_data["mu"]
        sigma = train_data["sigma"]

        if mode == "backtest":
            X_pred = train_data["X_test"]
            index_used = train_data["index_test"]
        elif mode == "live":
            X_full = self.build_feature_frame(stock)[
                self.features
            ]
            X_pred = X_full.iloc[[-1]]
            index_used = pd.Index([X_full.index[-1]])
        else:
            raise ValueError("mode must be 'backtest' or 'live'")

        if len(X_pred) == 0:
            return pd.DataFrame(
                columns=[
                    "Prediction",
                    "FlatQ",
                    "LongQ",
                    "SignalStrength",
                    "Position",
                    "Signal",
                    "return",
                ]
            )

        X_pred_norm = ((X_pred - mu) / sigma).to_numpy(dtype=float)
        q_values = self._predict_q_values(model, X_pred_norm)
        flat_q = q_values[:, 0]
        long_q = q_values[:, 1]
        q_edge = long_q - flat_q
        predictions = np.where(q_edge > threshold, 1, 0)

        signals = pd.DataFrame(index=index_used)
        signals["Prediction"] = predictions
        signals["FlatQ"] = flat_q
        signals["LongQ"] = long_q
        signals["SignalStrength"] = q_edge
        signals["Position"] = predictions.astype(int)
        signals["Signal"] = 0
        signals.loc[signals["Position"] > signals["Position"].shift(1), "Signal"] = 1
        signals.loc[signals["Position"] < signals["Position"].shift(1), "Signal"] = -1

        close = self.data[(stock, "Close")]
        signals["return"] = np.log(close / close.shift(1)).reindex(index_used)
        return signals

    def generate_signal_strategy(self, stock, mode="backtest", threshold=0.0):
        if stock not in self.models:
            self.train_model(stock)

        signals = self.predict_signals(stock, mode=mode, threshold=threshold)
        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest", threshold=0.0):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.signal_data[stock] = self.generate_signal_strategy(
                stock,
                mode=mode,
                threshold=threshold,
            )
        self.calculate_returns()
