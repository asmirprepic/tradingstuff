import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agents.base_agents.trading_agent import TradingAgent

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
except ModuleNotFoundError as exc:
    tf = Dense = Sequential = None
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
        if Sequential is None:
            raise ModuleNotFoundError(
                "tensorflow is required to train DeepQLearningAgent. "
                "Install tensorflow before using this agent."
            ) from _TENSORFLOW_IMPORT_ERROR

    def build_feature_frame(self, stock):
        df = self.data[stock].copy()
        df["Open-Close"] = df["Open"] - df["Close"]
        df["High-Low"] = df["High"] - df["Low"]
        return df.dropna(subset=["Open-Close", "High-Low", "Close"])

    def feature_engineering(self, stock):
        df = self.build_feature_frame(stock)
        df["ForwardReturn"] = np.log(df["Close"].shift(-1) / df["Close"])
        df = df.iloc[:-1].dropna(subset=["ForwardReturn"])
        X = df[["Open-Close", "High-Low"]].copy()
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
                Dense(self.hidden_units, input_dim=input_dim, activation="relu"),
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
            state = 0
            episode_reward = 0.0
            episode_steps = 0
            while state < len(X_train_norm) - 1:
                if rng.random() <= epsilon:
                    action = int(rng.integers(2))
                else:
                    action = int(np.argmax(self._predict_q_values(model, X_train_norm[state:state + 1])[0]))

                next_state = state + 1
                reward = reward_values[state] if action == 1 else 0.0
                adjacent_q = self._predict_q_values(model, X_train_norm[state:next_state + 1])
                next_q = adjacent_q[1]
                target = reward + self.gamma * float(np.max(next_q))
                target_f = adjacent_q[0:1].copy()
                target_f[0, action] = target
                model.train_on_batch(X_train_norm[state:state + 1], target_f)
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                state = next_state

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
            X_full = self.build_feature_frame(stock)[["Open-Close", "High-Low"]]
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
