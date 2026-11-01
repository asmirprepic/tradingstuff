import time
from collections import deque
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
        alpha=0.0003,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        episodes=100,
        split_ratio=0.8,
        hidden_units=16,
        verbose=0,
        random_state=42,
        progress_interval=10,
        features=None,
        transaction_cost=0.0,
        min_holding_period=0,
        cooldown_period=0,
        prediction_horizon=1,
        target_update_interval=25,
        replay_capacity=10000,
        replay_batch_size=64,
        replay_updates_per_episode=None,
    ):
        super().__init__(data)
        self._validate_parameters(
            alpha, gamma, epsilon, epsilon_decay, epsilon_min,
            episodes, split_ratio, hidden_units, verbose, random_state, progress_interval,
            transaction_cost, min_holding_period, cooldown_period, prediction_horizon,
            target_update_interval, replay_capacity, replay_batch_size,
            replay_updates_per_episode,
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
        self.transaction_cost = float(transaction_cost)
        self.min_holding_period = int(min_holding_period)
        self.cooldown_period = int(cooldown_period)
        self.prediction_horizon = int(prediction_horizon)
        self.target_update_interval = int(target_update_interval)
        self.replay_capacity = int(replay_capacity)
        self.replay_batch_size = int(replay_batch_size)
        self.replay_updates_per_episode = (
            None if replay_updates_per_episode is None else int(replay_updates_per_episode)
        )

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
        self.target_models = {}
        self.train_data = {}

    @staticmethod
    def _validate_parameters(alpha, gamma, epsilon, epsilon_decay, epsilon_min,
                             episodes, split_ratio, hidden_units, verbose, random_state,
                             progress_interval, transaction_cost, min_holding_period,
                             cooldown_period, prediction_horizon, target_update_interval,
                             replay_capacity, replay_batch_size, replay_updates_per_episode):
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
        if (isinstance(transaction_cost, bool)
                or not isinstance(transaction_cost, (int, float, np.number))
                or not np.isfinite(transaction_cost)
                or transaction_cost < 0):
            raise ValueError("transaction_cost must be a non-negative finite number.")
        for name, value in {
            "min_holding_period": min_holding_period,
            "cooldown_period": cooldown_period,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if (isinstance(prediction_horizon, bool)
                or not isinstance(prediction_horizon, (int, np.integer))
                or prediction_horizon < 1):
            raise ValueError("prediction_horizon must be a positive integer.")
        if (isinstance(target_update_interval, bool)
                or not isinstance(target_update_interval, (int, np.integer))
                or target_update_interval < 1):
            raise ValueError("target_update_interval must be a positive integer.")
        for name, value in {
            "replay_capacity": replay_capacity,
            "replay_batch_size": replay_batch_size,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (replay_updates_per_episode is not None
                and (isinstance(replay_updates_per_episode, bool)
                     or not isinstance(replay_updates_per_episode, (int, np.integer))
                     or replay_updates_per_episode < 1)):
            raise ValueError("replay_updates_per_episode must be None or a positive integer.")

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
        df["ForwardReturn"] = np.log(
            df["Close"].shift(-self.prediction_horizon) / df["Close"]
        )
        df = df.iloc[:-self.prediction_horizon].dropna(subset=["ForwardReturn"])
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
        if len(X_train) <= self.prediction_horizon:
            raise ValueError("Training split is too small to purge the boundary label.")

        # Remove labels whose forward-return windows cross into the test period.
        purge = self.prediction_horizon
        return (
            X_train.iloc[:-purge],
            X_test,
            rewards_train.iloc[:-purge],
            rewards_test,
            index_train[:-purge],
            index_test,
        )

    def build_model(self, input_dim):
        model = Sequential(
            [
                Input(shape=(input_dim,)),
                Dense(self.hidden_units, activation="swish"),
                Dense(2, activation="linear"),
            ]
        )
        model.compile(
            loss=tf.keras.losses.Huber(delta=0.02),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.alpha,
                clipnorm=1.0,
            ),
        )
        return model

    def _predict_q_values(self, model, X):
        return model.predict(X, verbose=0)

    def _model_predictor(self, model):
        """Create a low-overhead predictor for the fixed Sequential architecture."""
        def swish(values):
            sigmoid = 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))
            return values * sigmoid

        if hasattr(model, "get_weights"):
            weights = model.get_weights()
            if len(weights) == 4:
                dense_weights = tuple(np.asarray(value) for value in weights)

                def predict(states):
                    values = np.atleast_2d(np.asarray(states, dtype=float))
                    hidden = swish(values @ dense_weights[0] + dense_weights[1])
                    return hidden @ dense_weights[2] + dense_weights[3]

                return predict

            if len(weights) == 6:
                dense_weights = tuple(np.asarray(value) for value in weights)
                activation_names = {
                    getattr(getattr(layer, "activation", None), "__name__", "")
                    for layer in getattr(model, "layers", [])[:2]
                }
                uses_swish = (
                    bool(activation_names & {"swish", "silu"})
                    or dense_weights[2].shape[1] != dense_weights[0].shape[1]
                )

                def predict(states):
                    values = np.atleast_2d(np.asarray(states, dtype=float))
                    values = values @ dense_weights[0] + dense_weights[1]
                    values = swish(values) if uses_swish else np.maximum(values, 0.0)
                    values = values @ dense_weights[2] + dense_weights[3]
                    values = swish(values) if uses_swish else np.maximum(values, 0.0)
                    return values @ dense_weights[4] + dense_weights[5]

                return predict

        return lambda states: self._predict_q_values(
            model,
            np.atleast_2d(np.asarray(states, dtype=float)),
        )

    @staticmethod
    def _reward_utility(raw_returns):
        clipped_returns = np.clip(raw_returns, -0.05, 0.05)
        downside = np.maximum(-clipped_returns, 0.0)
        return clipped_returns - 0.5 * downside

    def _immediate_action_targets(self, current_positions, raw_returns):
        """Return known flat and long rewards for every historical state."""
        current_positions = np.asarray(current_positions, dtype=int)
        utility = self._reward_utility(raw_returns)
        flat_rewards = -current_positions * self.transaction_cost
        long_rewards = utility - (1 - current_positions) * self.transaction_cost
        return np.column_stack((flat_rewards, long_rewards))

    def _state_rollout(self, market_features, raw_returns, predictor, threshold=0.0,
                       epsilon=0.0, rng=None, initial_position=0,
                       initial_holding_period=0, initial_cooldown=0,
                       initial_drawdown=0.0, initial_max_drawdown=0.0,
                       portfolio_returns=None):
        row_count = len(market_features)
        raw_returns = np.asarray(raw_returns, dtype=float)
        if portfolio_returns is None:
            portfolio_returns = raw_returns
        portfolio_returns = np.asarray(portfolio_returns, dtype=float)
        if len(raw_returns) != row_count or len(portfolio_returns) != row_count:
            raise ValueError("Features, reward returns, and portfolio returns must align.")
        state_width = market_features.shape[1] + 5
        states = np.empty((row_count, state_width), dtype=float)
        q_values = np.empty((row_count, 2), dtype=float)
        actions = np.empty(row_count, dtype=int)
        positions = np.empty(row_count, dtype=int)
        holding_periods = np.empty(row_count, dtype=int)
        cooldowns = np.empty(row_count, dtype=int)
        drawdowns = np.empty(row_count, dtype=float)
        max_drawdowns = np.empty(row_count, dtype=float)
        rewards = np.empty(row_count, dtype=float)

        current_position = int(initial_position)
        holding_period = int(initial_holding_period)
        cooldown = int(initial_cooldown)
        current_drawdown = float(initial_drawdown)
        max_drawdown = float(initial_max_drawdown)
        portfolio_log_return = np.log1p(current_drawdown)
        portfolio_peak = 0.0
        utility = self._reward_utility(raw_returns)

        for row in range(row_count):
            holding_progress = (
                min(holding_period / self.min_holding_period, 1.0)
                if self.min_holding_period else 0.0
            )
            cooldown_progress = (
                min(cooldown / self.cooldown_period, 1.0)
                if self.cooldown_period else 0.0
            )
            state = np.concatenate(
                (
                    market_features[row],
                    [
                        current_position,
                        holding_progress,
                        cooldown_progress,
                        current_drawdown,
                        max_drawdown,
                    ],
                )
            )
            row_q_values = predictor(state)[0]
            if rng is not None and rng.random() <= epsilon:
                action = int(rng.integers(2))
            else:
                action = int(row_q_values[1] - row_q_values[0] > threshold)

            if current_position == 1 and holding_period < self.min_holding_period:
                action = 1
            elif current_position == 0 and cooldown > 0:
                action = 0

            turnover = abs(action - current_position)
            reward = (utility[row] if action == 1 else 0.0)
            reward -= turnover * self.transaction_cost

            states[row] = state
            q_values[row] = row_q_values
            actions[row] = action
            positions[row] = current_position
            holding_periods[row] = holding_period
            cooldowns[row] = cooldown
            drawdowns[row] = current_drawdown
            max_drawdowns[row] = max_drawdown
            rewards[row] = reward

            portfolio_log_return += action * portfolio_returns[row]
            portfolio_log_return -= turnover * self.transaction_cost
            portfolio_peak = max(portfolio_peak, portfolio_log_return)
            current_drawdown = np.exp(portfolio_log_return - portfolio_peak) - 1.0
            max_drawdown = min(max_drawdown, current_drawdown)

            if action == 1:
                holding_period = holding_period + 1 if current_position == 1 else 1
                cooldown = 0
            else:
                holding_period = 0
                if current_position == 1:
                    cooldown = self.cooldown_period
                elif cooldown > 0:
                    cooldown -= 1
            current_position = action

        return {
            "states": states,
            "q_values": q_values,
            "actions": actions,
            "positions": positions,
            "holding_periods": holding_periods,
            "cooldowns": cooldowns,
            "drawdowns": drawdowns,
            "max_drawdowns": max_drawdowns,
            "rewards": rewards,
        }

    @staticmethod
    def _append_rollout_to_replay(replay_buffer, rollout):
        states = rollout["states"]
        row_count = len(states)
        for row in range(row_count):
            terminal = row == row_count - 1
            next_state = np.zeros_like(states[row]) if terminal else states[row + 1]
            replay_buffer.append(
                (
                    states[row].copy(),
                    int(rollout["actions"][row]),
                    float(rollout["rewards"][row]),
                    next_state.copy(),
                    terminal,
                )
            )

    def _train_replay_batch(self, model, target_model, replay_buffer, rng):
        batch_size = min(self.replay_batch_size, len(replay_buffer))
        sample_indices = rng.choice(len(replay_buffer), size=batch_size, replace=False)
        samples = [replay_buffer[int(index)] for index in sample_indices]
        states = np.stack([sample[0] for sample in samples])
        actions = np.fromiter((sample[1] for sample in samples), dtype=int)
        rewards = np.fromiter((sample[2] for sample in samples), dtype=float)
        next_states = np.stack([sample[3] for sample in samples])
        terminals = np.fromiter((sample[4] for sample in samples), dtype=bool)

        online_predictor = self._model_predictor(model)
        current_q_values = online_predictor(states)
        next_actions = np.argmax(online_predictor(next_states), axis=1)
        target_q_values = self._model_predictor(target_model)(next_states)
        next_values = target_q_values[np.arange(batch_size), next_actions]
        td_targets = rewards + self.gamma * (~terminals) * next_values

        training_targets = current_q_values.copy()
        training_targets[np.arange(batch_size), actions] = td_targets
        return model.train_on_batch(states, training_targets)

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
        close = self.data[(stock, "Close")]
        one_step_returns = np.log(close.shift(-1) / close)
        portfolio_returns_train = one_step_returns.reindex(X_train.index)
        portfolio_returns_test = one_step_returns.reindex(X_test.index)

        model = self.build_model(input_dim=X_train_norm.shape[1] + 5)
        train_all_actions = self.gamma == 0
        target_model = None
        if not train_all_actions:
            target_model = self.build_model(input_dim=X_train_norm.shape[1] + 5)
            target_model.set_weights(model.get_weights())
        replay_buffer = deque(maxlen=self.replay_capacity) if target_model is not None else None
        epsilon = 0.0 if train_all_actions else float(self.epsilon)
        reward_values = rewards_train.to_numpy(dtype=float)
        started = time.perf_counter()
        total_steps = 0
        training_updates = 0
        if self.verbose:
            print(
                f"[DQN] {stock}: {len(X_train_norm)} training rows, "
                f"{self.episodes} episodes | "
                f"mode={'all-action targets' if train_all_actions else 'Double DQN'}",
                flush=True,
            )

        for episode in range(1, self.episodes + 1):
            episode_steps = len(X_train_norm)
            rollout = self._state_rollout(
                X_train_norm,
                reward_values,
                self._model_predictor(model),
                epsilon=epsilon,
                rng=None if train_all_actions else rng,
                portfolio_returns=portfolio_returns_train.to_numpy(dtype=float),
            )
            realized_rewards = rollout["rewards"]

            if train_all_actions:
                training_targets = self._immediate_action_targets(
                    rollout["positions"],
                    reward_values,
                )
                model.train_on_batch(
                    rollout["states"],
                    training_targets,
                )
                training_updates += 1
            else:
                self._append_rollout_to_replay(replay_buffer, rollout)
                replay_updates = self.replay_updates_per_episode
                if replay_updates is None:
                    replay_updates = max(
                        1,
                        int(np.ceil(episode_steps / self.replay_batch_size)),
                    )
                for _ in range(replay_updates):
                    self._train_replay_batch(
                        model,
                        target_model,
                        replay_buffer,
                        rng,
                    )
                    training_updates += 1
                    if training_updates % self.target_update_interval == 0:
                        target_model.set_weights(model.get_weights())

            episode_reward = float(realized_rewards.sum())
            total_steps += episode_steps

            if not train_all_actions and epsilon > self.epsilon_min:
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
                    f"updates={training_updates} | "
                    f"replay={len(replay_buffer) if replay_buffer is not None else 0} | "
                    f"elapsed={elapsed:.1f}s | {rate:.1f} steps/s",
                    flush=True,
                )

        training_seconds = time.perf_counter() - started

        self.models[stock] = model
        if target_model is not None:
            self.target_models[stock] = target_model
        self.train_data[stock] = {
            "X_train": X_train,
            "X_test": X_test,
            "rewards_train": rewards_train,
            "rewards_test": rewards_test,
            "portfolio_returns_train": portfolio_returns_train,
            "portfolio_returns_test": portfolio_returns_test,
            "index_train": pd.Index(index_train),
            "index_test": pd.Index(index_test),
            "mu": mu,
            "sigma": sigma,
            "epsilon_final": epsilon,
            "X_train_norm": X_train_norm,
            "X_test_norm": X_test_norm,
            "StateFeatures": [
                *self.features,
                "CurrentPosition",
                "HoldingProgress",
                "CooldownProgress",
                "CurrentDrawdown",
                "MaxDrawdown",
            ],
        }
        self.training_info[stock] = {
            "SplitRatio": self.split_ratio,
            "Episodes": self.episodes,
            "EpsilonFinal": epsilon,
            "TrainingSeconds": training_seconds,
            "TrainingUpdates": training_updates,
            "TransactionCost": self.transaction_cost,
            "MinHoldingPeriod": self.min_holding_period,
            "CooldownPeriod": self.cooldown_period,
            "PredictionHorizon": self.prediction_horizon,
            "TrainingMode": "AllActionTargets" if train_all_actions else "DoubleDQN",
            "TargetUpdateInterval": self.target_update_interval if target_model is not None else None,
            "DoubleDQN": target_model is not None,
            "ReplayCapacity": self.replay_capacity if replay_buffer is not None else None,
            "ReplayBatchSize": self.replay_batch_size if replay_buffer is not None else None,
            "ReplaySize": len(replay_buffer) if replay_buffer is not None else 0,
            "NetworkArchitecture": f"Dense({self.hidden_units}, swish) -> Dense(2, linear)",
            "Loss": "Huber(delta=0.02)",
        }

    def predict_signals(self, stock, mode="backtest", threshold=0.0, initial_position=0,
                        initial_holding_period=0, initial_cooldown=0,
                        initial_drawdown=0.0, initial_max_drawdown=0.0):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} has not been trained.")
        if stock not in self.train_data:
            raise ValueError(f"Training data for {stock} is missing.")
        if initial_position not in (0, 1):
            raise ValueError("initial_position must be 0 (flat) or 1 (long).")
        for name, value in {
            "initial_holding_period": initial_holding_period,
            "initial_cooldown": initial_cooldown,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not -1 < initial_drawdown <= 0:
            raise ValueError("initial_drawdown must be greater than -1 and at most 0.")
        if not -1 < initial_max_drawdown <= initial_drawdown:
            raise ValueError(
                "initial_max_drawdown must be greater than -1 and no greater than initial_drawdown."
            )

        model = self.models[stock]
        train_data = self.train_data[stock]
        mu = train_data["mu"]
        sigma = train_data["sigma"]

        if mode == "backtest":
            X_pred = train_data["X_test"]
            index_used = train_data["index_test"]
            raw_returns = train_data["rewards_test"].to_numpy(dtype=float)
            portfolio_returns = train_data["portfolio_returns_test"].to_numpy(dtype=float)
        elif mode == "live":
            X_full = self.build_feature_frame(stock)[
                self.features
            ]
            X_pred = X_full.iloc[[-1]]
            index_used = pd.Index([X_full.index[-1]])
            raw_returns = np.zeros(1, dtype=float)
            portfolio_returns = np.zeros(1, dtype=float)
        else:
            raise ValueError("mode must be 'backtest' or 'live'")

        if len(X_pred) == 0:
            return pd.DataFrame(
                columns=[
                    "Prediction",
                    "CurrentPosition",
                    "HoldingPeriod",
                    "CooldownRemaining",
                    "CurrentDrawdown",
                    "MaxDrawdown",
                    "FlatQ",
                    "LongQ",
                    "SignalStrength",
                    "Position",
                    "Signal",
                    "Turnover",
                    "TradingCost",
                    "return",
                ]
            )

        X_pred_norm = ((X_pred - mu) / sigma).to_numpy(dtype=float)
        rollout = self._state_rollout(
            X_pred_norm,
            raw_returns,
            self._model_predictor(model),
            threshold=threshold,
            initial_position=initial_position,
            initial_holding_period=initial_holding_period,
            initial_cooldown=initial_cooldown,
            initial_drawdown=initial_drawdown,
            initial_max_drawdown=initial_max_drawdown,
            portfolio_returns=portfolio_returns,
        )
        current_positions = rollout["positions"]
        predictions = rollout["actions"]
        q_values = rollout["q_values"]
        flat_q = q_values[:, 0]
        long_q = q_values[:, 1]
        q_edge = long_q - flat_q

        signals = pd.DataFrame(index=index_used)
        signals["Prediction"] = predictions
        signals["CurrentPosition"] = current_positions
        signals["HoldingPeriod"] = rollout["holding_periods"]
        signals["CooldownRemaining"] = rollout["cooldowns"]
        signals["CurrentDrawdown"] = rollout["drawdowns"]
        signals["MaxDrawdown"] = rollout["max_drawdowns"]
        signals["FlatQ"] = flat_q
        signals["LongQ"] = long_q
        signals["SignalStrength"] = q_edge
        signals["Position"] = predictions.astype(int)
        signals["Signal"] = 0
        signals.loc[signals["Position"] > signals["CurrentPosition"], "Signal"] = 1
        signals.loc[signals["Position"] < signals["CurrentPosition"], "Signal"] = -1
        signals["Turnover"] = np.abs(
            signals["Position"] - signals["CurrentPosition"]
        )
        signals["TradingCost"] = signals["Turnover"] * self.transaction_cost

        close = self.data[(stock, "Close")]
        signals["return"] = np.log(close / close.shift(1)).reindex(index_used)
        return signals

    def generate_signal_strategy(self, stock, mode="backtest", threshold=0.0,
                                 initial_position=0, initial_holding_period=0,
                                 initial_cooldown=0, initial_drawdown=0.0,
                                 initial_max_drawdown=0.0):
        if stock not in self.models:
            self.train_model(stock)

        signals = self.predict_signals(
            stock,
            mode=mode,
            threshold=threshold,
            initial_position=initial_position,
            initial_holding_period=initial_holding_period,
            initial_cooldown=initial_cooldown,
            initial_drawdown=initial_drawdown,
            initial_max_drawdown=initial_max_drawdown,
        )
        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest", threshold=0.0, initial_position=0,
                initial_holding_period=0, initial_cooldown=0,
                initial_drawdown=0.0, initial_max_drawdown=0.0):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.signal_data[stock] = self.generate_signal_strategy(
                stock,
                mode=mode,
                threshold=threshold,
                initial_position=initial_position,
                initial_holding_period=initial_holding_period,
                initial_cooldown=initial_cooldown,
                initial_drawdown=initial_drawdown,
                initial_max_drawdown=initial_max_drawdown,
            )
        self.calculate_returns()
