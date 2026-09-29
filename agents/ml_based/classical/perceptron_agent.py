import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.base_agents.ml_trading_agent import MLBasedAgent


class PerceptronAgent(MLBasedAgent):
    """Scaled linear direction classifier with a signed confidence margin."""

    def __init__(
        self,
        data,
        timing="open",
        alpha=0.0001,
        max_iter=1000,
        eta0=1.0,
        random_state=42,
    ):
        if timing not in ("open", "close"):
            raise ValueError("timing must be 'open' or 'close'.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if eta0 <= 0:
            raise ValueError("eta0 must be positive.")

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    Perceptron(
                        penalty="l2",
                        alpha=alpha,
                        max_iter=max_iter,
                        eta0=eta0,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        features = [
            "OC",
            "HL",
            "Return_1D",
            "Return_5D",
            "MA_5",
            "MA_10",
            "Momentum",
            "Volatility_5D",
            "Volume_Change",
        ]
        super().__init__(data, model=model, features=features)
        self.algorithm_name = "Perceptron"
        self.score_column = "DecisionScore"
        self.timing = timing
        self.alpha = alpha
        self.max_iter = max_iter
        self.eta0 = eta0
        self.random_state = random_state

    def feature_engineering(self, stock):
        return self.default_feature_engineering(stock, timing=self.timing)

    def create_train_split_group(self, X, Y, split_ratio):
        X_train, X_test, y_train, y_test = super().create_train_split_group(X, Y, split_ratio)
        if len(X_train) < 2:
            raise ValueError("Perceptron requires at least two training rows before purging.")
        return X_train.iloc[:-1], X_test, y_train.iloc[:-1], y_test

    def predict_signals(self, stock, mode="backtest"):
        signals = super().predict_signals(stock, mode=mode, timing=self.timing)
        if mode == "backtest":
            features = self.train_data[stock][1]
        else:
            features = self.live_feature_engineering(stock)
        margin = self.models[stock].decision_function(features)
        signals["DecisionScore"] = np.asarray(margin).reshape(-1)
        return signals

    def generate_signal_strategy(self, stock, mode="backtest"):
        if stock not in self.models:
            self.train_model(stock)
        signals = self.predict_signals(stock, mode=mode)
        self.signal_data[stock] = signals
        return signals
