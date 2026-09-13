import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.base_agents.ml_trading_agent import MLBasedAgent


class OnlineSGDAgent(MLBasedAgent):
    """Incremental logistic classifier that learns from newly labeled bars."""

    def __init__(
        self,
        data,
        timing="open",
        proba_threshold=0.55,
        alpha=0.0001,
        penalty="l2",
        random_state=42,
    ):
        if timing not in ("open", "close"):
            raise ValueError("timing must be 'open' or 'close'.")
        if not 0.0 < proba_threshold < 1.0:
            raise ValueError("proba_threshold must be between 0 and 1.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if penalty not in ("l2", "l1", "elasticnet", None):
            raise ValueError("penalty must be 'l2', 'l1', 'elasticnet', or None.")

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=alpha,
                        penalty=penalty,
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
        self.algorithm_name = "OnlineSGD"
        self.timing = timing
        self.proba_threshold = proba_threshold
        self.alpha = alpha
        self.penalty = penalty
        self.random_state = random_state

    def feature_engineering(self, stock):
        return self.default_feature_engineering(stock, timing=self.timing)

    def train_model(self, stock, split_ratio=0.8):
        metrics = super().train_model(stock, split_ratio=split_ratio)
        features, _ = self.default_feature_engineering(stock, force_refresh=True, timing=self.timing)
        self.training_info[stock].update(
            {
                "OnlineLastSeen": features.index[-1] if not features.empty else None,
                "OnlineUpdates": 0,
                "OnlineSamples": 0,
            }
        )
        return metrics

    def update_model(self, stock):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} has not been trained.")

        features, target = self.default_feature_engineering(stock, force_refresh=True, timing=self.timing)
        last_seen = self.training_info.get(stock, {}).get("OnlineLastSeen")
        if last_seen is not None:
            new_mask = features.index > pd.Timestamp(last_seen)
            features = features.loc[new_mask]
            target = target.loc[new_mask]
        if features.empty:
            return 0

        pipeline = self.models[stock]
        scaler = pipeline.named_steps["scaler"]
        classifier = pipeline.named_steps["classifier"]
        classifier.partial_fit(scaler.transform(features), target)

        info = self.training_info.setdefault(stock, {})
        info["OnlineLastSeen"] = features.index[-1]
        info["OnlineUpdates"] = int(info.get("OnlineUpdates", 0)) + 1
        info["OnlineSamples"] = int(info.get("OnlineSamples", 0)) + len(features)
        return len(features)

    def generate_signal_strategy(self, stock, mode="backtest"):
        if stock not in self.models:
            self.train_model(stock)
        signals = self.predict_signals(
            stock,
            mode=mode,
            threshold=self.proba_threshold,
            timing=self.timing,
        )
        self.signal_data[stock] = signals
        return signals
