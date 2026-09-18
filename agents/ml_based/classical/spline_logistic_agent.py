from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from agents.base_agents.ml_trading_agent import MLBasedAgent


class SplineLogisticAgent(MLBasedAgent):
    """Regularized logistic classifier with smooth nonlinear feature effects."""

    def __init__(
        self,
        data,
        n_knots=4,
        regularization=0.5,
        proba_threshold=0.55,
        timing="open",
    ):
        if not isinstance(n_knots, int) or n_knots < 2:
            raise ValueError("n_knots must be an integer of at least 2.")
        if regularization <= 0:
            raise ValueError("regularization must be positive.")
        if not 0.0 < proba_threshold < 1.0:
            raise ValueError("proba_threshold must be between 0 and 1.")
        if timing not in ("open", "close"):
            raise ValueError("timing must be 'open' or 'close'.")

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "splines",
                    SplineTransformer(
                        n_knots=n_knots,
                        degree=3,
                        knots="quantile",
                        extrapolation="constant",
                        include_bias=False,
                    ),
                ),
                ("classifier", LogisticRegression(C=regularization, max_iter=1000)),
            ]
        )
        features = ["Return_1D", "Return_5D", "Volatility_5D", "Volume_Change"]
        super().__init__(data, model=model, features=features)
        self.algorithm_name = "SplineLogistic"
        self.n_knots = n_knots
        self.regularization = regularization
        self.proba_threshold = proba_threshold
        self.timing = timing

    def feature_engineering(self, stock):
        return self.default_feature_engineering(stock, timing=self.timing)

    def create_train_split_group(self, X, Y, split_ratio):
        X_train, X_test, y_train, y_test = super().create_train_split_group(X, Y, split_ratio)
        # The final training target depends on the first held-out close.
        return X_train.iloc[:-1], X_test, y_train.iloc[:-1], y_test

    def predict_signals(self, stock, mode="backtest", threshold=None, timing=None):
        if threshold is None:
            threshold = self.proba_threshold
        if timing is None:
            timing = self.timing
        return super().predict_signals(stock, mode=mode, threshold=threshold, timing=timing)

    def generate_signal_strategy(self, stock, mode="backtest"):
        if stock not in self.models:
            self.train_model(stock)
        signals = self.predict_signals(stock, mode=mode)
        self.signal_data[stock] = signals
        return signals
