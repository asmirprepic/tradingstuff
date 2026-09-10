from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.base_agents.ml_trading_agent import MLBasedAgent


class GaussianProcessAgent(MLBasedAgent):
    """Gaussian-process classifier for bounded per-stock training windows."""

    def __init__(
        self,
        data,
        max_samples=250,
        timing="open",
        proba_threshold=0.55,
        length_scale=1.0,
        random_state=42,
    ):
        if not isinstance(max_samples, int) or max_samples < 20:
            raise ValueError("max_samples must be an integer of at least 20.")
        if timing not in ("open", "close"):
            raise ValueError("timing must be 'open' or 'close'.")
        if not 0.0 < proba_threshold < 1.0:
            raise ValueError("proba_threshold must be between 0 and 1.")
        if length_scale <= 0:
            raise ValueError("length_scale must be positive.")

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            length_scale=length_scale,
            length_scale_bounds="fixed",
        )
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "gaussian_process",
                    GaussianProcessClassifier(
                        kernel=kernel,
                        random_state=random_state,
                        n_restarts_optimizer=0,
                    ),
                ),
            ]
        )
        features = [
            "Return_1D",
            "Return_5D",
            "Momentum",
            "Volatility_5D",
            "Volume_Change",
            "OC",
            "HL",
        ]
        super().__init__(data, model=model, features=features)
        self.algorithm_name = "GaussianProcess"
        self.max_samples = max_samples
        self.timing = timing
        self.proba_threshold = proba_threshold
        self.length_scale = length_scale
        self.random_state = random_state

    def feature_engineering(self, stock):
        features, target = self.default_feature_engineering(stock, timing=self.timing)
        return features.tail(self.max_samples), target.tail(self.max_samples)

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
