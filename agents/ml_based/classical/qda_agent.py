from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.base_agents.ml_trading_agent import MLBasedAgent


class QDAAgent(MLBasedAgent):
    """Regularized quadratic discriminant classifier for next-bar direction."""

    def __init__(self, data, reg_param=0.2, proba_threshold=0.55, timing="open"):
        if not 0.0 <= reg_param <= 1.0:
            raise ValueError("reg_param must be between 0 and 1.")
        if not 0.0 < proba_threshold < 1.0:
            raise ValueError("proba_threshold must be between 0 and 1.")
        if timing not in ("open", "close"):
            raise ValueError("timing must be 'open' or 'close'.")

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("qda", QuadraticDiscriminantAnalysis(reg_param=reg_param)),
            ]
        )
        features = ["Return_1D", "Return_5D", "Volatility_5D", "Volume_Change", "OC", "HL"]
        super().__init__(data, model=model, features=features)
        self.algorithm_name = "QDA"
        self.reg_param = reg_param
        self.proba_threshold = proba_threshold
        self.timing = timing

    def feature_engineering(self, stock):
        return self.default_feature_engineering(stock, timing=self.timing)

    def create_train_split_group(self, X, Y, split_ratio):
        X_train, X_test, y_train, y_test = super().create_train_split_group(X, Y, split_ratio)
        # The final training label uses the first validation bar's close.
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
