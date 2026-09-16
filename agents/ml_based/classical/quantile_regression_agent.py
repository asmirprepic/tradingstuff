import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agents.base_agents.ml_trading_agent import MLBasedAgent


class QuantileRegressionAgent(MLBasedAgent):
    """Forecast next-close returns at three quantiles without random splitting."""

    def __init__(
        self,
        data,
        quantiles=(0.1, 0.5, 0.9),
        alpha=0.1,
        min_expected_return=0.001,
        max_adverse_return=0.02,
        timing="close",
    ):
        quantiles = tuple(quantiles)
        if len(quantiles) != 3 or not all(0 < q < 1 for q in quantiles) or not (quantiles[0] < quantiles[1] < quantiles[2]):
            raise ValueError("quantiles must be three strictly increasing values between 0 and 1.")
        if alpha < 0 or min_expected_return < 0 or max_adverse_return < 0:
            raise ValueError("alpha and return thresholds must be non-negative.")
        if timing not in ("open", "close"):
            raise ValueError("timing must be 'open' or 'close'.")

        features = ["OC", "HL", "Return_1D", "Return_5D", "MA_5", "MA_10", "Momentum", "Volatility_5D", "Volume_Change"]
        super().__init__(data, features=features)
        self.algorithm_name = "QuantileRegression"
        self.quantiles = quantiles
        self.alpha = alpha
        self.min_expected_return = min_expected_return
        self.max_adverse_return = max_adverse_return
        self.timing = timing

    def feature_engineering(self, stock):
        features, _ = self.default_feature_engineering(stock, timing=self.timing)
        close = self.data[(stock, "Close")]
        next_return = close.shift(-1).div(close).sub(1)
        target = next_return.reindex(features.index)
        valid = target.notna() & np.isfinite(target)
        return features.loc[valid], target.loc[valid]

    def train_model(self, stock, split_ratio=0.8):
        if not 0 < split_ratio < 1:
            raise ValueError("split_ratio must be between 0 and 1.")
        X, y = self.feature_engineering(stock)
        boundary = int(len(X) * split_ratio)
        # The final training target uses the next bar's close, so purge it.
        X_train, y_train = X.iloc[:boundary - 1], y.iloc[:boundary - 1]
        X_test, y_test = X.iloc[boundary:], y.iloc[boundary:]
        if len(X_train) < 20 or len(X_test) < 5:
            raise ValueError(f"{stock}: need at least 20 training and 5 test rows after the time split.")

        models = {}
        predictions = {}
        for quantile in self.quantiles:
            model = make_pipeline(StandardScaler(), QuantileRegressor(quantile=quantile, alpha=self.alpha))
            model.fit(X_train, y_train)
            models[quantile] = model
            predictions[quantile] = model.predict(X_test)

        raw_lower, median, raw_upper = (predictions[q] for q in self.quantiles)
        lower = np.minimum(raw_lower, median)
        upper = np.maximum(median, raw_upper)
        metrics = {
            "mae": mean_absolute_error(y_test, median),
            "median_pinball_loss": mean_pinball_loss(y_test, median, alpha=self.quantiles[1]),
            "interval_coverage": float(np.mean((y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper))),
            "interval_width": float(np.mean(upper - lower)),
        }
        self.models[stock] = models
        self.train_data[stock] = (X_train, X_test, y_train, y_test)
        self.training_info[stock] = {
            "SplitRatio": split_ratio,
            "MAE": metrics["mae"],
            "MedianPinballLoss": metrics["median_pinball_loss"],
            "IntervalCoverage": metrics["interval_coverage"],
            "MeanIntervalWidth": metrics["interval_width"],
            "LowerQuantile": self.quantiles[0],
            "MedianQuantile": self.quantiles[1],
            "UpperQuantile": self.quantiles[2],
        }
        return metrics

    def predict_signals(self, stock, mode="backtest"):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} has not been trained.")
        if mode == "backtest":
            if stock not in self.train_data:
                raise ValueError(f"Training data for {stock} is missing.")
            features = self.train_data[stock][1]
        elif mode == "live":
            features = self.default_live_features(stock, force_refresh=True, timing=self.timing)
        else:
            raise ValueError("mode must be 'backtest' or 'live'.")

        raw = np.column_stack([self.models[stock][q].predict(features) for q in self.quantiles])
        # Independent quantile fits may cross; enforce ordered output for decisions.
        lower = np.minimum(raw[:, 0], raw[:, 1])
        upper = np.maximum(raw[:, 1], raw[:, 2])
        median = raw[:, 1]
        long = (median > self.min_expected_return) & (lower >= -self.max_adverse_return)
        short = (median < -self.min_expected_return) & (upper <= self.max_adverse_return)

        signals = pd.DataFrame(index=features.index)
        signals["ExpectedReturnLower"] = lower
        signals["ExpectedReturnMedian"] = median
        signals["ExpectedReturnUpper"] = upper
        signals["IntervalWidth"] = upper - lower
        signals["SignalStrength"] = median / np.maximum(upper - lower, 1e-6)
        signals["Position"] = np.where(long, 1, np.where(short, -1, 0))
        previous = signals["Position"].shift(1).fillna(0)
        signals["Signal"] = np.sign(signals["Position"] - previous).astype(int)
        close = self.data[(stock, "Close")]
        signals["return"] = np.log(close / close.shift(1)).reindex(features.index)
        return signals

    def generate_signal_strategy(self, stock, mode="backtest"):
        if stock not in self.models:
            self.train_model(stock)
        signals = self.predict_signals(stock, mode=mode)
        self.signal_data[stock] = signals
        return signals
