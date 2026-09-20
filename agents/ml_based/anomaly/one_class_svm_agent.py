import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from agents.base_agents.trading_agent import TradingAgent


class OneClassSVMAgent(TradingAgent):
    """Rank unusual OHLCV observations without assigning trade direction."""

    def __init__(self, data, nu=0.05, alert_percentile=95.0, split_ratio=0.8):
        super().__init__(data)
        if not 0 < nu <= 1:
            raise ValueError("nu must be in (0, 1].")
        if not 0 < alert_percentile < 100:
            raise ValueError("alert_percentile must be between 0 and 100.")
        if not 0 < split_ratio < 1:
            raise ValueError("split_ratio must be between 0 and 1.")

        self.algorithm_name = "OneClassSVM"
        self.score_column = "SignalStrength"
        self.features = ["Return_1D", "Return_5D", "RangePct", "VolumeChange", "Volatility_5D"]
        self.nu = nu
        self.alert_percentile = alert_percentile
        self.split_ratio = split_ratio
        self.models = {}
        self.train_data = {}
        self.thresholds = {}
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

    def feature_engineering(self, stock):
        bars = self.data[stock]
        close = bars["Close"]
        returns = close.pct_change(fill_method=None)
        features = pd.DataFrame(
            {
                "Return_1D": returns,
                "Return_5D": close.pct_change(5, fill_method=None),
                "RangePct": (bars["High"] - bars["Low"]) / close,
                "VolumeChange": bars["Volume"].pct_change(fill_method=None),
                "Volatility_5D": returns.rolling(5).std(),
            },
            index=bars.index,
        )
        return features.replace([np.inf, -np.inf], np.nan).dropna()

    def train_model(self, stock):
        features = self.feature_engineering(stock)
        boundary = int(len(features) * self.split_ratio)
        train = features.iloc[:boundary]
        test = features.iloc[boundary:]
        if len(train) < 20 or len(test) < 5:
            raise ValueError(f"{stock}: need at least 20 training and 5 test rows.")

        model = Pipeline([("scaler", StandardScaler()), ("svm", OneClassSVM(nu=self.nu, gamma="scale"))])
        model.fit(train)
        train_scores = -model.decision_function(train).ravel()
        test_scores = -model.decision_function(test).ravel()
        threshold = float(np.percentile(train_scores, self.alert_percentile))

        self.models[stock] = model
        self.train_data[stock] = (train, test, None, None)
        self.thresholds[stock] = threshold
        self.training_info[stock] = {
            "SplitRatio": self.split_ratio,
            "Nu": self.nu,
            "AlertPercentile": self.alert_percentile,
            "Threshold": threshold,
            "TrainAnomalyRate": float(np.mean(train_scores > threshold)),
            "TestAnomalyRate": float(np.mean(test_scores > threshold)),
        }
        return self.training_info[stock]

    def predict_signals(self, stock, mode="backtest"):
        if stock not in self.models or stock not in self.thresholds:
            raise ValueError(f"Model for {stock} has not been trained.")
        if mode == "backtest":
            features = self.train_data[stock][1]
        elif mode == "live":
            features = self.feature_engineering(stock).tail(1)
        else:
            raise ValueError("mode must be 'backtest' or 'live'.")
        if features.empty:
            raise ValueError(f"No usable features for {stock}.")

        threshold = self.thresholds[stock]
        anomaly_score = -self.models[stock].decision_function(features).ravel()
        signals = pd.DataFrame(index=features.index)
        signals["Anomaly"] = anomaly_score > threshold
        signals["AnomalyScore"] = anomaly_score
        signals["AnomalyThreshold"] = threshold
        signals["SignalStrength"] = anomaly_score - threshold
        signals["Position"] = 0
        signals["Signal"] = 0
        close = self.data[(stock, "Close")]
        signals["return"] = np.log(close / close.shift(1)).reindex(features.index)
        return signals

    def generate_signal_strategy(self, stock, mode="backtest"):
        if stock not in self.models:
            self.train_model(stock)
        signals = self.predict_signals(stock, mode=mode)
        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest"):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock, mode=mode)
        self.calculate_returns()
