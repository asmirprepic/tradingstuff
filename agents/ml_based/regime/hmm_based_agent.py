import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agents.base_agents.trading_agent import TradingAgent

try:
    from hmmlearn.hmm import GaussianHMM
except ModuleNotFoundError as exc:
    GaussianHMM = None
    _HMMLEARN_IMPORT_ERROR = exc
else:
    _HMMLEARN_IMPORT_ERROR = None


class HMMRegimeAgent(TradingAgent):
    def __init__(self, data, n_states=3, split_ratio=0.8, auto_generate=False):
        super().__init__(data)
        self.algorithm_name = "HMMRegime"
        self.score_column = "SignalStrength"
        self.n_states = int(n_states)
        self.split_ratio = float(split_ratio)
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        self.hmm_models = {}
        self.scalers = {}
        self.train_data = {}
        self.best_regimes = {}
        self.regime_return_maps = {}

        if auto_generate:
            self.run_all()

    def _require_hmmlearn(self):
        if GaussianHMM is None:
            raise ModuleNotFoundError(
                "hmmlearn is required to use HMMRegimeAgent. Install hmmlearn before training this agent."
            ) from _HMMLEARN_IMPORT_ERROR

    def feature_engineering(self, stock):
        df = self.data[stock].copy()
        df["Return_1D"] = df["Close"].pct_change()
        df["Volatility"] = df["Close"].pct_change().rolling(10).std()
        df["Volume_Change"] = df["Volume"].pct_change()
        df["return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df.dropna(subset=["Return_1D", "Volatility", "Volume_Change", "return"])

    def create_train_split_group(self, df, split_ratio=None):
        split_ratio = self.split_ratio if split_ratio is None else split_ratio
        return train_test_split(df, shuffle=False, test_size=1 - split_ratio)

    def train_hmm(self, stock, split_ratio=None):
        self._require_hmmlearn()

        df = self.feature_engineering(stock)
        df_train, df_test = self.create_train_split_group(df, split_ratio=split_ratio)

        feature_cols = ["Return_1D", "Volatility", "Volume_Change"]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[feature_cols])

        hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        hmm.fit(X_train)

        train_regimes = hmm.predict(X_train)
        df_train = df_train.copy()
        df_train["Regime"] = train_regimes

        regime_returns = df_train.groupby("Regime")["Return_1D"].mean()
        best_regime = int(regime_returns.idxmax())

        self.hmm_models[stock] = hmm
        self.scalers[stock] = scaler
        self.best_regimes[stock] = best_regime
        self.regime_return_maps[stock] = regime_returns.to_dict()
        self.train_data[stock] = {
            "df_train": df_train,
            "df_test": df_test.copy(),
            "feature_cols": feature_cols,
            "index_train": pd.Index(df_train.index),
            "index_test": pd.Index(df_test.index),
        }

    def predict_signals(self, stock, mode="backtest"):
        if stock not in self.hmm_models:
            raise ValueError(f"HMM for {stock} has not been trained.")

        train_data = self.train_data[stock]
        feature_cols = train_data["feature_cols"]
        scaler = self.scalers[stock]
        hmm = self.hmm_models[stock]
        best_regime = self.best_regimes[stock]

        if mode == "backtest":
            df_pred = train_data["df_test"].copy()
        elif mode == "live":
            df_full = self.feature_engineering(stock)
            df_pred = df_full.iloc[[-1]].copy()
        else:
            raise ValueError("mode must be 'backtest' or 'live'")

        if df_pred.empty:
            return pd.DataFrame(
                columns=[
                    "Regime",
                    "Good_Regime",
                    "SignalStrength",
                    "Position",
                    "Signal",
                    "return",
                ]
            )

        X_scaled = scaler.transform(df_pred[feature_cols].values)
        regimes = hmm.predict(X_scaled)
        regime_probs = hmm.predict_proba(X_scaled)
        best_regime_prob = regime_probs[:, best_regime]

        signals = pd.DataFrame(index=df_pred.index)
        signals["Regime"] = regimes
        signals["Good_Regime"] = (signals["Regime"] == best_regime).astype(int)
        signals["SignalStrength"] = best_regime_prob
        signals["Position"] = signals["Good_Regime"].astype(int)
        signals["Signal"] = 0
        signals.loc[signals["Position"] > signals["Position"].shift(1), "Signal"] = 1
        signals.loc[signals["Position"] < signals["Position"].shift(1), "Signal"] = -1
        signals["return"] = df_pred["return"]
        return signals

    def generate_signal_strategy(self, stock, mode="backtest"):
        if stock not in self.hmm_models:
            self.train_hmm(stock)

        signals = self.predict_signals(stock, mode=mode)
        self.signal_data[stock] = signals
        return signals

    def run_all(self, mode="backtest"):
        self.signal_data = {}
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock, mode=mode)
        self.calculate_returns()

    def run_regime_strategy(self, stocks=None, mode="backtest"):
        stocks = self.stocks_in_data if stocks is None else stocks
        self.signal_data = {}
        for stock in stocks:
            self.generate_signal_strategy(stock, mode=mode)
        self.calculate_returns()
