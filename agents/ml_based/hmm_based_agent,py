from agents.base_agents.ml_trading_agent import MLBasedAgent
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class HMMRegimeAgent(MLBasedAgent):
    def __init__(self, data, n_states=3, timing='close'):
        # This is more of a filter than a predictive classifier
        super().__init__(data, model=None, features=['Return_1D', 'Volatility', 'Volume_Change'])

        self.algorithm_name = "HMMRegime"
        self.n_states = n_states
        self.hmm_models = {}       # store per-stock HMM
        self.scalers = {}          # store per-stock scalers
        self.timing = timing

    def feature_engineering(self, stock):
        df = self.data[stock].copy()
        df['Return_1D'] = df['Close'].pct_change()
        df['Volatility'] = df['Close'].pct_change().rolling(10).std()
        df['Volume_Change'] = df['Volume'].pct_change()
        return df.dropna()

    def train_hmm(self, stock):
        df = self.feature_engineering(stock)
        X = df[self.features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        hmm = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=200, random_state=42)
        hmm.fit(X_scaled)

        self.hmm_models[stock] = hmm
        self.scalers[stock] = scaler

    def detect_regimes(self, stock):
        if stock not in self.hmm_models:
            self.train_hmm(stock)

        df = self.feature_engineering(stock)
        X_scaled = self.scalers[stock].transform(df[self.features].values)

        regimes = self.hmm_models[stock].predict(X_scaled)
        df['Regime'] = regimes

        # Identify "best" regime based on average return
        avg_returns = df.groupby('Regime')['Return_1D'].mean()
        best_regime = avg_returns.idxmax()
        df['Good_Regime'] = (df['Regime'] == best_regime).astype(int)

        self.signal_data[stock] = df
        self.best_regime = best_regime
        return df

    def generate_signal_strategy(self, stock, mode='backtest'):
        print(f"[{stock}] Generating regime-based signals in {mode} mode...")
        df = self.detect_regimes(stock)

        # Simple example: long if in best regime, else flat
        df['Signal'] = np.where(df['Good_Regime'] == 1, 1, 0)

        self.signal_data[stock] = df[['Signal']]
        return self.signal_data[stock]

    def run_regime_strategy(self, stocks=None, mode='backtest'):
        if stocks is None:
            stocks = self.data.columns.get_level_values(0).unique()

        for stock in stocks:
            self.generate_signal_strategy(stock, mode)

        self.calculate_returns()
