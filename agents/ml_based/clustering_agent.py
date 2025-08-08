from agents.base_agents.ml_trading_agent import MLBasedAgent
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class ClusteringFilteredKNNAgent(MLBasedAgent):
    def __init__(self, data, n_neighbors=10, n_clusters=3, cluster_lookback=20):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        features = ['Return_1D', 'OC', 'HL']
        super().__init__(data, model=model, features=features)

        self.algorithm_name = "ClusteringFilteredKNN"
        self.n_clusters = n_clusters
        self.cluster_lookback = cluster_lookback
        self.timing = 'open'

    def feature_engineering(self, stock):
        return self.default_feature_engineering(stock,timing=self.timing)

    def generate_signal_strategy(self, stock, mode='backtest'):
        print(f"[{stock}] Generating signals in {mode} mode using {self.algorithm_name}...")
        if stock not in self.models:
            self.train_model(stock)
        self.signal_data[stock] = self.predict_signals(stock, mode=mode)

    def filter_stocks_by_cluster(self, as_of_date=None):
        """
        Applies clustering using price/volume features and sets self.filtered_stocks
        """
        features = []
        kept_names = []
        stock_names = self.data.columns.get_level_values(0).unique()

        for stock in stock_names:
            try:
                price = self.data[(stock, 'Close')]
                volume = self.data[(stock, 'Volume')]
                if as_of_date:
                    price = price.loc[:as_of_date]
                    volume = volume.loc[:as_of_date]

                if len(price) < self.cluster_lookback + 1 or len(volume) < self.cluster_lookback + 1:
                    continue

                returns = price.pct_change()
                momentum = price.pct_change(self.cluster_lookback).shift(1).iloc[-1]
                volatility = returns.rolling(self.cluster_lookback).std().shift(1).iloc[-1]
                avg_volume = volume.rolling(self.cluster_lookback).mean().shift(1).iloc[-1]

                if np.isfinite(momentum) and np.isfinite(volatility) and np.isfinite(avg_volume):
                    features.append([momentum, volatility, avg_volume])
                    kept_names.append(stock)

            except Exception:
                continue

        df_feat = pd.DataFrame(features, index=kept_names,
                               columns=['momentum', 'volatility', 'volume']).dropna()
        X_scaled = StandardScaler().fit_transform(df_feat)

        n_clusters_eff = min(self.n_clusters, len(df_feat))
        if n_clusters_eff < 1:
            self.filtered_stocks = []
            print("[CLUSTERING] Not enough stocks to cluster.")
            return

        kmeans = KMeans(n_clusters=n_clusters_eff, n_init=10, random_state=42)
        df_feat['cluster'] = kmeans.fit_predict(X_scaled)

        best_cluster = df_feat.groupby('cluster')['momentum'].mean().idxmax()
        selected = df_feat[df_feat['cluster'] == best_cluster].index.tolist()

        self.cluster_df = df_feat
        self.kmeans_model = kmeans
        self.filtered_stocks = selected
        print(f"[CLUSTERING] Selected stocks: {selected}")

    def run_filtered(self,mode = 'backtest'):
        if not getattr(self,'filtered_stocks',None):
            print("[RUN] No filtered stocks. Call filter_stocks_by_cluster() first.")
            return
        for stock in self.filtered_stocks:
            self.generate_signal_strategy(stock,mode = mode)
        self.calculate_returns()
