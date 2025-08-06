from agents.base_agents.ml_trading_agent import MLBasedAgent
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class ClusteringFilteredKNNAgent(MLBasedAgent):
    def __init__(self, data, n_neighbors=10, n_clusters=3, cluster_lookback=20):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        features = ['Return_1D', 'OC', 'HL']
        super().__init__(data, model=model, features=features)

        self.algorithm_name = "ClusteringFilteredKNN"
        self.n_clusters = n_clusters
        self.cluster_lookback = cluster_lookback

    def feature_engineering(self, stock):
        return self.default_feature_engineering(stock)

    def generate_signal_strategy(self, stock, mode='backtest'):
        print(f"[{stock}] Generating signals in {mode} mode using {self.algorithm_name}...")
        if stock not in self.models:
            self.train_model(stock)
        self.signal_data[stock] = self.predict_signals(stock, mode=mode)

    def filter_stocks_by_cluster(self, as_of_date=None):
        """
        Applies clustering using price/volume features and sets self.filtered_stocks
        """
        stock_features = []
        stock_names = self.data.columns.get_level_values(0).unique()

        for stock in stock_names:
            try:
                price = self.data[(stock, 'Close')]
                volume = self.data[(stock, 'Volume')]
                if as_of_date:
                    price = price.loc[:as_of_date]
                    volume = volume.loc[:as_of_date]

                returns = price.pct_change()
                momentum = price.pct_change(self.cluster_lookback).shift(1).iloc[-1]
                volatility = returns.rolling(self.cluster_lookback).std().shift(1).iloc[-1]
                avg_volume = volume.rolling(self.cluster_lookback).mean().shift(1).iloc[-1]

                stock_features.append([momentum, volatility, avg_volume])
            except Exception:
                continue

        df_feat = pd.DataFrame(stock_features, index=stock_names,
                               columns=['momentum', 'volatility', 'volume']).dropna()
        X_scaled = StandardScaler().fit_transform(df_feat)

        kmeans = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=42)
        df_feat['cluster'] = kmeans.fit_predict(X_scaled)

        best_cluster = df_feat.groupby('cluster')['momentum'].mean().idxmax()
        selected = df_feat[df_feat['cluster'] == best_cluster].index.tolist()

        self.filtered_stocks = selected
        print(f"[CLUSTERING] Selected stocks: {selected}")
