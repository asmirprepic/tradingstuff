from agents.base_agents.ml_trading_agent import MLBasedAgent
from sklearn.neighbors import KNeighborsClassifier

class KNNAgent(MLBasedAgent):
    """
    A trading agent using the K-Nearest Neighbors (KNN) classifier to generate trading signals.
    """

    def __init__(self, data, n_neighbors=15):
        """
        Initializes the KNN-based agent.

        Args:
            data (pd.DataFrame): MultiIndex DataFrame with OHLCV data.
            n_neighbors (int): Number of neighbors to use in KNN.
        """
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        features = ['OC', 'HL', 'Return_1D', 'Return_5D', 'MA_5', 'MA_10', 'Momentum', 'Volatility_5D', 'Volume_Change']
        super().__init__(data, model=model, features=features)
        self.algorithm_name = "KNN"

    def feature_engineering(self, stock):
        """
        Uses the default feature engineering: Open-Close and High-Low.
        """
        return self.default_feature_engineering(stock)

    def generate_signal_strategy(self, stock, mode='backtest'):
        """
        Train (if needed) and generate signals for a specific stock using KNN.

        Args:
            stock (str): The stock symbol to generate signals for.
            mode (str): 'backtest' or 'live' mode.
        """
        print(f"[{stock}] Generating signals in {mode} mode using {self.algorithm_name}...")


        if stock not in self.models:
            self.train_model(stock)


        self.signal_data[stock] = self.predict_signals(stock, mode=mode)
