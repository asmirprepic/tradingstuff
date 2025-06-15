from agents.base_agents.ml_trading_agent import MLBasedAgent
from sklearn.neighbors import KNeighborsClassifier

class KNNAgent(MLBasedAgent):
    """
    A trading agent using the K-Nearest Neighbors (KNN) classifier to generate trading signals.
    """

    def __init__(self, data, n_neighbors=15):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        features = ['Open-Close', 'High-Low']
        super().__init__(data, model=model, features=features)
        self.algorithm_name = "KNN"
        self.models = {}

    def feature_engineering(self, stock):
        return self.default_feature_engineering(stock)

    def generate_signal_strategy(self, stock, mode='backtest'):
        print(f"[{stock}] Generating signals in {mode} mode using KNN...")

        # Train model if needed
        if stock not in self.models:
            X, Y = self.feature_engineering(stock)
            X_train, X_test, Y_train, Y_test = self.create_train_split_group(X, Y, split_ratio=0.8)
            model = KNeighborsClassifier(n_neighbors=self.model.n_neighbors)
            model.fit(X_train, Y_train)
            self.models[stock] = model
        else:
            model = self.models[stock]

        self.model = model  # Use this model in base class
        self.trained = True  # Allow predict_signals to proceed

        signals = self.predict_signals(stock)
        self.signal_data[stock] = signals
