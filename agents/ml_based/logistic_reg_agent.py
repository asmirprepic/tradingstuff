from agents.base_agents.ml_trading_agent import MLBasedAgent
from sklearn.linear_model import LogisticRegression

class LRAgent(MLBasedAgent):
    """
    A trading agent using Logistic Regression to generate trading signals.
    Inherits the full ML pipeline from MLBasedAgent and only specifies model + features.
    """

    def __init__(self, data):
        model = LogisticRegression(max_iter=1000)
        features = ['OC', 'HL', 'Return_1D', 'Return_5D', 'MA_5', 'MA_10', 'Momentum', 'Volatility_5D', 'Volume_Change']
        super().__init__(data, model=model, features=features)
        self.algorithm_name = 'Logistic_Regression'

    def feature_engineering(self, stock):
        """
        Optionally override if you'd like to customize; otherwise use default.
        """
        return self.default_feature_engineering(stock)

    def generate_signal_strategy(self, stock, mode='backtest'):
        """
        Generate trading signals for a single stock using the trained model.
        """
        print(f"[{stock}] Running generate_signal_strategy in {mode} mode.")

        self.train_model(stock)
        signals = self.predict_signals(stock)
        self.signal_data[stock] = signals
