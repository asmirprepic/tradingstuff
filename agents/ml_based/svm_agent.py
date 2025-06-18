from agents.base_agents.ml_trading_agent import MLBasedAgent
from sklearn.svm import SVC

class SVMAgent(MLBasedAgent):
    """
    A trading agent using Support Vector Machine (SVM) to generate trading signals.
    Inherits all logic from MLBasedAgent and plugs in SVC model and feature config.
    """

    def __init__(self, data):
        model = SVC()  # Default is fine; you can tune it later
        features = ['Open-Close', 'High-Low']
        super().__init__(data, model=model, features=features)
        self.algorithm_name = 'SVM'

    def feature_engineering(self, stock):
        """
        Optional: Override only if you want to use custom features.
        Otherwise uses default: Open-Close, High-Low.
        """
        return self.default_feature_engineering(stock)

    def generate_signal_strategy(self, stock, mode='backtest'):
        """
        Wrapper to train and generate signals (based on mode).
        """
        print(f"[{stock}] Running generate_signal_strategy in {mode} mode.")

        self.train_model(stock)
        signals = self.predict_signals(stock)
        self.signal_data[stock] = signals
