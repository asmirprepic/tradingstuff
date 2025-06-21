from agents.base_agents.ml_trading_agent import MLBasedAgent
from sklearn.naive_bayes import GaussianNB
import logging

logger = logging.getLogger(__name__)

class NaiveBayesAgent(MLBasedAgent):
    """
    A trading agent using Gaussian Naive Bayes to generate trading signals.
    Inherits the full ML pipeline from MLBasedAgent and specifies model + features.
    """

    def __init__(self, data):
        model = GaussianNB()
        features = ['Open-Close', 'High-Low']
        super().__init__(data, model=model, features=features)
        self.algorithm_name = 'Naive_Bayes'

    def generate_signal_strategy(self, stock, mode='backtest'):
        logger.info(f"[{stock}] Running generate_signal_strategy in {mode} mode.")
        self.train_model(stock)
        signals = self.predict_signals(stock, mode=mode)
        self.signal_data[stock] = signals
        return signals
