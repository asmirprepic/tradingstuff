from agents.base_agents.trading_agent import TradingAgent
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class DeepQLearningAgent(TradingAgent):
    """
    A trading agent that uses Deep Q-Learning to generate trading signals.
    This agent employs a deep reinforcement learning algorithm to learn trading strategies
    through trial and error interactions with the stock market environment.

    Attributes:
        algorithm_name (str): Name of the algorithm, set to "DeepQLearning".
        stocks_in_data (pd.Index): Unique stock symbols present in the data.
        signal_data (dict): A dictionary to store signal data for each stock.
    """

    def __init__(self, data, alpha=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, episodes=1000):
        super().__init__(data)
        self.algorithm_name = 'DeepQLearning'
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.model = self.build_model()

        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock)
        self.calculate_returns()

    def build_model(self):
        """
        Builds the Deep Q-Learning model.

        Returns:
            Sequential: The built neural network model.
        """
        model = Sequential()
        model.add(Dense(24, input_dim=2, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.alpha))
        return model

    def create_classification_trading_condition(self, stock):
        """
        Creates the feature set for Deep Q-Learning. The features include 
        'Open-Close' (difference between opening and closing prices) and 'High-Low' 
        (difference between high and low prices).

        Args:
            stock (str): The stock symbol for which to create features.

        Returns:
            pd.DataFrame: The feature set.
            pd.Series: The target variable, where 1 indicates an upward price movement 
                        and -1 indicates a downward price movement.
        """
        data_copy = self.data[stock].copy()
        data_copy['Open-Close'] = self.data[stock]['Open'] - self.data[stock]['Close']
        data_copy['High-Low'] = self.data[stock]['High'] - self.data[stock]['Low']
        data_copy.dropna(inplace=True)

        X = data_copy[['Open-Close', 'High-Low']].values
        Y = np.where(self.data[stock]['Close'].shift(-1) > self.data[stock]['Close'], 1, -1)
        Y_series = pd.Series(Y, index=self.data[stock].index)
        Y = Y_series.loc[data_copy.index].values

        return X, Y

    def train_deep_q_learning(self, X, Y):
        """
        Trains a Deep Q-Learning agent on the provided data.

        Args:
            X (np.ndarray): The feature set.
            Y (np.ndarray): The target variable.
        """
        for episode in range(self.episodes):
            state = 0
            while state < len(X) - 1:
                if np.random.rand() <= self.epsilon:
                    action = np.random.randint(2)
                else:
                    action = np.argmax(self.model.predict(np.array([X[state]])))

                next_state = state + 1
                reward = 1 if Y[state] == 1 and action == 1 else -1
                target = reward + self.gamma * np.amax(self.model.predict(np.array([X[next_state]])))
                target_f = self.model.predict(np.array([X[state]]))
                target_f[0][action] = target
                self.model.fit(np.array([X[state]]), target_f, epochs=1, verbose=0)
                
                state = next_state

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

    def generate_signal_strategy(self, stock):
        """
        Generates trading signals for the specified stock using the trained Deep Q-Learning model.
        A signal is generated for each day based on the Q-values.

        Args:
            stock (str): The stock symbol for which to generate signals.

        The method updates the `signal_data` attribute with signals for the given stock.
        """
        signals = pd.DataFrame(index=self.data.index)
        X, Y = self.create_classification_trading_condition(stock)
        self.train_deep_q_learning(X, Y)
        actions = [np.argmax(self.model.predict(np.array([x]))) for x in X]

        signals['Prediction'] = actions
        signals['Position'] = signals['Prediction'].apply(lambda x: 1 if x == 1 else 0)
        
        # Calculate Signal as the change in position
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        signals['return'] = np.log(self.data[(stock, 'Close')] / self.data[(stock, 'Close')].shift(1))
        
        self.signal_data[stock] = signals
