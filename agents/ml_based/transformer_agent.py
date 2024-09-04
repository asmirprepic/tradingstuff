import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from agents.base_agents.trading_agent import TradingAgent

class TransformerAgent(TradingAgent):
    """
    A trading agent that uses a Transformer neural network to generate trading signals.
    This agent employs a neural network model based on price movement features to predict
    the direction of stock price movement.

    Attributes:
        algorithm_name (str): Name of the algorithm, set to "Transformer".
        stocks_in_data (pd.Index): Unique stock symbols present in the data.
        signal_data (dict): A dictionary to store signal data for each stock.
    """

    def __init__(self, data):
        super().__init__(data)
        self.algorithm_name = 'Transformer'
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock)
        self.calculate_returns()

    def create_classification_trading_condition(self, stock):
        """
        Creates the feature set for the Transformer model. The features include 
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

    def create_train_split_group(self, X, Y, split_ratio):
        """
        Splits the dataset into training and testing sets.

        Args:
            X (np.ndarray): The feature set.
            Y (np.ndarray): The target variable.
            split_ratio (float): The proportion of the dataset to include in the train split.

        Returns:
            tuple: The training and testing sets (X_train, X_test, Y_train, Y_test).
        """
        split_index = int(len(X) * split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        Y_train, Y_test = Y[:split_index], Y[split_index:]

        return X_train, X_test, Y_train, Y_test

    def prepare_transformer_data(self, X, time_steps):
        """
        Prepares the data for Transformer input by creating sequences of the specified time steps.

        Args:
            X (np.ndarray): The feature set.
            time_steps (int): The number of time steps to include in each sequence.

        Returns:
            np.ndarray: The reshaped feature set suitable for Transformer input.
        """
        transformer_data = []
        for i in range(time_steps, len(X)):
            transformer_data.append(X[i-time_steps:i])
        return np.array(transformer_data)

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """
        Transformer encoder block.

        Args:
            inputs (tf.Tensor): Input tensor.
            head_size (int): Size of the attention heads.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward layer dimension.
            dropout (float): Dropout rate.

        Returns:
            tf.Tensor: Output tensor.
        """
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_transformer_model(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
        """
        Builds the Transformer model.

        Args:
            input_shape (tuple): Shape of the input data.
            head_size (int): Size of the attention heads.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward layer dimension.
            num_transformer_blocks (int): Number of Transformer blocks.
            mlp_units (list): List of units in the MLP.
            dropout (float): Dropout rate.
            mlp_dropout (float): Dropout rate for MLP.

        Returns:
            tf.keras.Model: The Transformer model.
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs, outputs)

    def train_transformer_model(self, stock):
        """
        Trains the Transformer model for the specified stock.

        Args:
            stock (str): The stock symbol for which to train the model.

        Returns:
            tf.keras.Model: The trained Transformer model.
        """
        time_steps = 10
        X, Y = self.create_classification_trading_condition(stock)
        X_train, X_test, Y_train, Y_test = self.create_train_split_group(X, Y, split_ratio=0.8)
        
        X_train = self.prepare_transformer_data(X_train, time_steps)
        X_test = self.prepare_transformer_data(X_test, time_steps)
        Y_train = Y[time_steps:len(X_train) + time_steps]
        Y_test = Y[len(X_train) + time_steps:]

        input_shape = X_train.shape[1:]
        model = self.build_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0)

        return model

    def generate_signal_strategy(self, stock):
        """
        Generates trading signals for the specified stock using the trained Transformer model.
        A signal is generated for each day based on the predicted direction of the stock price.

        Args:
            stock (str): The stock symbol for which to generate signals.

        The method updates the `signal_data` attribute with signals for the given stock.
        """
        signals = pd.DataFrame(index=self.data.index)
        time_steps = 10
        X, _ = self.create_classification_trading_condition(stock)
        X = self.prepare_transformer_data(X, time_steps)
        transformer_model = self.train_transformer_model(stock)
        prediction = (transformer_model.predict(X) > 0.5).astype(int).flatten()
        
        signals = signals.loc[self.data.index[time_steps:len(prediction) + time_steps]]
        signals['Prediction'] = prediction
        signals['Position'] = signals['Prediction'].apply(lambda x: 1 if x == 1 else 0)
        
        # Calculate Signal as the change in position
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        signals['return'] = np.log(self.data[(stock, 'Close')] / self.data[(stock, 'Close')].shift(1))
        
        self.signal_data[stock] = signals
