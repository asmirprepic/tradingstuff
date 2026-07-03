from agents.base_agents.trading_agent import TradingAgent
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Attention, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class LSTMAttentionAgent(TradingAgent):
    """
    A trading agent that uses LSTM with an attention mechanism to generate trading signals. 
    This agent employs a neural network model based on price movement features to predict the
    direction of the stock price movement. 
    """

    def __init__(self,data):
        super().__init__(data)
        self.algorithm_name = 'LSTM Attention'
        self.stocks_in_data = self.data.columns.get_level(0).unique()

        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock)
        
        self.calculate_returns()


    def create_classification_trading_condition(self,stock):
        """
        Creates a feature for the LSTM model. The features include
        'Open-Close', 'High-Low', Volume and 'Close'
        
        Args: 
            stock(str): The stock symbol for which to create the features

        Returns: 
            np.ndarray: The feature set
            np.ndarray: The target variable, where 1 indicates an upward price movement and -1 indicates a downward price movement
        """

        data_copy = self.data[stock].copy()
        data_copy['Open-Close'] = self.data[stock]['Open'] - self.data[stock]['Close']
        data_copy['High-Low'] = self.data[stock]['High'] - self.data[stock]['Low']
        data_copy['Close'] = self.data[stock]['Close']
        data_copy['Volume'] = self.data[stock]['Volume']

        # Forward fill missing values. This is done to just use the last available price
        data_copy = data_copy.ffill()

        # Standardize the data
        scaler = MinMaxScaler()
        X = scaler.fit_transform(data_copy[['Open-Close','High-Low','Close','Volume']].values)
        Y = np.where(self.data[stock]['Close']>self.data['Close'].shift(1),1,-1)
        Y_series = pd.Series(Y,index = self.data[stock].index)
        Y = Y_series.loc[data_copy.index].values

        return X, Y
    
    def prepare_lstm_data(self,X,Y,time_steps):
        """
        Prepares the LSTM data by creating sequences of the specified time steps

        Args: 
            X (np.ndarray): The feature set
            time_steps (int): Thu number of time steps to include in each sequence

        Returns: 
           np.ndarray: The reshaped feature set suitable for LSTM input.
           np.ndarray: The target variable aligned with the sequences.
        """

        X_lstm = []
        Y_lstm = []
        for i in range(time_steps,len(X)):
            X_lstm.append(X[i-time_steps:time_steps])
            Y_lstm.append(Y[i])

        return np.array(X_lstm),np.array(Y_lstm)
    
    def LSTM_attention_model(self,input_shape):
        """
        Builds and returns the LSTM model with attention mechanism

        Args: 
            input_shape (tuple): The shape of the input data

        Returns: 
            Model: the LSTM model
        
        """

        inputs = Input(shape = input_shape)
        lstm_out = LSTM(64,return_sequences = True)(inputs)

        # Attention mechanism
        attention = Dense(1, activation = 'tanh')(lstm_out)
        attention = tf.nn.softmax(attention,axis = 1)
        context_vector = tf.reduce_sum(attention*lstm_out,axis = 1)

        dense_out = Dense(50,activation = 'relu')(context_vector)
        outputs = Dense(1, activation = 'sigmoid')(dense_out)

        model = Model(inputs,outputs)
        model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
        return model
    
    def generate_signal_strategy(self,stock):
        """
        Generate trading signals for the trained LSTM with attention model. 
        A signal is generated for each time period based on the predicted direction of the stock price
        
        Args:
            stock (string): The symbol of the stock to generate the signal to
        
        The method updates the signal_data attribute with signal df for the stock.
        """

        signals = pd.DataFrame(self.data.index)
        time_steps = 10
        X,Y = self.create_classification_trading_condition(stock)
        X_lstm,Y_lstm = self.prepare_lstm_data(X,time_steps)
        
        input_shape = (X_lstm.shape[1],X_lstm.shape[2])
        lstm_attention_model = self.LSTM_attention_model(input_shape)

        # Train the LSTM model
        lstm_attention_model.fit(X_lstm,Y_lstm,epochs = 20, batch_size = 32, verbose = 0)

        # Make predictions
        predictions = (lstm_attention_model.predict(X_lstm)>0.5).astype(int).flatten()

        # Generate signals based on the trading condition
        signals = signals.iloc[time_steps:]
        signals['Prediction'] = predictions
        signals['Position'] = signals['Prediction'].apply(lambda x: 1 if x == 1 else 0)

        # Calculate signals as change in position
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1),'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1),'Signal'] = -1

        signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))

        self.signal_data[stock] = signals
