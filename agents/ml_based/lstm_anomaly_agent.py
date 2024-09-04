import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from agents.base_agents.trading_agent import TradingAgent


class LSTMAnomalyAgent(TradingAgent):
    def __init__(self,data):
        super.__init__(data)
        self.algorithm_name = 'LSTMAnomaly'
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock)
        
        self.calculate_returns()

    def create_classification_trading_condition(self,stock):
        data_copy = self.data[stock].copy()
        data_copy['Open-Close'] = self.data[stock]['Open']-self.data[stock]['Close']
        data_copy['High-Low'] = self.data[stock]['High']-self.data[stock]['Low']

        data_copy.ffill(inplace = True)

        X = data_copy[['Open_Close','High_Low','Volume']].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X
    
    def build_lstm_model(self,input_shape):

        model = Sequential()
        model.add(LSTM(50,input_shape=input_shape,return_sequences = True))
        model.add(Dense(input_shape[1]))
        model.compile(optimizer = 'adam',loss = 'mse')
        return model

    def train_lstm(self,X,time_steps):

        X_lstm = []
        for i in range(time_steps,len(X)):
            X_lstm.append(X[i-time_steps:i])
        X_lstm = np.array(X_lstm)

        model = self.build_lstm_model((time_steps,X.shape[1]))
        model.fit(X_lstm,X[time_steps:],epochs= 20, batch_size = 32, verbose = 0)
        return model,time_steps
    
    def generate_signal_strategy(self,stock):

        signals = pd.DataFrame(index = self.data.index)
        X = self.create_classification_trading_condition(stock)
        time_steps = 10
        lstm_model,time_steps = self.train_lstm(X,time_steps)

        X_lstm = []
        for i in range(time_steps,len(X)):
            X_lstm.append(X[i-time_steps:i])
        X_lstm = np.array(X_lstm)

        X_pred = lstm_model.predict(X_lstm)
        reconstruction_error = np.mean(np.power(X[time_steps:]-X_pred,2),axis = 1)

        # Identify anomalies
        threshold= np.percentile(reconstruction_error,95)
        signals['Anomaly'] = reconstruction_error > threshold

        signals['Position'] = signals['Anomaly'].apply(lambda x: -1 if x else 0)
        signals['Position'] = signals['Anomaly'].shift(1).ffill(0).fillna(0)

        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1),'Signal'] = 1
        signals.loc[signals['Position']< signals['Position'].shift(1),'Signal'] = -1

        signals['return'] = np.log(self.data[stock]['Close']/self.data[stock]['Close'].shift(1))

        self.signal_data[stock] = signals


