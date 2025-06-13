from agents.base_agents.trading_agent import TradingAgent
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np
import pandas as pd


class AutoencoderAgent(TradingAgent):
    def __init__(self, data):
        super().__init__(data)
        self.algorithm_name = 'Autoencoder'
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        self.models = {}
        self.scalers = {}
        self.signal_data = {}

    def create_classification_trading_condition(self, stock):
        data_copy = self.data[stock].copy()
        data_copy['Open-Close'] = self.data[stock]['Open'] - self.data[stock]['Close']
        data_copy['High-Low'] = self.data[stock]['High'] - self.data[stock]['Low']
        data_copy.dropna(inplace=True)

        X = data_copy[['Open-Close', 'High-Low']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.scalers[stock] = scaler

        return X_scaled,data_copy

    def Autoencoder_model(self, stock):
        X_scaled,_ = self.create_classification_trading_condition(stock)
        input_dim = X_scaled.shape[1]
        encoding_dim = 2

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="tanh")(input_layer)
        decoder = Dense(input_dim, activation="sigmoid")(encoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)

        return autoencoder

    def generate_signal_strategy(self, stock,anomaly_threshold_percentile = 95):
        
        if stock not in self.models:
            model = self.Autoencoder_model(stock)
            self.models[stock] = model
        else:
            model = self.models[stock]

        X_scaled, data_copy = self.create_classification_trading_condition(stock)
        predictions = model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)

        threshold = np.percentile(mse, anomaly_threshold_percentile)

        signals = pd.DataFrame(index=data_copy.index)
        signals['Anomaly'] = mse > threshold
        signals['Position'] = signals['Anomaly'].apply(lambda x: 1 if x else 0)

        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        signals['return'] = np.log(self.data[(stock, 'Close')].loc[data_copy.index] /
                                   self.data[(stock, 'Close')].shift(1).loc[data_copy.index])
        
        self.signal_data[stock] = signals

    def generate_signals_for_all(self, anomaly_threshold_percentile=95):
        """
        Convenience method to generate signals for all stocks.
        """
        for stock in self.stocks_in_data:
            print(f"Generating signals for {stock} ...")
            self.generate_signal_strategy(stock, anomaly_threshold_percentile=anomaly_threshold_percentile)
