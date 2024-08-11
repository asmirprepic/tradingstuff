from agents.trading_agent import TradingAgent
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np



class AutoencoderAgent(TradingAgent):
    def __init__(self, data):
        super().__init__(data)
        self.algorithm_name = 'Autoencoder'
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()

        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock)
        self.calculate_returns()

    def create_classification_trading_condition(self, stock):
        data_copy = self.data[stock].copy()
        data_copy['Open-Close'] = self.data[stock]['Open'] - self.data[stock]['Close']
        data_copy['High-Low'] = self.data[stock]['High'] - self.data[stock]['Low']
        data_copy.dropna(inplace=True)

        X = data_copy[['Open-Close', 'High-Low']].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X

    def Autoencoder_model(self, stock):
        X = self.create_classification_trading_condition(stock)
        input_dim = X.shape[1]
        encoding_dim = 2

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="tanh")(input_layer)
        decoder = Dense(input_dim, activation="sigmoid")(encoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, verbose=0)

        return autoencoder

    def generate_signal_strategy(self, stock):
        signals = pd.DataFrame(index=self.data.index)
        X = self.create_classification_trading_condition(stock)
        model = self.Autoencoder_model(stock)
        predictions = model.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=1)

        threshold = np.percentile(mse, 95)
        signals['Anomaly'] = mse > threshold
        signals['Position'] = signals['Anomaly'].apply(lambda x: 1 if x else 0)

        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        signals['return'] = np.log(self.data[(stock, 'Close')] / self.data[(stock, 'Close')].shift(1))
        
        self.signal_data[stock] = signals
