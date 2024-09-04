import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import mse
from agents.base_agents.trading_agent import TradingAgent


class VAEAgent(TradingAgent):
    def __init__(self,data):
        super.__init__(data)
        self.algorithm_name = 'VAE'
        self.stocks_in_data = self.data.columns.get_level(0).unique()

        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock)
        
        self.calculate_returns()

    def create_classification_trading_condition(self,stock):
        """
        Creates features for the VAE model. The Open-Close, High-Low differences are used
        as features and standardize
        """

        data_copy = self.data[stock].copy()
        data_copy['Open-Close'] = self.data[stock]['Open'] - self.data[stock]['Close']
        data_copy['High-Low'] = self.data[stock]['High'] - self.data[stock]['Low']
        data_copy['Close'] = self.data[stock]['Close']
        data_copy = data_copy.ffill()

        X = data_copy[['Open-Close','High-Low','Close']].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X
    
    def sampling(self,args):
        """Sampling function to draw samples from the learned latent space"""

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[0]

        epsilon = K.random_normal(shape = (batch,dim))

        return z_mean + K.exp(0.5*z_log_var)*epsilon
    
    def build_vae_model(self,input_dim,latent_dim):
        """Build and compile the VAE model"""
        inputs = Input(shape=(input_dim,))
        h = Dense(32,activation = 'relu')(inputs)

        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)
        z = Lambda(self.sampling,output_shape =(latent_dim,))([z_mean,z_log_var])

        encoder = Model(inputs,z_mean)

        # Decoder
        decoder_h = Dense(32,activation = 'relu')
        decoder_mean = Dense(input_dim,activation = 'sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        
        vae = Model(inputs,x_decoded_mean)

        def vae_loss(inputs,x_decoded_mean):
            """VAE loss = Reconstruction loss + KL divergence loss"""
            reconstruction_loss = mse(inputs,x_decoded_mean) * input_dim
            kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),axis = -1)
            return K.mean(reconstruction_loss + kl_loss)
        
        vae.compile(optimizer = 'adam',loss = vae_loss)

        return vae,encoder
    
    def generate_signal_strategy(self, stock):
        """Generates trading signals using VAE anomaly detection"""

        signals = pd.DataFrame(index = self.data.index)
        X = self.create_classification_trading_condition(stock)
        input_dim = X.shape[1]
        latent_dim = 2

        vae,encoder = self.build_vae_model(input_dim,latent_dim)
        vae.fit(X,X,epochs = 50,batch_size = 32, shuffle = True,verbose = 0)

        # Get reconstruction errors
        X_pred = vae.predict(X)
        mse = np.mean(np.power(X-X_pred,2),axis = 1)

        # Define anomaly threshold
        threshold = np.percentile(mse,95)
        signals['Anomaly'] = mse > threshold

        # Generate trading signals based on anomalies
        signals['Position'] = signals['Anomaly'].apply(lambda x: -1 if x else 0)
        signals['Position'] = signals['Position'].shift(1).fillna(0)

        signals['Signal'] = 0
        signals.loc[signals['Position']>signals['Position'].shift(1),'Signal'] = 1
        signals.loc[signals['Position']< signals['Position'].shift(1),'Signal'] = -1

        signals['return'] = np.log(self.data[stock]['Close']/self.data[stock]['Close'].shift(1))

        self.signal_data[stock] = signals




