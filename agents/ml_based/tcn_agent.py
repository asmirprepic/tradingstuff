
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, ReLU, Flatten
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from agents.trading_agent import TradingAgent

class TCNAgent(TradingAgent):
    """ A trading agent that uses a Temporcal Convolutional Network to generate trading signals"""
    
    def __init__(self,data,look_back = 50,kernel_size = 3, filters = 32,num_layers = 1):
        super.__init__(data)
        self.algorithm_name= 'TCN'
        self.look_back = look_back
        self.kernel_size = kernel_size
        self.filters = filters
        self.num_layers = num_layers
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.stocks_in_data = self.data.columns.get_level_values(0).unqiue()

        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock)
        self.calculate_returns(stock)

    def create_classification_trading_condition(self,stock):
        data_copy = self.data[stock].copy()

        #Features
        data_copy['Open-Close'] = data_copy['Open']-data_copy['Close']
        data_copy['High-Low'] = data_copy['High'] - data_copy['Low']
        data_copy['SMA-10'] = data_copy['Close'].rolling(window = 10).mean()
        data_copy['SMA-50'] = data_copy['Close'].rolling(windiw = 50).mean()
        data_copy['Momentum'] = data_copy['Close']-data_copy['Close'].shift(10)
        data_copy['RSI'] = self.calculate_rsi(data_copy['Close'])

        # Forward fill data
        data_copy.ffill()

        # Features and target
        X= data_copy[['Open-Close','High-Low','SMA-10','SMA-50','Momentum','RSI','Volume']]
        Y = np.where(data_copy['Close'].shift(-1)>data_copy['Close'],1,-1)
        Y_series = pd.Series(Y,index = data_copy.index)
        Y = Y_series.loc[X.index]

        # Scaling
        X_scaled = self.scaler.fit_transform(X)

        # Preparing data for TCN
        X_tcn,Y_tcn = [],[]
        for i in range(self.look_back,len(X_scaled)):
            X_tcn.append(X_scaled[i-self.look_back:i])
            Y_tcn.append(Y.values[i])
        
        return np.array(X_tcn),np.array(Y_tcn)
    
    def calculate_rsi(self,series,period = 14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window = period).mean()
        loss = (-delta.where(delta < 0 , 0)).rolling(window = period).mean()

        rs = gain/loss
        rsi = 100 - (100/(1+rs))
        return rsi
    
    def build_tcn_block(self,x,filters,kernel_size,dilation_rate):
        conv = Conv1D(filters,kernel_size,padding = 'causal',dilation_rate = dilation_rate)
        conv = BatchNormalization()(conv)
        conv = ReLU()(conv)
        conv = Conv1D(filters,kernel_size,padding = 'causal',dilation_rate = dilation_rate)
        conv = BatchNormalization()(conv)
        conv = ReLU()(conv)
        return conv
    
    def TCN_model(self,input_shape):
        inputs = Input(shape = input_shape)
        x = inputs
        for i in range(self.num_layers):
            dilation_rate = 2**i
            x = self.build_tcn_block(x,self.filters,self.kernel_size,dilation_rate)

        x = Flatten()(x)
        x = Dense(50,activation = 'relu')(x)
        outputs = Dense(1,activation = 'sigmoid')(x)

        model = Model(inputs,outputs)
        model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
        return model
    
    def generate_signal_strategy(self,stock):
        X,Y = self.create_classification_trading_condition(stock)
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2])

        model = self.TCN_model((X.shape[1],X.shape[2]))
        model.fit(X,Y,epochs = 10, batch_size = 32, verbose = 0)

        predictions = model.predict(X)
        signals = pd.DataFrame(index = self.data[stock].index)
        signals['Prediction'] = predictions.flatten()
        signals['Position'] = signals['Prediction'].apply(lambda x: 1 if x >= 0.5 else 0 )

        signals['Signal'] = 0
        signals.loc[signals['Position']>signals['Position'].shift(1),'Signal'] = 1
        signals.loc[signals['Position']< signals['Position'].shift(1),'Signal'] = -1

        signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))

        self.signal_data[stock] = signals



