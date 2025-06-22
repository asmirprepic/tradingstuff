from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from agents.base_agents.nn_based_agent import NNBasedAgent
import pandas as pd
import numpy as np


class CNNAgent(NNBasedAgent):
    def __init__(self, data):
        super().__init__(data, epochs=25)
        self.algorithm_name = "CNN"

    def feature_engineering(self, stock):
        df = self.data[stock].copy()
        df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Return'].rolling(window=5).std()
        df = df.dropna()

        X = df[['Return', 'Volatility']]
        y = (df['Close'].shift(-1) > df['Close']).astype(int)
        X = X.dropna()
        y = y.loc[X.index]
        return X, y, ['Return', 'Volatility']

    def build_model(self, input_shape):
        model = Sequential([
            Conv1D(16, kernel_size=2, activation='relu', input_shape=(input_shape[0], 1)),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train_model(self, stock):
        X, y, feature_cols = self.feature_engineering(stock)
        X = X[feature_cols].values.reshape((X.shape[0], len(feature_cols), 1))
        X_train, X_test, y_train, y_test = self.create_train_split_group(X, y, shuffle=False, test_size=1 - self.split_ratio)

        model = self.build_model(input_shape=(len(feature_cols), 1))
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  verbose=self.verbose,
                  callbacks=[es])
        self.models[stock] = model
        self.train_data[stock] = (X_train, X_test, y_train, y_test)

    def predict_signals(self, stock, mode='backtest', threshold=0.5):
        model = self.models[stock]
        X_train, X_test, _, _ = self.train_data[stock]

        if mode == 'backtest':
            X_pred = X_test
        else:
            raise NotImplementedError("Live mode not implemented for CNN")

        probs = model.predict(X_pred, verbose=0).flatten()
        predictions = (probs > threshold).astype(int)

        index_used = pd.date_range(end=self.data[stock].index[-1], periods=len(predictions))
        signals = pd.DataFrame(index=index_used)
        signals['Prediction'] = predictions
        signals['Position'] = predictions
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        close = self.data[(stock, 'Close')].reindex(index_used)
        signals['return'] = np.log(close / close.shift(1))
        return signals
