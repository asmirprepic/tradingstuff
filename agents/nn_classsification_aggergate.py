class NNClassificationAgg(TradingAgent):
    # ...

    def __init__(self, data, lags=5):
        super().__init__(data)
        self.algorithm_name = "NNClassificationAgg"
        self.lags = lags
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        
        # Aggregate data for training
        aggregated_X, aggregated_Y = self.aggregate_data_for_training(self.stocks_in_data, lags)
        
        # Train model on aggregated data
        self.model, self.res = self.NNmodelFit(aggregated_X, aggregated_Y, lags)
        
        # Generate signals for each stock
        for stock in self.stocks_in_data:
            self.generate_signal_strategy(stock, lags)

        self.calculate_returns()

    def aggregate_data_for_training(self, stocks, lags):
        all_X, all_Y = [], []
        for stock in stocks:
            X, Y, _ = self.create_classification_trading_condition(stock, lags)
            all_X.append(X)
            all_Y.append(Y)
        return pd.concat(all_X), pd.concat(all_Y)

    def NNmodelFit(self, X, Y, lags):
        # Prepare data
        X_train, X_test, Y_train, Y_test = self.create_train_split_group(X, Y, split_ratio=0.8)
        mu_train, std_train = X_train.mean(), X_train.std()
        X_train_ = (X_train - mu_train) / std_train
        X_test_ = (X_test - mu_train) / std_train

        # Define model
        optimizer = Adam(learning_rate=0.0001)
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(X_train_.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_, Y_train)

        res = pd.DataFrame(model.history.history)

        return model, res

    def generate_signal_strategy(self, stock, lags):
        # Generate signals using the trained model
        X, _, cols = self.create_classification_trading_condition(stock, lags)
        X_ = (X - X.mean()) / X.std()  # Normalize using mean and std of the feature set
        signal = self.model.predict(X_[cols])

        signals = pd.DataFrame(index=X.index)
        signals['Position'] = np.where(signal > 0.5, 1, -1)
        signals['Prediction'] = signal
        signals['return'] = np.log(self.data[(stock, 'Close')] / self.data[(stock, 'Close')].shift(1))

        # Calculate Signal as the change in position
        signals['Signal'] = signals['Position'].diff().fillna(0)
        self.signal_data[stock] = signals
