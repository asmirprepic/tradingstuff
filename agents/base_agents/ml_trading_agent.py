from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from agents.base_agents.trading_agent import TradingAgent
from concurrent.futures import ThreadPoolExecutor, as_completed


class MLBasedAgent(TradingAgent, ABC):
    def __init__(self, data, model=None, features=None):
        super().__init__(data)
        self._base_model = model
        self.features = features
        self.algorithm_name = 'MLBaseAlgorithm'
        self.models = {}
        self.train_data = {}
        self.signal_data = {}
        self.stocks_in_data = self.data.columns.get_level_values(0).unique()
        self.feature_cache = {}

    def default_feature_engineering(self, stock, force_refresh=False,timing = 'open'):
        if stock in self.feature_cache and not force_refresh:
            return self.feature_cache[stock]

        df = self.data[stock].copy()
        required_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns for {stock}")

        r1     = df['Close'].pct_change()
        r5     = df['Close'].pct_change(5)
        ma5    = df['Close'].rolling(5).mean()
        ma10   = df['Close'].rolling(10).mean()
        vol5d  = r1.rolling(5).std()
        volchg = df['Volume'].pct_change()
        oc     = df['Open'] - df['Close']
        hl     = df['High'] - df['Low']

        if timing == 'open':
            df['Return_1D']     = r1.shift(1)
            df['Return_5D']     = r5.shift(1)
            df['MA_5']          = ma5.shift(1)
            df['MA_10']         = ma10.shift(1)
            df['Momentum']      = df['Close'].shift(1) - df['MA_5']
            df['Volatility_5D'] = vol5d.shift(1)
            df['Volume_Change'] = volchg.shift(1)
            df['OC']            = oc.shift(1)
            df['HL']            = hl.shift(1)
        else:  # close timing
            df['Return_1D']     = r1
            df['Return_5D']     = r5
            df['MA_5']          = ma5
            df['MA_10']         = ma10
            df['Momentum']      = df['Close'] - df['MA_5']
            df['Volatility_5D'] = vol5d
            df['Volume_Change'] = volchg
            df['OC']            = oc
            df['HL']            = hl

        df[self.features] = df[self.features].ffill()
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

        df = df.dropna(subset=['Target'])

        X = df[self.features].copy()
        Y = df['Target'].copy()
        self.feature_cache[stock] = (X, Y)
        return X,Y


    @abstractmethod
    def feature_engineering(self, stock):
        pass

    @abstractmethod
    def generate_signal_strategy(self, stock, *args, **kwargs):
        pass



    def create_train_split_group(self, X, Y, split_ratio):
        return train_test_split(X, Y, shuffle=False, test_size=1 - split_ratio)

    def train_model(self, stock, split_ratio=0.8):
        if self._base_model is None or self.features is None:
            raise ValueError("Model and features must be defined.")

        X, Y = self.feature_engineering(stock)
        X_train, X_test, Y_train, Y_test = self.create_train_split_group(X, Y, split_ratio)

        model = clone(self._base_model)
        model.fit(X_train, Y_train)
        self.models[stock] = model
        self.train_data[stock] = (X_train, X_test, Y_train, Y_test)

        Y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(Y_test, Y_pred),
            'precision': precision_score(Y_test, Y_pred, zero_division=0),
            'recall': recall_score(Y_test, Y_pred, zero_division=0),
            'f1_score': f1_score(Y_test, Y_pred, zero_division=0)
        }

        print(f"\nModel Performance for {stock} ({self.algorithm_name}):")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        return metrics

    def predict_signals(self, stock, mode='backtest', threshold=0.5):
        if stock not in self.models:
            raise ValueError(f"Model for {stock} has not been trained.")

        model = self.models[stock]
        X, _ = self.feature_engineering(stock)

        if mode == 'backtest':
            if stock not in self.train_data:
                raise ValueError(f"Training data for {stock} is missing.")
            X_pred = self.train_data[stock][1]
            index_used = X_pred.index
        elif mode == 'live':
            X_pred = X
            index_used = X.index
        else:
            raise ValueError("mode must be 'backtest' or 'live'")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_pred)[:, 1]
            predictions = np.where(prob > threshold, 1, -1)
        else:
            predictions = model.predict(X_pred)

        signals = pd.DataFrame(index=index_used)
        signals['Prediction'] = predictions
        signals['Position'] = (signals['Prediction'] == 1).astype(int)
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        close = self.data[(stock, 'Close')]
        signals['return'] = np.log(close / close.shift(1)).reindex(index_used)
        return signals

    def walk_forward_predict(self, stock, initial_train_size=100, step_size=1):
        """
        Perform walk-forward retraining and prediction for one stock.

        Args:
            stock (str): Stock symbol
            initial_train_size (int): Minimum number of days to start training
            step_size (int): How many days to move forward each iteration

        Returns:
            pd.DataFrame: Signals with predicted positions across full history
        """
        X, Y = self.feature_engineering(stock)
        predictions = []
        indices = []

        for start in range(initial_train_size, len(X) - step_size + 1, step_size):
            X_train = X.iloc[:start]
            Y_train = Y.iloc[:start]
            X_test = X.iloc[start:start + step_size]

            model = clone(self._base_model)
            model.fit(X_train, Y_train)



            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_test)[:, 1]
                pred = np.where(prob > 0.5, 1, -1)
            else:
                pred = model.predict(X_test)

            predictions.extend(pred)
            indices.extend(X_test.index)


        signals = pd.DataFrame(index=indices)
        signals['Prediction'] = predictions
        signals['Position'] = (signals['Prediction'] == 1).astype(int)
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1


        close = self.data[(stock, 'Close')].reindex(signals.index)
        signals['return'] = np.log(close / close.shift(1))

        return signals

    def _process_stock(self,stock,mode = 'backtest',walk_forward = True,**kwargs):
        try:
            if walk_forward:
                print(f"[WALK-FORWARD] {stock}")
                signals = self.walk_forward_predict(stock,**kwargs)
            else:
                self.train_model(stock)
                signals = self.predict_signals(stock,mode = mode)
            return stock,signals
        except Exception as e:
            print(f"[ERROR] {stock}: {e}")
            return stock,None

    def run_all(self, mode='backtest'):
        """
        Train and generate signals for all stocks in data.
        """
        for stock in self.stocks_in_data:
            self.train_model(stock)
            self.signal_data[stock] = self.predict_signals(stock, mode=mode)
        self.calculate_returns()

    def run_all_walk_forward(self, initial_train_size=100, step_size=1,max_workers = 10):
        """
        Apply walk-forward retraining and prediction for all stocks.

        Args:
            initial_train_size (int): Number of observations to start training
            step_size (int): How many steps to move forward each iteration
        """
        self.signal_data = {}

        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            futures = {
                executor.submit(self._process_stock,stock,walk_forward =True,
                                initial_train_size = initial_train_size,
                                step_size = step_size): stock
                for stock in self.stocks_in_data
            }

            for future in as_completed(futures):
                stock,signals = future.result()
                if signals is not None:
                    self.signal_data[stock] = signals
        self.calculate_returns()

    def _build_signals(self, predictions, index, stock):
        signals = pd.DataFrame(index=index)
        signals['Prediction'] = predictions
        signals['Position'] = (signals['Prediction'] == 1).astype(int)
        signals['Signal'] = 0
        signals.loc[signals['Position'] > signals['Position'].shift(1), 'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1), 'Signal'] = -1

        close = self.data[(stock, 'Close')].reindex(index)
        signals['return'] = np.log(close / close.shift(1))
        return signals
        # for stock in self.stocks_in_data:
        #     try:
        #         print(f"[WALK-FORWARD] {stock}")
        #         signals = self.walk_forward_predict(stock, initial_train_size, step_size)
        #         self.signal_data[stock] = signals
        #     except Exception as e:
        #         print(f"[WARNING] Walk-forward failed for {stock}: {e}")

        # self.calculate_returns()

