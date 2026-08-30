from abc import ABC,abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class TradingAgent(ABC):

    """
    Abstract base class for trading algorithms.

    This class provides a template for trading algorithms with common methods
    for plotting and return calculations. Specific trading strategies should be
    implemented in subclasses by overriding the abstract methods.

    Attributes:
        data (pd.DataFrame): The dataset containing stock prices or relevant trading data.
        signal_data (dict): A dictionary to store signal data for each stock.
        returns_data (dict): A dictionary to store returns data for each stock.
    """

    def __init__(self,data):


        """
        Initializes the TradingAlgorithm with provided data.

        Args:
            data (pd.DataFrame): The dataset containing stock prices or relevant trading data.
        """
        self.algorithm_name = "BaseAlgorithm"
        self.data = data
        self.signal_data = {}
        self.returns_data = {}
        self.score_column = None
        self.training_info = {}

    def _trained_model_for_stock(self, stock):
        for attr in ("models", "hmm_models"):
            container = getattr(self, attr, None)
            if isinstance(container, dict) and stock in container:
                return container[stock]
        return None

    def _training_parts(self, stock):
        train_data = getattr(self, "train_data", {})
        if stock not in train_data:
            return {}

        data = train_data[stock]
        if isinstance(data, tuple) and len(data) >= 4:
            x_train, x_test, y_train, y_test = data[:4]
            return {
                "x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": y_test,
                "index_train": getattr(x_train, "index", None),
                "index_test": getattr(x_test, "index", None),
                "raw": data,
            }

        if isinstance(data, dict):
            x_train = data.get("X_train", data.get("df_train"))
            x_test = data.get("X_test", data.get("df_test"))
            y_train = data.get("y_train", data.get("rewards_train"))
            y_test = data.get("y_test", data.get("rewards_test"))
            return {
                "x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": y_test,
                "index_train": data.get("index_train", getattr(x_train, "index", None)),
                "index_test": data.get("index_test", getattr(x_test, "index", None)),
                "raw": data,
            }

        return {"raw": data}

    def _length_or_none(self, value):
        try:
            return len(value)
        except TypeError:
            return None

    def _shape_or_none(self, value):
        shape = getattr(value, "shape", None)
        if shape is not None:
            return tuple(shape)
        try:
            return tuple(np.asarray(value).shape)
        except Exception:
            return None

    def _feature_metadata(self, x_train, raw_train_data):
        feature_count = None
        input_shape = None

        if x_train is not None:
            shape = self._shape_or_none(x_train)
            if shape:
                if len(shape) >= 2:
                    feature_count = int(shape[-1])
                if len(shape) > 2:
                    input_shape = shape[1:]
            columns = getattr(x_train, "columns", None)
            if columns is not None:
                feature_count = len(columns)
                input_shape = (len(columns),)

        if (feature_count is None or feature_count == 0) and isinstance(raw_train_data, dict):
            feature_cols = raw_train_data.get("feature_cols")
            if feature_cols is not None:
                feature_count = len(feature_cols)
                input_shape = (len(feature_cols),)

        return feature_count, input_shape

    def _index_bounds(self, index_like):
        if index_like is None:
            return None, None
        try:
            if len(index_like) == 0:
                return None, None
            return index_like[0], index_like[-1]
        except Exception:
            return None, None

    def _training_row(self, stock):
        parts = self._training_parts(stock)
        raw_train_data = parts.get("raw")
        x_train = parts.get("x_train")
        x_test = parts.get("x_test")
        train_start, train_end = self._index_bounds(parts.get("index_train"))
        test_start, test_end = self._index_bounds(parts.get("index_test"))
        feature_count, input_shape = self._feature_metadata(x_train, raw_train_data)

        row = {
            "Stock": stock,
            "Agent": self.algorithm_name,
            "ModelClass": None,
            "TrainSamples": self._length_or_none(x_train),
            "TestSamples": self._length_or_none(x_test),
            "FeatureCount": feature_count,
            "InputShape": input_shape,
            "TrainStart": train_start,
            "TrainEnd": train_end,
            "TestStart": test_start,
            "TestEnd": test_end,
        }

        model = self._trained_model_for_stock(stock)
        if model is not None:
            row["ModelClass"] = type(model).__name__

        for attr_name, column_name in (
            ("split_ratio", "SplitRatio"),
            ("epochs", "EpochsRequested"),
            ("episodes", "Episodes"),
            ("n_states", "NumStates"),
        ):
            if hasattr(self, attr_name):
                row[column_name] = getattr(self, attr_name)

        if isinstance(raw_train_data, dict):
            if "percentile" in raw_train_data:
                row["Percentile"] = raw_train_data["percentile"]
            if "epsilon_final" in raw_train_data:
                row["EpsilonFinal"] = raw_train_data["epsilon_final"]

        thresholds = getattr(self, "thresholds", None)
        if isinstance(thresholds, dict) and stock in thresholds:
            row["Threshold"] = thresholds[stock]

        best_regimes = getattr(self, "best_regimes", None)
        if isinstance(best_regimes, dict) and stock in best_regimes:
            row["BestRegime"] = best_regimes[stock]

        training_info = getattr(self, "training_info", {})
        if isinstance(training_info, dict):
            row.update(training_info.get(stock, {}))

        return row

    def training_summary(self, stock=None):
        train_data = getattr(self, "train_data", {})
        training_info = getattr(self, "training_info", {})

        if stock is not None:
            stocks = [stock]
        else:
            stocks = sorted(set(train_data.keys()) | set(training_info.keys()))

        if not stocks:
            raise ValueError("No training information is available yet.")

        rows = [self._training_row(stock_name) for stock_name in stocks]
        summary = pd.DataFrame(rows)

        preferred_columns = [
            "Stock",
            "Agent",
            "ModelClass",
            "TrainSamples",
            "TestSamples",
            "FeatureCount",
            "InputShape",
            "TrainStart",
            "TrainEnd",
            "TestStart",
            "TestEnd",
            "SplitRatio",
            "EpochsRequested",
            "EpochsRan",
            "Accuracy",
            "Precision",
            "Recall",
            "F1Score",
            "Loss",
            "ValLoss",
            "Threshold",
            "Percentile",
            "BestRegime",
            "NumStates",
            "Episodes",
            "EpsilonFinal",
        ]
        ordered_columns = [col for col in preferred_columns if col in summary.columns]
        remaining_columns = [col for col in summary.columns if col not in ordered_columns]
        return summary[ordered_columns + remaining_columns]

    def calculate_returns(self):
        """
        Calculates and stores the returns for each stock based on the trading signals.
        """

        self.returns_data = {}
        algorithm_name = self.algorithm_name

        for stock,signals in self.signal_data.items():
            strategy_return,buy_and_hold_return,total_entries = self._strategy_return_data(signals)
            self.returns_data[stock] = {
                f'{self.algorithm_name}_return': strategy_return,
                'buy_and_hold_return': buy_and_hold_return,
                'total_entries': total_entries
            }


    def _strategy_return_data(self,signals):
        """
        Calculate strategy return data for the given signals.

        Args:
            signals (pd.DataFrame): DataFrame containing trading signals and returns.

        Returns:
            tuple: A tuple containing the strategy return, buy and hold return, and total entries.
        """

        algorithm_name = self.algorithm_name
        signals = signals.dropna(subset=['return', 'Position']).copy()
        signals['agent_returns'] = signals['return']*signals['Position'].shift(1)
        strategy_return = round(signals['agent_returns'].sum()*100,3)
        buy_and_hold_return = round(signals['return'].sum()*100,3)
        total_entries = int(signals['Position'].diff().fillna(0).abs().sum() / 2)
        return strategy_return,buy_and_hold_return,total_entries

    @abstractmethod
    def generate_signal_strategy(self,stock,*args):
        """
        Abstract method to generate trading signals for a given stock.

        This method should be implemented by subclasses to define the specific
        trading strategy for generating signals.

        Args:
            stock (str): The stock symbol for which to generate signals.
        """
        pass

    def evaluate_performance(self):
        rows = []
        for stock,df in self.signal_data.items():
            df = df.dropna(subset = ['return'])

            if len(df) < 2:
                continue

            strategy_return = (df['return'] * df['Position'].shift(1)).sum()

            close_series = self.data[(stock,'Close')]
            signal_index = df.index[[0, -1]]
            buyhold_return = np.log(
                close_series.loc[signal_index[-1]] / close_series.loc[signal_index[0]]
            )

            cumulative_returns = (df['return'] * df['Position'].shift(1)).cumsum()
            running_max = cumulative_returns.cummax()
            drawdown = cumulative_returns - running_max
            max_drawdown = drawdown.min()

            outperformance = strategy_return-buyhold_return

            rows.append({
                'Stock': stock,
                'Strategy_Return_%': round(strategy_return*100,2),
                'BuyHold_Return_%': round(buyhold_return*100,2),
                'Max_Drawdown_%': round(max_drawdown*100,2),
                'Outperformance_%': round(outperformance * 100,2),
                'Profitable': strategy_return > 0.0,
               # 'Outperformed': outperformance > 0.0
            })

        res = pd.DataFrame(rows)
        if res.empty:
            return res

        res = res.sort_values(
            by=['Profitable', 'Outperformance_%', 'Strategy_Return_%'],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        return res


    def latest_row(self, stock: str) -> pd.Series:
        if stock not in self.signal_data:
            raise KeyError(f"No signal data for stock={stock}")
        df = self.signal_data[stock]
        # Drop rows where return is nan (early part)
        df = df.dropna(subset=["return"]) if "return" in df.columns else df.dropna()
        if df.empty:
            raise ValueError(f"Signal data for {stock} is empty after dropping NaNs.")
        return df.iloc[-1]

    def action_now(self, stock: str) -> dict:
        """
        Interprets the most recent signal as an action for the *next* period.
        """
        row = self.latest_row(stock)
        ts = row.name

        sig = int(row.get("Signal", 0)) if pd.notna(row.get("Signal")) else 0
        pos = int(row.get("Position", 0)) if pd.notna(row.get("Position")) else 0



        if sig == 1:
            action = "BUY"
        elif sig == -1:
            action = "SELL"
        elif pos == 1:
            action = "HOLD (LONG)"
        elif pos == -1:
            action = "HOLD (SHORT)"
        else:
            action = "HOLD (FLAT)"
        return {
            "Stock": stock,
            "Timestamp": ts,
            "Action": action,
            "Position": pos,
            "Signal": sig,

        }

    def score_now(self, stock: str) -> float:
        """
        Generic 'strength' score used to rank recommendations right now.

        Default behavior:
          1) If agent sets self.score_column and it exists -> use it
          2) Else if a 'Score' column exists -> use it
          3) Else fall back to 0.0 (or historical return if you prefer)
        """
        row = self.latest_row(stock)

        if self.score_column and self.score_column in row.index:
            val = row.get(self.score_column)
            return float(val) if pd.notna(val) else float("nan")

        for col in ("Score", "Strength", "Edge", "Alpha"):
            if col in row.index:
                val = row.get(col)
                return float(val) if pd.notna(val) else float("nan")

        return float("nan")


    def plot(self, stock):
        """
        Plots the stock price along with buy and sell signals.

        Args:
            stock (str): The stock symbol to plot.
        """

        fig,ax = plt.subplots(figsize = (10,5))

        plot_index = (stock,'Close')

        print(stock)

        # Original Stock price plot
        ax.plot(self.data.index,self.data[plot_index],label = 'Price')

        # Buy prices

        # Buy and sell markers
        buy_signal = self.signal_data[stock][self.signal_data[stock]['Signal']==1]
        sell_signal = self.signal_data[stock][self.signal_data[stock]['Signal']==-1]

        #Plot the markers
        ax.scatter(buy_signal.index,self.data[plot_index][buy_signal.index],color = 'green',marker = '^',label = 'Buy')
        ax.scatter(sell_signal.index,self.data[plot_index][sell_signal.index],color = 'red',marker = 'v',label = 'Sell')

        plt.title(f'{stock} Price and Trading Signals' )
        plt.legend()
        #plt.show()
        # Filling areas
        signals = self.signal_data[stock]
        ax.fill_between(signals.index,self.data[(stock,'Close')].min(),self.data[(stock,'Close')].max(),
        where = signals['Position']==1,color='green',alpha=0.1,label = 'Hold Period')
        ax.fill_between(signals.index, self.data[(stock, 'Close')].min(), self.data[(stock, 'Close')].max(),
                          where=signals['Position'] == 0, color='red', alpha=0.1, label='Sell Period')
        return fig,ax


    def plot_returns(self):
        """
        Plots the returns of the trading strategy compared to buy-and-hold strategy.
        """
        stock_names = []
        strategy_returns = []
        buy_and_hold_returns = []
        algorithm_name = self.algorithm_name


        for stock,returns in self.returns_data.items():

            stock_names.append(stock)
            strategy_returns.append(returns[f'{algorithm_name}_return'])
            buy_and_hold_returns.append(returns['buy_and_hold_return'])

        num_stocks = len(stock_names)
        fig_width = min(20,num_stocks)

        fig,ax = plt.subplots(figsize = (fig_width,6))
        bar_width = 0.35
        index = np.arange(num_stocks)

        bar1 = ax.bar(index,strategy_returns,bar_width,label = f"{algorithm_name} Log-Return")
        bar2 = ax.bar(index+bar_width,buy_and_hold_returns,bar_width,label = 'Buy & Hold Log-Return')

        ax.set_xlabel("Stock")
        ax.set_ylabel("Returns")
        ax.set_title("Strategy vs. Buy & Hold")

        ax.set_xticks(index + bar_width/2)
        ax.set_xticklabels(stock_names,rotation = 90)
        ax.legend()

        for bar in bar1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(round(bar.get_height(), 2)),
                    ha='center', va='bottom')

        for bar in bar2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(round(bar.get_height(), 2)),
                    ha='center', va='bottom')

        plt.show()





