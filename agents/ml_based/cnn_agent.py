from agents.base_agents.trading_agent import TradingAgent
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Flatten,Dense
from sklearn.preprocessing import MinMaxScaler


class CNNAgent(TradingAgent):
  """
  A trading agent that uses convolutional neural network to generate trading signals. 
  This agent employs a neural network model based on price movement features to predict
  the direction of the stock price movement. 

  Attributes: 
    algorithm_name(string): Name of the algorithm implementet, set to "CNN"
    stocks_in_data (pd.Index): Unique stock symbols present in the data
    signal_data (dict): A dictionary to store signal data for each stock
  """

  def __init__(self,data):
    super.__init__(data)
    self.algorithm_name = 'CNN'
    self.stocks_in_data = self.data.columns.get_level_values(0).unique()

    for stock in self.stocks_in_data:
      self.generate_signal_strategy(stock)

    self.calculate_returns()

  def create_classification_trading_condition(self,stock):
    """
    Creates a feature set for the CNN model. The features include
    'Open-Close' and 'High-low'

    Args: 
      stock(str): The stock symbol for which to create features. 

    Returns: 
      pd.DataFrame: the feature set
      pd.Series: The target variable, where 1 indicates an upward price movement and -1 indicates a downward price movement
    """

    data_copy = self.data[stock].copy()
    data_copy['Open-Close'] = self.data[stock]['Open'] - self.data[stock]['Close']
    data_copy['High-low'] = self.data[stock]['High'] - self.data[stock]['Low']
    data_copy.dropna(inplace = True)

    X = data_copy[['Open-Close','High-Low']].values
    Y = np.where(self.data[stock]['Close'].shift(-1) > self.data[stock]['Close'],1,-1)
    Y_series = pd.Series(Y, index = self.data[stock].index]
    Y = Y_series.loc[data_copy.index].values

    return X,Y

    def create_train_split_group(self,X,Y,split_ratio):
      """
      Splits the dataset into test and train set

      Args: 
        X (np.ndarray): The feature set
        Y (np.ndarray): The target variable
        split_ratio (float): The proportion of the dataset to in include in the train test split. 
      """
      split_index = int(len(X)*split_ratio)
      X_train,X_test = X[:split_index], X[split_index:]
      Y_train,Y_test = Y[:split_index],Y[split_index:]

      return X_train,X_test,Y_train,Y_test

    def prepare_cnn_data(self,X,time_steps):
      """
      Prepares the data for CNN input by creating sequences of the specified time steps. 

      Args: 
        X (np.ndarray): The feature set. 
        time_steps (int): The number of time steps to include in each sequence. 

      Returns:
        np.ndarray: The reshaped feature set suitable for CNN input.
      """
      cnn_data =[]
      for i in range(time_steps,len(X)):
        cnn_data.append(X[i-time_steps:i])
      return np.array(cnn_data)

    def CNN_model(self,stock):
      """
      Trains the CNN model for the specified stock. 

      Args: 
        stock (str): The stock symbol for which to train the model

      Returns: 
        Sequential: The trained nn
      """
      time_teps = 10
      X, Y = self.create_classification_trading_condition(stock)
      X_train,X_test, Y_train,Y_test = self.create_train_test_split_group(X,Y,split_ratio= 0.8)

      X_train = self.prepare_cnn_data(X_train,time_steps)
      X_test = self.prepare_cnn_data(X_test,time_steps)
      Y_train = Y[time_steps:len(X_train) + time_steps]
      Y_test = Y[len(X_train) + time_stemps:]

      model = Sequential()
      model.add(Conv1D(filters = 64, kernel_size = 3, activation = 'relu', input_shape=(X_train.shape[1], X_train.shape[2])))
      model.add(MaxPooling1D(pool_size = 2))
      model.add(Flatten())
      model.add(Dense(50,activation = 'relu'))
      model.add(Dense(1,activation = 'sigmoid'))

      model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
      model.fit(X_train,Y_train,epochs = 20, batch_size = 32, verbose = 0)

      return model

      def generate_signal_strategy(self,stock):
        """  
        Generates trading signals for the specified stock using trained CNN model. 
        A signal is generated for each time period based on the predicted direction of the stock price. 

        Args: 
          stock (str): The stock symbol for which to generate signals

        The method updates the signal_data attribute with signals for the given stock. 

        """

        signals = pd.DataFrame(index = self.data.index)
        time_steps = 10
        X,_ = self.create_classifcation_trading_condition(stock)
        X = self.prepare_cnn_data(X,time_steps)
        cnn_model = self.CNN_model(stock)
        prediction = (self.cnn_model.predict(X) > 0.5).as_type(int).flatten()

        signals= signals.loc[self.data.index[time_steps:len(prediction) + time_steps]]
        signals['Prediction'] = prediction
        signals['Position']= signals['Prediction'].apply(lambda x: 1 if x == 1 else 0)

        # Calculate Signal as change in position
        signals['Signal'] =0 
        signals.loc[signals['Position'] > signals['Position'].shift(1),'Signal'] = 1
        signals.loc[signals['Position'] < signals['Position'].shift(1),'Signal'] = -1
        
        signals['return'] = np.log(self.data[(stock,'Close')]/self.data[(stock,'Close')].shift(1))

        self.signal_data[stock] = signals

        
                
      

      
