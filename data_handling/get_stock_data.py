import pandas as pd
import yfinance as yf
import datetime as dt

class GetStockData:
  """  
  A class to fetch and process stock data from Yahoo Finance. 

  Attributes: 
  ----------------
    stocks (list): A list of stock tickers to fetch data for
    start_date (pd.Timestamp): The start date for historical data
    end_date (pd.Timestamp): The end date for fetching historical data
    interval (str): The data interval for fetching historical data
    data_types (list): List of data types to fetch e.g ['Close']

  Methods: 
  ---------------
    getData(): Fetch and process stock data based on initialization parameters
    update_date(): Update stock data with new information if available

  """
  def __init__(self,stocks,startdate,enddate,interval = "60m", data_types = None):
    """  
    Initalizes the GetStockData instance with specified parameters.

    Parameters:
    -------------
      stocks (list or str): List of stock tickers or a single ticker
      startdate (str): Start date for fetching historial data
      endate (str): End date for fetching historical data
      interval (str): Data interval (default is "60m")
      data_types (list): Types of data to fetch

    """
    if data_types is None:
      data_types = ["Close"]

    self.stocks = stocks if isinstance(stocks,list) else [stocks]
    self.start_date=pd.to_datetime(startdate)
    self.end_date=pd.to_datetime(enddate)
    self.data_types = data_types if isinstance(data_types,list) else [data_types]
    self.interval = interval
    
    


  def getData(self):
    """   
    Fetch and process historical stock data for the given parameters
    """
    if self.start_date > self.end_date:
        raise ValueError("Start date must be earlier than end date.")

    all_data = []
    for stock in self.stocks:
        try:
            ticker = yf.Ticker(stock)
            prices = ticker.history(start=self.start_date, end=self.end_date, interval=self.interval)

            if prices.empty:
                print(f"No data for {stock}. Skipping...")
                continue

            prices = prices[self.data_types].copy()
            prices['Stock'] = stock
            all_data.append(prices)
            print(f"Successfully fetched data for {stock}")  # Log successful fetch
        except Exception as e:
            print(f"Error fetching data for {stock}]: {e}")  # Log any errors

    if not all_data:
        return pd.DataFrame()

    combined_data = pd.concat(all_data)
    combined_data.index = pd.to_datetime(combined_data.index)

    if self.interval.endswith('m') or self.interval.endswith('h'):
        combined_data = self._filter_trading_hours(combined_data)

    return self._process_yfinance_data(combined_data)

  def _process_yfinance_data(self, raw_data):
      """
      Process raw data from Yahoo Finance

      Parameters: 
      ------------
        raw_data (pd.DataFrame): The raw stock data fetched from Yahoo Finance

      Returns: 
      ------------
        pd.DataFrame: A DataFrame with processed stock data, where each stock data is in sepearate columns
      """
      if raw_data.empty:
          print("Warning: No data available for the specified data types and stocks.")
          return pd.DataFrame()  # Or handle this case differently based on your needs
      
      raw_data.reset_index(inplace=True)
      if "m" in self.interval or "h" in self.interval: 
        raw_data.set_index(['Datetime', 'Stock'], inplace=True)
      else: 
        raw_data.set_index(['Date','Stock'],inplace = True)
        
      raw_data = raw_data[self.data_types]
      processed_data = raw_data.unstack(level='Stock')
      processed_data.columns = processed_data.columns.swaplevel(0, 1)
      processed_data.sort_index(axis=1, level=0, inplace=True)
      self.data = processed_data
      

  def _filter_trading_hours(self,data):
    """
    Filter the data to include only trading hours. 

    Parameters: 
    -------------
      data (pd.DataFrame): The stock data to filter

    Returns: 
    ------------
      pd.DataFrame: The filtered data containng only trading hours
    """
    
    trading_start = dt.time(9,0)
    trading_end = dt.time(17,30)
    trading_days = [0,1,2,3,4]

    def is_trading_time(timestamp):
      return(timestamp.time() >= trading_start and timestamp.time() <= trading_end)
    
    filtered_data = data[data.index.map(is_trading_time)]

    return data.sort_index(axis=1)
  
  def update_data(self):
    """
    Update stock data with new information if available. 

    Returns:
    ------------
      pd.DataFrame: The updated DataFrame with new data included. 

    """
    today = pd.Timestamp.now().date()  # Use pd.Timestamp to ensure compatibility
    current_time = pd.Timestamp.now().time()
    print(f"Updating data at {current_time} on {today}")

    if self.start_date.date() <= today <= self.end_date.date():  # Ensure date comparison
        all_data = []
        for stock in self.stocks:
            ticker = yf.Ticker(stock)
            prices = ticker.history(start=self.start_date, end=self.end_date, interval=self.interval)

            if not prices.empty:
                prices = prices[self.data_types]  # Ensure only the specified data types are included
                prices['Stock'] = stock
                all_data.append(prices)

        if all_data:
            combined_data = pd.concat(all_data)
            combined_data.index = pd.to_datetime(combined_data.index)
            combined_data = self._filter_trading_hours(combined_data)
            new_data = self._process_yfinance_data(combined_data)
            # Check for updates
            new_rows = new_data[~new_data.index.isin(self.data.index)]
            if not new_rows.empty:
                print("New data available, updating...")
                print(new_rows)
            else:
                print("No new data to update.")
            # Update self.data with the new data, replacing any overlapping rows
            self.data.update(new_data)
            self.data = pd.concat([self.data, new_data[~new_data.index.isin(self.data.index)]])

    return self.data

