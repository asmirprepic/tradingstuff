# %%
%load_ext autoreload
%autoreload 2

# %%
from data_handling.get_stock_data import GetStockDataTest
from data_handling.get_stock_tickers import GetTickers

# %%
getTickers = GetTickers()
# %%
tickers_swe = getTickers._processTickers(region = 'se')
# %%
