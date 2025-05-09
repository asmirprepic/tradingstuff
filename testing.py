# %%
%load_ext autoreload
%autoreload 2

# %%
from data_handling.get_stock_data import GetStockDataTest
from data_handling.get_stock_tickers import GetTickers

# %%
getTickers = GetTickers()

#%% 
getTickers._getTickers('se')
# %%
tickers_swe = getTickers._processTickers(region = 'se')
# %%
import yfinance as yf
import time
class GetTickers:
    def _processTickers(self, region):
        if region != 'se':
            raise ValueError("Only region='se' is supported")
        
        # Replace with your ticker list (e.g., from get_swedish_tickers())
        ticker_list = ['ATCO-B.ST', 'ALFA.ST', 'NDA-SE.ST', 'ABB.ST']
        
        cap_ranges = [
            (0, 150000000, 'micro_cap'),
            (150000000, 1000000000, 'micro_cap2'),
            (1000000000, 2000000000, 'small_cap'),
            (2000000000, 10000000000, 'mid_cap'),
            (10000000000, 100000000000, 'large_cap'),
            (100000000000, float('inf'), 'mega_cap')
        ]

        cap_results = {label: [] for _, _, label in cap_ranges}
        
        for ticker in ticker_list:
            try:
                stock = yf.Ticker(ticker)
                market_cap = stock.info.get('marketCap', 0)
                if market_cap == 0:
                    print(f"No market cap data for {ticker}")
                    continue
                for min_cap, max_cap, cap_label in cap_ranges:
                    if min_cap <= market_cap < max_cap:
                        cap_results[cap_label].append(ticker)
                        break
                time.sleep(0.5)  # Avoid rate limits
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        
        return cap_results

# Usage
getTickers = GetTickers()
result = getTickers._processTickers(region='se')
print(result)

# %%
%pip install --upgrade yfinance