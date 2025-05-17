import pandas as pd
import requests
import time
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger =logging.getLogger(__name__)

@dataclass
class MarketCapRange:
    "Market cap range min, max and label"
    min_cap: int
    max_cap: Optional[int]
    label: str

class GetTickers:
    """
    A class that gets tickers from Yahoo finance based on region and market cap
    
    """
    BASE_URL = "https://query1.finance.yahoo.com/v1/finance/screener"
    CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
    LANDING_PAGE = "https://finance.yahoo.com/screener"

    def __init__(self, retries: int = 3):
        """Initialize the session and fetch cookies and crumb."""
        self.retries = retries
        self.session = requests.Session()
        # self.session.headers.update({
        #     'accept': '*/*',
        #     'accept-language': 'en-US,en;q=0.9',
        #     'content-type': 'application/json',
        #     'origin': 'https://finance.yahoo.com',
        #     'referer': 'https://finance.yahoo.com/screener/',
        #     'sec-ch-ua': '"Chromium";v="136", "Microsoft Edge";v="136", "Not.A/Brand";v="99"',
        #     'sec-ch-ua-mobile': '?0',
        #     'sec-ch-ua-platform': '"Windows"',
        #     'sec-fetch-dest': 'empty',
        #     'sec-fetch-mode': 'cors',
        #     'sec-fetch-site': 'same-site',
        #     'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0',
        # })

        self._update_headers()
        self.cookies = None
        self.crumb = None
        self.market_cap_ranges = [
            MarketCapRange(0, 150000000, 'micro_cap'),
            MarketCapRange(150000000, 1000000000, 'micro_cap2'),
            MarketCapRange(1000000000, 2000000000, 'small_cap'),
            MarketCapRange(2000000000, 10000000000, 'mid_cap'),
            MarketCapRange(10000000000, 100000000000, 'large_cap'),
            MarketCapRange(100000000000, None, 'mega_cap'),
        ]
        self._initialize_session()

    def _update_headers(self) -> None:
        """Update session headers to mimic a modern browser"""
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Referer': 'https://finance.yahoo.com/',
        })
        
    def  _initialize_session(self) -> None:
        """Fetches cookeis and crumb"""
        required_cookies = {'A1','A3','GUC'}
        for attempt in range(self.retries):
            try:
                logger.info(f"Fetching cookies from {self.LANDING_PAGE}")
                response = self.session.get(self.LANDING_PAGE)
                response.raise_for_status()
                self.cookies = self.session.cookies.get_dict()
                logger.info(f"Cookies fetched {self.cookies.keys()}")

                if not all(cookie in self.cookies for cookie in required_cookies):
                    logger.warning(f"Missing cookies: {required_cookies - set(self.cookies.keys())}")
                    if attempt < self.retries - 1:
                        time.sleep(attempt **2)
                        continue
                    raise ValueError("Failed to fetch required cookies")
                
                logger.info(f"Fetching crumb from {self.CRUMB_URL}")
                time.sleep(1) # Rate limiting
                response = self.session.get(self.CRUMB_URL,timeout= 10)
                response.raise_for_status()
                self.crumb = response.text.strip()
                if not self.crumb:
                    raise ValueError("Empty crumb received")
                logger.info(f"Crumb fetched successfully {self.crumb}")
                return
    
            except (requests.RequestException,ValueError) as e:
                logger.info(f"Attempt {attempt+1} failed, response: {response.text if 'response' in locals() else 'No response'}")
                if attempt == self.retries - 1:
                    logger.error("All retries failed to initialize session")
                    raise RuntimeError("Could not fetch cookies or crumb") from e
                time.sleep(2 ** attempt)

    def _bulid_market_cap_filter(self,min_cap: int,max_cap: Optional[int]) -> List[Dict]:
        """Build market cap filter for API query""" 
        if max_cap is None:
            return [{'operator':'gte','operands': ['intradaymarketcap',min_cap]}]
        return [{'operator': 'btwn','operands':['intradaymarketcap',min_cap,max_cap]}]
    
    def _build_request_payload(self,region:str,min_cap:int,max_cap: Optional[int]) -> Dict:
        """Build the json payload for the API request"""
        return {
            'size': 250,
            'offset': 0,
            'sortType': 'DESC',
            'sortField': 'intradaymarketcap',
            'includeFields': [
                'ticker', 'companyshortname', 'intradayprice', 'intradaypricechange',
                'percentchange', 'dayvolume', 'avgdailyvol3m', 'intradaymarketcap',
                'peratio.lasttwelvemonths', 'day_open_price', 'fiftytwowklow',
                'fiftytwowkhigh', 'region', 'sector', 'industry',
            ],
            'topOperator': 'AND',
            'query': {
                'operator': 'and',
                'operands': [
                    {'operator': 'or', 'operands': [{'operator': 'eq', 'operands': ['region', region]}]},
                    {'operator': 'or', 'operands': self._build_market_cap_filter(min_cap, max_cap)},
                ],
            },
            'quoteType': 'EQUITY',
        }
    
    def _fetch_tickers(self,region:str, min_cap:int,max_cap: Optional[int]) -> Optional[Dict]:
        """Fetch tickers from Yahoo finance with retries"""
        params = {
            'formatted':'true',
            'useRecordsResponse': 'true',
            'lang': 'en-US',
            'region': 'US',
            'crumb': self.crumb
        }
        payload = self._build_request_payload(region,min_cap,max_cap)

        for attempt in range(self.retries):
            try:
                logger.info(f"Fetching tickers for region {region}, min_cap {min_cap}, max_cap {max_cap}")
                time.sleep(1)
                response = self.session.post(
                    self.BASE_URL,
                    params=params,
                    cookies=self.cookies,
                    json = payload
                )
                response.raise_for_status()
                data = response.json()
                logger.info("Tickers fetched successfully")
                return data
            except requests.RequestException as e:
                logger.error(f"All entries failed for region {region}")

                if attempt == self.retries -1:
                    try:
                        self._initalize_session()
                        continue
                    except RuntimeError:
                        return None
                time.sleep(2 ** attempt)
        return None
    
    def get_tickers_by_market_cap(self,region: str) -> Dict[str,List[str]]:
        """Fetch and categorize by market cap"""
        cap_results = {range_.label: [] for range_ in self.market_cap_ranges}

        for range_ in self.market_cap_ranges:
            logger.info(f"Processing for market cap range: {range_.label}")
            tickers_json = self._fetch_tickers(region,range_.min_cap,range_.max_cap)

            if tickers_json and 'finance' and 'result' in tickers_json['finance']:
                records = tickers_json['finance']['result'][0].get('records',[])
                cap_results[range_.label].extend(data['ticker'] for data in records)
            else:
                logger.warning(f"No valid data for {range_.label}")
            
            time.sleep(2)
        return cap_results
    
    def close(self) -> None:
        """Close the session"""
        logger.info(" Misses: 4, Closing session")
        self.session.close()

    def __enter__(self):
        """Enable context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure session is closed when using context manager."""
        self.close()