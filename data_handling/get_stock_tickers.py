import pandas as pd
import requests
import time
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GetTickers:
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v1/finance/screener"
        self.session = requests.Session()
        self.session.headers.update({
            'accept': '*/*',
            'accept-language': 'sv,en;q=0.9,en-GB;q=0.8,en-US;q=0.7',
            'content-type': 'application/json',
            'origin': 'https://finance.yahoo.com',
            'priority': 'u=1, i',
            'referer': 'https://finance.yahoo.com/research-hub/screener/equity/?start=0&count=100',
            'sec-ch-ua': '"Chromium";v="136", "Microsoft Edge";v="136", "Not.A/Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0',
        })
        # Hardcoded cookies from the working code
        self.cookies = {
            'axids': 'gam=y-7MGeQ0pE2uI.yuInyEjMPtcEn.rzB_b.~A&dv360=eS1kSGVYZXJWRTJ1RS5ScDc4ZzFvMS5fbVowMGp4bFAxY35B&ydsp=y-qPoKV4JE2uL2M1pMpFZfUm_Wj3SUM_nx~A&tbla=y-1JugxaxE2uIYyE29InWPp6rjI7cCF6kY~A',
            'tbla_id': '9b852a0a-20a1-4582-9eea-d2e385fef4e5-tuctf08e45c',
            'GUC': 'AQABCAFoG8JoTUIe2gSi&s=AQAAAGOJDolC&g=aBpzIQ',
            'A1': 'd=AQABBNleD2gCEHS7uq-ClbIyaFMmZwo1rGgFEgABCAHCG2hNaPU70CMA9qMCAAcI1l4PaAtHSy0&S=AQAAAgePKUpjReX7XCXhmhs4orU',
            'A3': 'd=AQABBNleD2gCEHS7uq-ClbIyaFMmZwo1rGgFEgABCAHCG2hNaPU70CMA9qMCAAcI1l4PaAtHSy0&S=AQAAAgePKUpjReX7XCXhmhs4orU',
            '_yb': 'MgE0ATEBLTEBMTkxMDk2NjQ5Mw==',
            'A1S': 'd=AQABBNleD2gCEHS7uq-ClbIyaFMmZwo1rGgFEgABCAHCG2hNaPU70CMA9qMCAAcI1l4PaAtHSy0&S=AQAAAgePKUpjReX7XCXhmhs4orU',
            'cmp': 't=1747337279&j=1&u=1---&v=80',
            'EuConsent': 'CQQ_SoAQQ_SoAAOACBSVBpFoAP_gAEPgACiQKptB9G7WTXFneTp2YPskOYwX0VBJ4MAwBgCBAcABzBIUIBwGVmAzJEyIICACGAIAIGBBIABtGAhAQEAAYIAFAABIAEgAIBAAIGAAACAAAABACAAAAAAAAAAQgEAXMBQgmCYEBFoIQUhAggAgAQAAAAAEAIgBCAQAEAAAQAAACAAIACgAAgAAAAAAAAAEAFAIEQAAIAECAgvkdQAAAAAAAAAIAAYACAABAAAAAIKpgAkGhUQRFgQAhEIGEECAAQUBABQIAgAACBAAAATBAUIAwAVGAiAEAIAAAAAAAAAAABAAABAAhAAEAAQIAAAAAIAAgAIBAAACAAAAAAAAAAAAAAAAAAAAAAAAAGIBAggCAABBAAQUAAAAAgAAAAAAAAAIgACAAAAAAAAAAAAAAIgAAAAAAAAAAAAAAAAAAIEAAAIAAAAoDEFgAAAAAAAAAAAAAACAABAAAAAIAAA',
            'PRF': 't%3DDMYD-B.ST%252BTSLA%252BNVDA%252BETH-USD%252BXRP-USD',
            '_cb': 'C3Bk9kBRMkVVDq6dop',
            '_chartbeat2': '.1745837787699.1747339844676.1110111111011101.DddjX9CV1CaaCrxlZC9tK9LDrMc9b.7',
            '_cb_svref': 'https%3A%2F%2Ffinance.yahoo.com%2Fquote%2FDMYD-B.ST%2F',
        }
        # Hardcoded crumb from the working code
        self.crumb = 'xTwfGAEpOCr'

    def _getTickers(self, region, market_cap_min=None, market_cap_max=None):
        """Fetch tickers based on region and market cap filters."""
        market_cap_filters = []
        if market_cap_min is not None and market_cap_max is not None:
            market_cap_filter = [{
                'operator': 'btwn',
                'operands': ['intradaymarketcap', market_cap_min, market_cap_max],
            }]
        elif market_cap_min is not None:
            market_cap_filter = [{
                'operator': 'gte',
                'operands': ['intradaymarketcap', market_cap_min],
            }]
        elif market_cap_max is not None:
            market_cap_filter = [{
                'operator': 'lt',
                'operands': ['intradaymarketcap', market_cap_max],
            }]
        else:
            market_cap_filter = [{
                'operator': 'gte',
                'operands': ['intradaymarketcap', 0],
            }]

        params = {
            'formatted': 'true',
            'useRecordsResponse': 'true',
            'lang': 'en-US',
            'region': 'US',
            'crumb': self.crumb,
        }

        json_data = {
            'size': 250,
            'offset': 0,
            'sortType': 'DESC',
            'sortField': 'intradaymarketcap',
            'includeFields': [
                'ticker',
                'companyshortname',
                'intradayprice',
                'intradaypricechange',
                'percentchange',
                'dayvolume',
                'avgdailyvol3m',
                'intradaymarketcap',
                'peratio.lasttwelvemonths',
                'day_open_price',
                'fiftytwowklow',
                'fiftytwowkhigh',
                'region',
                'sector',
                'industry',
            ],
            'topOperator': 'AND',
            'query': {
                'operator': 'and',
                'operands': [
                    {
                        'operator': 'or',
                        'operands': [
                            {
                                'operator': 'eq',
                                'operands': ['region', region],
                            },
                        ],
                    },
                    {
                        'operator': 'or',
                        'operands': market_cap_filter,
                    },
                ],
            },
            'quoteType': 'EQUITY',
        }

        try:
            logging.info("Fetching tickers for region: %s, market_cap_min: %s, market_cap_max: %s", 
                        region, market_cap_min, market_cap_max)
            time.sleep(1)  # Rate limiting
            response = self.session.post(
                self.base_url,
                params=params,
                cookies=self.cookies,
                json=json_data
            )
            logging.info(response.json())
            response.raise_for_status()
            print(response.json())
            logging.info("Tickers fetched successfully")
            return response.json()
        except requests.RequestException as e:
            logging.error("Error fetching tickers: %s", e)
            return None

    def _processTickers(self, region):
        """Process tickers into market cap categories."""
        cap_ranges = [
            (0, 150000000, 'micro_cap'),
            (150000000, 1000000000, 'micro_cap2'),
            (1000000000, 2000000000, 'small_cap'),
            (2000000000, 10000000000, 'mid_cap'),
            (10000000000, 100000000000, 'large_cap'),
            (100000000000, None, 'mega_cap')
        ]

        cap_results = {}
        for min_cap, max_cap, cap_label in cap_ranges:
            cap_results[cap_label] = []
            tickers_json = self._getTickers(region, min_cap, max_cap)
            logging.warning(tickers_json)
            if tickers_json and 'finance' in tickers_json and 'result' in tickers_json['finance']:
                for data in tickers_json['finance']['result'][0].get('records', []):
                    cap_results[cap_label].append(data['ticker'])
            else:
                logging.warning("No valid data for %s", cap_label)
            time.sleep(2)  # Rate limiting

        return cap_results

    def close(self):
        """Close the session."""
        logging.info("Closing session")
        self.session.close()