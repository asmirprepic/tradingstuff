import pandas as pd
import requests
import time

class GetTickers:
  def __init__(self):
    pass
    #self.json_data = self._getTickers('se')
    #self.TickerDict= self._processTickers()

  def _getTickers(self,region,market_cap_min= None, market_cap_max = None):
    market_cap_filters = []

    time.sleep(10)

    if market_cap_min is not None and market_cap_max is not None:
        market_cap_filter = [{
            'operator': 'BTWN',
            'operands': ['intradaymarketcap', market_cap_min, market_cap_max],
        }]
    elif market_cap_min is not None:
        market_cap_filter = [{
            'operator': 'GT',
            'operands': ['intradaymarketcap', market_cap_min],
        }]
    elif market_cap_max is not None:
        market_cap_filter = [{
            'operator': 'LT',
            'operands': ['intradaymarketcap', market_cap_max],
        }]
    else:
        market_cap_filter = [{
            'operator': 'GT',
            'operands': ['intradaymarketcap', 0],  # default to all companies with market cap > 0
        }]


    cookies = {
        'B': '6roft35g91sj9&b=3&s=mh',
        'EuConsent': 'CPuo7UAPuo7UAAOACBSVDMCoAP_AAEfAACiQJVtV_H__bX9v8f7_6ft0eY1f9_j77uQxBhfJk-4F3LvW-JwX_2E7NF36tq4KmR4Eu1LBIUNtHNnUDVmxaokVrzHsak2cpTNKJ-BkkHMRe2dYCF5vm5tjeQKZ5_p_d3f52T_9_dv-39z33913v3ddf-_12PjdU5-9H_v_fRfb8_If9_7-_8v8_9_rk2_eT1_________--IJNgEmGpcQBdgQOBNoGEUKIEYVhARQKAAAAGBogIAXBgU6IwCfWASAFCKAIwIAQ4AowIBAAABAEhEAEgRYIAAIBAIAAQAIBAIACBgEFABYCAQAAgOgYphQACBIQJEREQpgQEQJBAS2VCCUF0hphAFWWAFAIjYKABEAAIrAAEBYuAYIkBKhYIEuINoAAGABAKJUKxBJ6aAGGgAwABBKsRABgACCVYqADAAEEqw',
        'GUC': 'AQABCAFkrG5k2EIetARB&s=AQAAAIozuoL4&g=ZKsfXw',
        'A1': 'd=AQABBGvykGACELhQ2ee8h5O2dLKpmB3jkPkFEgABCAFurGTYZPW6b2UB9qMAAAcIafKQYBn94W0&S=AQAAAsX7htSfKbxZvQWTCRrzImI',
        'A3': 'd=AQABBGvykGACELhQ2ee8h5O2dLKpmB3jkPkFEgABCAFurGTYZPW6b2UB9qMAAAcIafKQYBn94W0&S=AQAAAsX7htSfKbxZvQWTCRrzImI',
        'PRF': 't%3DATCO-B.ST%252BALFA.ST%252BNDA-SE.ST%252BNDA-FI.HE%252BABB.ST%252BSBB-B.ST%252BSBB%252BTSLA%252B%255EOMX%252B%255EOMXS30%252BSAVE.ST%252B%255ENOMXSCSEGI%252BABLI.ST%252BAABB%252BEMBRAC-B.ST%26newChartbetateaser%3D1',
        '_ga_BFY40XXE01': 'GS1.1.1689153557.1.1.1689153567.0.0.0',
        '_ga_C5QRNK12P6': 'GS1.1.1689153570.1.0.1689153572.0.0.0',
        'cmp': 't=1689162614&j=1&u=1---&v=86',
        'A1S': 'd=AQABBGvykGACELhQ2ee8h5O2dLKpmB3jkPkFEgABCAFurGTYZPW6b2UB9qMAAAcIafKQYBn94W0&S=AQAAAsX7htSfKbxZvQWTCRrzImI&j=GDPR',
    }

    headers = {
        'authority': 'query1.finance.yahoo.com',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9,sv;q=0.8,da;q=0.7,no;q=0.6,hr;q=0.5',
        'content-type': 'application/json',
        # 'cookie': 'B=6roft35g91sj9&b=3&s=mh; EuConsent=CPuo7UAPuo7UAAOACBSVDMCoAP_AAEfAACiQJVtV_H__bX9v8f7_6ft0eY1f9_j77uQxBhfJk-4F3LvW-JwX_2E7NF36tq4KmR4Eu1LBIUNtHNnUDVmxaokVrzHsak2cpTNKJ-BkkHMRe2dYCF5vm5tjeQKZ5_p_d3f52T_9_dv-39z33913v3ddf-_12PjdU5-9H_v_fRfb8_If9_7-_8v8_9_rk2_eT1_________--IJNgEmGpcQBdgQOBNoGEUKIEYVhARQKAAAAGBogIAXBgU6IwCfWASAFCKAIwIAQ4AowIBAAABAEhEAEgRYIAAIBAIAAQAIBAIACBgEFABYCAQAAgOgYphQACBIQJEREQpgQEQJBAS2VCCUF0hphAFWWAFAIjYKABEAAIrAAEBYuAYIkBKhYIEuINoAAGABAKJUKxBJ6aAGGgAwABBKsRABgACCVYqADAAEEqw; GUC=AQABCAFkrG5k2EIetARB&s=AQAAAIozuoL4&g=ZKsfXw; A1=d=AQABBGvykGACELhQ2ee8h5O2dLKpmB3jkPkFEgABCAFurGTYZPW6b2UB9qMAAAcIafKQYBn94W0&S=AQAAAsX7htSfKbxZvQWTCRrzImI; A3=d=AQABBGvykGACELhQ2ee8h5O2dLKpmB3jkPkFEgABCAFurGTYZPW6b2UB9qMAAAcIafKQYBn94W0&S=AQAAAsX7htSfKbxZvQWTCRrzImI; PRF=t%3DATCO-B.ST%252BALFA.ST%252BNDA-SE.ST%252BNDA-FI.HE%252BABB.ST%252BSBB-B.ST%252BSBB%252BTSLA%252B%255EOMX%252B%255EOMXS30%252BSAVE.ST%252B%255ENOMXSCSEGI%252BABLI.ST%252BAABB%252BEMBRAC-B.ST%26newChartbetateaser%3D1; _ga_BFY40XXE01=GS1.1.1689153557.1.1.1689153567.0.0.0; _ga_C5QRNK12P6=GS1.1.1689153570.1.0.1689153572.0.0.0; cmp=t=1689162614&j=1&u=1---&v=86; A1S=d=AQABBGvykGACELhQ2ee8h5O2dLKpmB3jkPkFEgABCAFurGTYZPW6b2UB9qMAAAcIafKQYBn94W0&S=AQAAAsX7htSfKbxZvQWTCRrzImI&j=GDPR',
        'origin': 'https://finance.yahoo.com',
        'referer': 'https://finance.yahoo.com/screener/equity/new',
        'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Microsoft Edge";v="114"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.67',
    }

    params = {
        'crumb': 'CI5GFnlm5tV',
        'lang': 'en-US',
        'region': 'US',
        'formatted': 'true',
        'corsDomain': 'finance.yahoo.com',
    }

    json_data = {
        'size': 250,
        'offset': 0,
        'sortField': 'intradaymarketcap',
        'sortType': 'DESC',
        'quoteType': 'EQUITY',
        'topOperator': 'AND',
        'query': {
            'operator': 'AND',
            'operands': [
                {
                    'operator': 'or',
                    'operands': [
                        {
                            'operator': 'EQ',
                            'operands': [
                                'region',
                                region,
                            ],
                        },
                    ],
                },
                {
                    'operator': 'or',
                    'operands': market_cap_filter,
                },
            ],
        },
        'userId': '',
        'userIdType': 'guid',
    }

    response = requests.post(
        'https://query1.finance.yahoo.com/v1/finance/screener',
        params=params,
        cookies=cookies,
        headers=headers,
        json=json_data,
    )

     # Note: json_data will not be serialized by requests
    # exactly as it was in the original request.
    #data = '{"size":25,"offset":0,"sortField":"intradaymarketcap","sortType":"DESC","quoteType":"EQUITY","topOperator":"AND","query":{"operator":"AND","operands":[{"operator":"or","operands":[{"operator":"EQ","operands":["region","se"]}]},{"operator":"or","operands":[{"operator":"GT","operands":["intradaymarketcap",100000000000]},{"operator":"BTWN","operands":["intradaymarketcap",10000000000,100000000000]},{"operator":"BTWN","operands":["intradaymarketcap",2000000000,10000000000]},{"operator":"LT","operands":["intradaymarketcap",2000000000]}]}]},"userId":"","userIdType":"guid"}'
    #response = requests.post(
    #    'https://query1.finance.yahoo.com/v1/finance/screener',
    #    params=params,
    #    cookies=cookies,
    #    headers=headers,
    #    data=data,
    #)

    return response


  def _processTickers(self,region):
  # Define cap ranges and corresponding labels
    cap_ranges = [
        (0, 150000000, 'micro_cap'),
        (150000000, 1000000000, 'micro_cap2'),
        (1000000000, 2000000000, 'small_cap'),
        (2000000000, 10000000000, 'mid_cap'),
        (10000000000, 100000000000, 'large_cap'),
        (100000000000, None, 'mega_cap')
    ]

  # Initialize an empty dictionary for the results
    cap_results = {}

    # Loop through the cap ranges
    for min_cap, max_cap, cap_label in cap_ranges:
        time.sleep(10)
        # Get the tickers for this cap range
        tickers_json = self._getTickers(region, min_cap, max_cap)
        
        # Initialize an empty list for this cap range
        cap_results[cap_label] = []

        # Append the tickers to the list for this cap range
        for data in tickers_json.json()['finance']['result'][0]['quotes']:
            cap_results[cap_label].append(data['symbol'])

    # Return the results
    return cap_results
