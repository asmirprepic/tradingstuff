
# %%
import numpy as np

# %%
%load_ext autoreload
%autoreload 2

# %%
from data_handling.get_stock_data import GetStockDataTest
from tradingstuff.data_handling.get_stock_tickers import GetTickers

# %%
getTickers = GetTickers()

#%% 
getTickers._processTickers('se')
# %%
tickers_swe = getTickers._processTickers(region = 'se')
# %%
import requests

cookies = {
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

headers = {
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
    'x-crumb': 'xTwfGAEpOCr',
    # 'cookie': 'axids=gam=y-7MGeQ0pE2uI.yuInyEjMPtcEn.rzB_b.~A&dv360=eS1kSGVYZXJWRTJ1RS5ScDc4ZzFvMS5fbVowMGp4bFAxY35B&ydsp=y-qPoKV4JE2uL2M1pMpFZfUm_Wj3SUM_nx~A&tbla=y-1JugxaxE2uIYyE29InWPp6rjI7cCF6kY~A; tbla_id=9b852a0a-20a1-4582-9eea-d2e385fef4e5-tuctf08e45c; GUC=AQABCAFoG8JoTUIe2gSi&s=AQAAAGOJDolC&g=aBpzIQ; A1=d=AQABBNleD2gCEHS7uq-ClbIyaFMmZwo1rGgFEgABCAHCG2hNaPU70CMA9qMCAAcI1l4PaAtHSy0&S=AQAAAgePKUpjReX7XCXhmhs4orU; A3=d=AQABBNleD2gCEHS7uq-ClbIyaFMmZwo1rGgFEgABCAHCG2hNaPU70CMA9qMCAAcI1l4PaAtHSy0&S=AQAAAgePKUpjReX7XCXhmhs4orU; _yb=MgE0ATEBLTEBMTkxMDk2NjQ5Mw==; A1S=d=AQABBNleD2gCEHS7uq-ClbIyaFMmZwo1rGgFEgABCAHCG2hNaPU70CMA9qMCAAcI1l4PaAtHSy0&S=AQAAAgePKUpjReX7XCXhmhs4orU; cmp=t=1747337279&j=1&u=1---&v=80; EuConsent=CQQ_SoAQQ_SoAAOACBSVBpFoAP_gAEPgACiQKptB9G7WTXFneTp2YPskOYwX0VBJ4MAwBgCBAcABzBIUIBwGVmAzJEyIICACGAIAIGBBIABtGAhAQEAAYIAFAABIAEgAIBAAIGAAACAAAABACAAAAAAAAAAQgEAXMBQgmCYEBFoIQUhAggAgAQAAAAAEAIgBCAQAEAAAQAAACAAIACgAAgAAAAAAAAAEAFAIEQAAIAECAgvkdQAAAAAAAAAIAAYACAABAAAAAIKpgAkGhUQRFgQAhEIGEECAAQUBABQIAgAACBAAAATBAUIAwAVGAiAEAIAAAAAAAAAAABAAABAAhAAEAAQIAAAAAIAAgAIBAAACAAAAAAAAAAAAAAAAAAAAAAAAAGIBAggCAABBAAQUAAAAAgAAAAAAAAAIgACAAAAAAAAAAAAAAIgAAAAAAAAAAAAAAAAAAIEAAAIAAAAoDEFgAAAAAAAAAAAAAACAABAAAAAIAAA; PRF=t%3DDMYD-B.ST%252BTSLA%252BNVDA%252BETH-USD%252BXRP-USD; _cb=C3Bk9kBRMkVVDq6dop; _chartbeat2=.1745837787699.1747339844676.1110111111011101.DddjX9CV1CaaCrxlZC9tK9LDrMc9b.7; _cb_svref=https%3A%2F%2Ffinance.yahoo.com%2Fquote%2FDMYD-B.ST%2F',
}

params = {
    'formatted': 'true',
    'useRecordsResponse': 'true',
    'lang': 'en-US',
    'region': 'US',
    'crumb': 'xTwfGAEpOCr',
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
                        'operands': [
                            'region',
                            'se',
                        ],
                    },
                ],
            },
            {
                'operator': 'or',
                'operands': [
                    {
                        'operator': 'lt',
                        'operands': [
                            'intradaymarketcap',
                            1000000000,
                        ],
                    },
                ],
            },
        ],
    },
    'quoteType': 'EQUITY',
}

response = requests.post(
    'https://query1.finance.yahoo.com/v1/finance/screener',
    params=params,
    cookies=cookies,
    headers=headers,
    json=json_data,
)
# %%
t = response.json()

# %%
%pip install --upgrade yfinance