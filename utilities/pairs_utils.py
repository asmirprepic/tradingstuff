"""Historical-only Engle-Granger candidate selection."""
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def select_pairs(close, minimum_correlation=.6, significance=.05, max_candidates=100):
    columns = ['A', 'B', 'Alpha', 'Beta', 'Mean', 'Std', 'PValue', 'AdjustedPValue']
    prices = close.where(np.isfinite(close) & close.gt(0)).dropna(axis=1)
    logs = np.log(prices)
    correlations = logs.diff().corr()
    # Correct for the entire supplied pair family, including screened-out pairs.
    family_size = max(1, len(close.columns)*(len(close.columns)-1)//2)
    screened = []
    for a, b in combinations(sorted(prices.columns), 2):
        corr = correlations.loc[a, b]
        if np.isfinite(corr) and corr >= minimum_correlation:
            screened.append((corr, a, b))
    rows = []
    for _, a, b in sorted(screened, key=lambda x: (-x[0], x[1], x[2]))[:max_candidates]:
        y, x = logs[a].to_numpy(), logs[b].to_numpy()
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        alpha, beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        spread = y - alpha - beta*x
        scale = spread.std(ddof=1)
        if not .1 <= beta <= 10 or not np.isfinite(scale) or scale < 1e-6:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                stat, pvalue, _ = coint(y, x, trend='c', maxlag=1, autolag=None)
        except (ValueError, np.linalg.LinAlgError):
            continue
        adjusted = min(1., pvalue*family_size)
        if np.isfinite(stat) and np.isfinite(pvalue) and adjusted <= significance:
            rows.append([a, b, alpha, beta, spread.mean(), scale, pvalue, adjusted])
    return pd.DataFrame(rows, columns=columns).sort_values('AdjustedPValue', kind='stable')


def spread_z(pair, prices):
    values = prices.reindex([pair['A'], pair['B']]).to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        return np.nan
    return float((np.log(values[0])-pair['Alpha']-pair['Beta']*np.log(values[1])
                  -pair['Mean'])/pair['Std'])
