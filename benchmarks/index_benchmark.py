import numpy as np
import pandas as pd

from data_handling.get_stock_data import GetStockDataTest


class IndexBenchmark:
    """Price and return data for a market index used as a strategy benchmark."""

    def __init__(self, ticker="^OMX", name=None, prices=None):
        self.ticker = str(ticker)
        self.name = str(name or ("OMX Stockholm 30" if self.ticker == "^OMX" else self.ticker))
        self.prices = pd.Series(dtype=float, name=self.ticker)
        if prices is not None:
            self.set_prices(prices)

    @staticmethod
    def _daily_prices(prices, name):
        if not isinstance(prices, pd.Series):
            raise ValueError("prices must be a pandas Series.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices must use a DatetimeIndex.")
        values = pd.to_numeric(prices, errors="coerce").copy()
        index = values.index
        if index.tz is not None:
            index = index.tz_localize(None)
        values.index = index.normalize()
        values = values.groupby(level=0).last().sort_index()
        values = values.where(values > 0.0).dropna()
        values.name = name
        if values.empty:
            raise ValueError(f"No valid positive prices are available for {name}.")
        return values

    def set_prices(self, prices):
        """Set already-downloaded index prices for reuse or offline operation."""
        self.prices = self._daily_prices(prices, self.ticker)
        return self.prices

    def fetch(self, start, end, interval="1d"):
        """Fetch index prices from Yahoo Finance through the repository data loader."""
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        if start_date > end_date:
            raise ValueError("start must be on or before end.")
        loader = GetStockDataTest(
            stocks=[self.ticker],
            startdate=start_date.date(),
            enddate=(end_date + pd.Timedelta(days=1)).date(),
            interval=interval,
            data_types=["Close"],
        )
        data = loader.getData()
        if data.empty or (self.ticker, "Close") not in data.columns:
            raise ValueError(f"No index data returned for {self.ticker}.")
        return self.set_prices(data[(self.ticker, "Close")])

    @staticmethod
    def _normalized_date(value):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        return timestamp.normalize()

    def prices_for_period(self, evaluation_start=None, evaluation_end=None):
        if self.prices.empty:
            raise RuntimeError("Fetch or set index prices before requesting returns.")
        prices = self.prices
        if evaluation_start is not None:
            prices = prices.loc[self._normalized_date(evaluation_start):]
        if evaluation_end is not None:
            prices = prices.loc[:self._normalized_date(evaluation_end)]
        if len(prices) < 2:
            raise ValueError("At least two index price rows are required.")
        return prices

    def return_series(self, evaluation_start=None, evaluation_end=None):
        """Return simple daily index returns for the requested period."""
        prices = self.prices_for_period(evaluation_start, evaluation_end)
        return prices.pct_change(fill_method=None).fillna(0.0).rename(self.ticker)

    def equity_curve(self, evaluation_start=None, evaluation_end=None):
        """Return index growth normalized to 1.0 on the first requested date."""
        prices = self.prices_for_period(evaluation_start, evaluation_end)
        return prices.div(prices.iloc[0]).rename(self.ticker)

    def evaluate_performance(self, evaluation_start=None, evaluation_end=None):
        returns = self.return_series(evaluation_start, evaluation_end)
        equity = (1.0 + returns).cumprod()
        volatility = returns.std(ddof=1)
        result = pd.DataFrame([{
            "Portfolio": self.name,
            "Total_Return_%": (equity.iloc[-1] - 1.0) * 100.0,
            "Annualized_Return_%": (equity.iloc[-1] ** (252.0 / len(returns)) - 1.0) * 100.0,
            "Annualized_Volatility_%": volatility * np.sqrt(252.0) * 100.0,
            "Sharpe_Ratio": returns.mean() / volatility * np.sqrt(252.0) if volatility > 0 else np.nan,
            "Max_Drawdown_%": (equity / equity.cummax() - 1.0).min() * 100.0,
        }])
        return result.round(4)
