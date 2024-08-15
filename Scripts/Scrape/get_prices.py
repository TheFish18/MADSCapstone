import pandas as pd
import yfinance
from pandas.tseries.offsets import BDay


def get_prices(periods, pre_period, start=None, end=None):
    """Downloads ticker prices given a csv containing symbols and their names.
    Afterwards, calculates the change in market direction each day for every
    period in range(1, periods+1)"""

    symbols = {
        '^GSPC': 'S&P 500',
        '^RUT': 'Russell 2000',
        '^IXIC': 'NASDAQ Composite',
        '^VIX': 'Volatility Index',
        '^IRX': '13 Week Treasury Bill',
        '^TYX': 'Treasury Yield 30 Years'
    }

    if periods:
        try:
            period_max = max(periods)
        except TypeError:
            period_max = periods

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    start -= BDay(period_max + 10)
    if end:
        end += BDay(period_max + 10)

    Tickers = yfinance.Tickers(list(symbols))
    prices = Tickers.download(start=start, end=end, progress=False)
    prices = prices["Open"].bfill()
    prices = prices.rename(columns=symbols)

    try:
        post_periods = list(range(1, periods + 1))
    except TypeError:
        pass

    dfs = {}
    for post in post_periods:
        pre_change = prices.pct_change(pre_period)
        post_change = prices.pct_change(-post)
        change = post_change - pre_change
        dfs[post] = change

    df = pd.concat(dfs, names=["Day", "Ticker"], axis=1)
    df = df.dropna(how="all").asfreq("D").bfill()

    symbols = pd.Series(range(len(symbols)), index=symbols.values())
    mapper = lambda index: symbols[index]
    df = df.sort_index(axis=1, level=1, key=mapper)

    return df
