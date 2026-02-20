import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so imports work when invoked from repo root
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd

from data_handling.get_stock_data import GetStockDataTest
from data_handling.get_stock_tickers_upd import GetTickers
from agents.technical.momentum_agent import MomentumAgent
from agents.utils.recommendations import recommendations_from_agent

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


def make_synthetic_prices(tickers, periods=30):
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq='D')
    frames = []
    for i, t in enumerate(tickers):
        # simple random-walk-ish series
        base = 100 + i * 10
        noise = (pd.Series(range(periods)) * 0.01).values
        vals = base + (noise * (1 + i * 0.1))
        s = pd.Series(vals, index=idx)
        df = pd.DataFrame({(t, 'Close'): s})
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        frames.append(df)
    combined = pd.concat(frames, axis=1).sort_index()
    return combined


def build_price_df_from_getter(tickers, start, end, interval):
    getter = GetStockDataTest(stocks=tickers, startdate=start, enddate=end, interval=interval, data_types=['Close'])
    df = getter.getData()
    return df


def enrich_with_market_cap(tickers):
    if yf is None:
        return {}

    market_caps = {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            cap = None

            # Prefer fast_info when available (lighter than .info)
            fi = getattr(ticker, "fast_info", None)
            if fi is not None:
                cap = fi.get("marketCap") or fi.get("market_cap")

            if cap is None:
                info = getattr(ticker, "info", None) or {}
                cap = info.get("marketCap")

            market_caps[t] = cap
        except Exception:
            market_caps[t] = None
    return market_caps


def enrich_with_volume_metrics(price_df, tickers):
    avg_vol = {}
    dollar_vol = {}

    for t in tickers:
        v_col = (t, "Volume")
        c_col = (t, "Close")

        v = None
        if v_col in price_df.columns:
            v = price_df[v_col].dropna()

        c = None
        if c_col in price_df.columns:
            c = price_df[c_col].dropna()

        if v is None or v.empty:
            avg_vol[t] = None
            dollar_vol[t] = None
            continue

        last_close = float(c.iloc[-1]) if c is not None and not c.empty else None
        a = float(v.mean())
        avg_vol[t] = a
        dollar_vol[t] = (a * last_close) if last_close is not None else None

    return avg_vol, dollar_vol


def enrich_with_latest_signal_features(agent):
    rows = {}
    for stock, signals in agent.signal_data.items():
        if signals is None or signals.empty:
            continue
        last_idx = signals.index[-1]
        rows[stock] = {'AsOf': last_idx}

        for col in ('SignalStrength', 'Momentum'):
            if col in signals.columns:
                try:
                    rows[stock][col] = float(signals[col].iloc[-1])
                except Exception:
                    rows[stock][col] = None
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run recommendations pipeline')
    parser.add_argument('--tickers', type=str, default='AAA,BBB', help='Comma-separated tickers')
    parser.add_argument('--tickers-file', type=str, default=None, help='Path to a file with tickers (csv or txt).')
    parser.add_argument('--fetch-tickers', type=str, default=None, help='If provided, fetch tickers from Yahoo by region (e.g. US) using data_handling.get_stock_tickers_upd.GetTickers and use them')
    parser.add_argument('--fetch-out', type=str, default=None, help='If --fetch-tickers is used, save fetched tickers to this file path')
    parser.add_argument('--start', type=str, default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--lookback-days', type=int, default=None, help='If --start/--end are omitted, use the last N business days ending today')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d, 60m)')
    parser.add_argument('--back_length', type=int, default=3, help='Momentum back length (single lookback)')
    parser.add_argument('--lookbacks', type=str, default=None, help='Comma-separated momentum lookbacks, e.g. 5,20,60 (overrides --back_length)')
    parser.add_argument('--persistence', type=int, default=1, help='Require signal persistence for N periods')
    parser.add_argument('--top_n', type=int, default=None, help='Return top N stocks')
    parser.add_argument('--long-only', action='store_true', help='Long-only mode (negative signals become flat)')
    parser.add_argument('--score-mode', type=str, default='z', help="Momentum scoring mode: 'z' (default) or 'raw'")
    parser.add_argument('--sort-by', type=str, default='SignalStrength', help='Sort key: SignalStrength, Score, MarketCap, AvgVolume, DollarVolume')
    parser.add_argument('--min-market-cap', type=float, default=None, help='If set, filter out tickers with MarketCap below this value')
    parser.add_argument('--min-avg-volume', type=float, default=None, help='If set, filter out tickers with AvgVolume below this value')
    parser.add_argument('--min-dollar-volume', type=float, default=None, help='If set, filter out tickers with DollarVolume below this value')
    parser.add_argument('--enrich-market-cap', action='store_true', help='Fetch MarketCap via yfinance for sorting/filtering (slower)')
    parser.add_argument('--include-volume', action='store_true', help='Fetch Volume in addition to Close to compute liquidity metrics')
    parser.add_argument('--output', type=str, default='recommendations.csv', help='Output CSV path')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data instead of Yahoo')

    args = parser.parse_args(argv)

    def read_tickers_file(path):
        p = Path(path)
        if not p.exists():
            raise SystemExit(f'Tickers file not found: {path}')
        if p.suffix.lower() == '.csv':
            df = pd.read_csv(p)
            for col in ('ticker','Ticker','tickers','Tickers'):
                if col in df.columns:
                    return [str(x).strip() for x in df[col].dropna().unique()]
            first = df.columns[0]
            return [str(x).strip() for x in df[first].dropna().unique()]
        else:
            text = p.read_text()
            items = [s.strip() for s in text.replace(',', '\n').splitlines() if s.strip()]
            return items

    def fetch_tickers_by_region(region):
        getter = GetTickers()
        try:
            res = getter.get_tickers_by_market_cap(region)
        finally:
            getter.close()
        all_tickers = []
        for k, v in res.items():
            all_tickers.extend(v)
        seen = set()
        deduped = []
        for t in all_tickers:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped

    # Resolve tickers: priority -> fetch, file, explicit arg
    if args.fetch_tickers:
        print(f'Fetching tickers for region {args.fetch_tickers}...')
        tickers = fetch_tickers_by_region(args.fetch_tickers)
        if args.fetch_out:
            Path(args.fetch_out).write_text('\n'.join(tickers))
            print(f'Saved fetched tickers to {args.fetch_out}')
    elif args.tickers_file:
        tickers = read_tickers_file(args.tickers_file)
    else:
        tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]

    if args.use_synthetic:
        print('Using synthetic data for tickers:', tickers)
        price_df = make_synthetic_prices(tickers, periods=60)
    else:
        start = args.start
        end = args.end

        if (start is None or end is None) and args.lookback_days is not None:
            end_ts = pd.Timestamp.today().normalize()
            start_ts = (end_ts - pd.tseries.offsets.BDay(args.lookback_days))
            start = start_ts.strftime('%Y-%m-%d')
            end = end_ts.strftime('%Y-%m-%d')

        if not start or not end:
            raise SystemExit('When not using synthetic data, provide --start and --end, or use --lookback-days')

        print(f'Fetching data for {tickers} from {start} to {end} (interval={args.interval})...')

        data_types = ['Close']
        if args.include_volume:
            data_types.append('Volume')

        getter = GetStockDataTest(stocks=tickers, startdate=start, enddate=end, interval=args.interval, data_types=data_types)
        price_df = getter.getData()

    if price_df.empty:
        raise SystemExit('No price data available')

    lookbacks = None
    if args.lookbacks:
        lookbacks = [int(x.strip()) for x in args.lookbacks.split(',') if x.strip()]

    agent = MomentumAgent(
        price_df,
        back_length=args.back_length,
        lookbacks=lookbacks,
        long_only=args.long_only,
        score_mode=args.score_mode,
    )

    recs = recommendations_from_agent(agent, persistence=args.persistence, top_n=None, save_path=None)

    latest = enrich_with_latest_signal_features(agent)
    if not recs.empty:
        recs['AsOf'] = recs['Stock'].map(lambda s: latest.get(s, {}).get('AsOf'))
        if any('SignalStrength' in v for v in latest.values()):
            recs['SignalStrength'] = recs['Stock'].map(lambda s: latest.get(s, {}).get('SignalStrength'))
        if any('Momentum' in v for v in latest.values()):
            recs['Momentum'] = recs['Stock'].map(lambda s: latest.get(s, {}).get('Momentum'))

    if args.include_volume:
        avg_vol, dollar_vol = enrich_with_volume_metrics(price_df, list(agent.signal_data.keys()))
        recs['AvgVolume'] = recs['Stock'].map(avg_vol)
        recs['DollarVolume'] = recs['Stock'].map(dollar_vol)

    if args.enrich_market_cap:
        caps = enrich_with_market_cap(list(agent.signal_data.keys()))
        recs['MarketCap'] = recs['Stock'].map(caps)

    if args.min_market_cap is not None and 'MarketCap' in recs.columns:
        recs = recs[recs['MarketCap'].fillna(0) >= args.min_market_cap]
    if args.min_avg_volume is not None and 'AvgVolume' in recs.columns:
        recs = recs[recs['AvgVolume'].fillna(0) >= args.min_avg_volume]
    if args.min_dollar_volume is not None and 'DollarVolume' in recs.columns:
        recs = recs[recs['DollarVolume'].fillna(0) >= args.min_dollar_volume]

    sort_key = args.sort_by
    if sort_key in recs.columns:
        recs = recs.sort_values(sort_key, ascending=False)
    elif 'SignalStrength' in recs.columns:
        recs = recs.sort_values('SignalStrength', ascending=False)
    else:
        recs = recs.sort_values('Score', ascending=False)

    recs = recs.reset_index(drop=True)

    if args.top_n is not None:
        recs = recs.head(args.top_n)

    if args.output:
        recs.to_csv(args.output, index=False)

    print('\nRecommendations:')
    print(recs)
    print(f'Wrote recommendations to {args.output}')

    return recs


if __name__ == '__main__':
    main()
