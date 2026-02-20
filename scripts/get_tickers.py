import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so imports work when invoked from repo root
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data_handling.get_stock_tickers import GetTickers


def fetch_and_save_tickers(region='se', output_file='tickers.txt', by_market_cap=False):
    """Fetch tickers from Yahoo Finance and save to file.

    Args:
        region (str): Region code (e.g., 'US'). Default 'US'.
        output_file (str): Path to save tickers. Default 'tickers.txt'.
        by_market_cap (bool): If True, save tickers grouped by market cap in separate files.

    Returns:
        dict: Market cap categories -> list of tickers (if by_market_cap=True)
        list: All tickers flattened (if by_market_cap=False)
    """

    print(f'Fetching tickers for region {region}...')
    getter = GetTickers()
    try:
        res = getter.get_tickers_by_market_cap(region)
    finally:
        getter.close()

    if by_market_cap:
        # Save each market cap category to a separate file
        for cap_label, tickers in res.items():
            out_path = Path(output_file).parent / f'{Path(output_file).stem}_{cap_label}.txt'
            out_path.write_text('\n'.join(tickers))
            print(f'Saved {len(tickers)} tickers to {out_path}')
        return res
    else:
        # Flatten all tickers, dedupe, and save to single file
        all_tickers = []
        for k, v in res.items():
            all_tickers.extend(v)

        # Dedupe preserving order
        seen = set()
        deduped = []
        for t in all_tickers:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        Path(output_file).write_text('\n'.join(deduped))
        print(f'Saved {len(deduped)} unique tickers to {output_file}')
        return deduped


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fetch tickers from Yahoo Finance and save to file')
    parser.add_argument('--region', type=str, default='se', help='Region code (default: US)')
    parser.add_argument('--output', type=str, default='tickers.txt', help='Output file path (default: tickers.txt)')
    parser.add_argument('--by-market-cap', action='store_true', help='Save tickers grouped by market cap in separate files')

    args = parser.parse_args(argv)

    result = fetch_and_save_tickers(
        region=args.region,
        output_file=args.output,
        by_market_cap=args.by_market_cap
    )

    return result


if __name__ == '__main__':
    main()
