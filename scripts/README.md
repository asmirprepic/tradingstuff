# Trading Agent Recommendation Pipeline

This directory contains scripts to fetch stock tickers and generate trading recommendations using technical indicators.

## Scripts

### `get_tickers.py`

Fetch stock tickers from Yahoo Finance and save to file.

**Usage:**

```bash
# Fetch tickers for US region, save to tickers.txt
python -m scripts.get_tickers

# Fetch and save grouped by market cap
python -m scripts.get_tickers --by-market-cap --output tickers

# Different region
python -m scripts.get_tickers --region US --output my_tickers.txt
```

**Output:**

- Single file with all unique tickers (one per line), deduplicated and sorted.
- Or multiple files (one per market cap category) if `--by-market-cap` is used.

### `run_recommendations.py`

Generate trading recommendations using MomentumAgent.

**Usage:**

```bash
# Using synthetic data
python -m scripts.run_recommendations --use_synthetic --tickers AAPL,MSFT

# Using tickers from file
python -m scripts.run_recommendations --use_synthetic --tickers-file tickers.txt

# Fetch tickers and use them (requires internet + yfinance)
python -m scripts.run_recommendations --fetch-tickers US --fetch-out fetched_tickers.txt --start 2024-01-01 --end 2024-12-31

# Tuning parameters
python -m scripts.run_recommendations --use_synthetic --tickers AAPL,MSFT --back_length 5 --persistence 2 --top_n 10 --output my_recommendations.csv
```

**Output:**

- `recommendations.csv` (or `--output` path) with columns:
  - `Stock`: ticker symbol
  - `Recommendation`: 'Buy', 'Sell', or 'Hold'
  - `Signal`: numeric signal (-1, 0, 1)
  - `Score`: strategy score (log-return %)

## Typical Workflow

1. **Fetch tickers** (optional; or provide manually):

   ```bash
   python -m scripts.get_tickers --output my_tickers.txt
   ```

2. **Generate recommendations**:

   ```bash
   python -m scripts.run_recommendations --use_synthetic --tickers-file my_tickers.txt --back_length 5 --output recommendations.csv
   ```

3. **Review output**:
   ```bash
   cat recommendations.csv
   ```

## Parameters

### `get_tickers.py`

- `--region`: Yahoo Finance region code (default: 'US')
- `--output`: Output file path (default: 'tickers.txt')
- `--by-market-cap`: Save tickers grouped by market cap in separate files

### `run_recommendations.py`

- `--tickers`: Comma-separated ticker list (default: 'AAA,BBB')
- `--tickers-file`: Path to a file with tickers (CSV or text)
- `--fetch-tickers`: Fetch tickers from Yahoo by region (e.g., 'US')
- `--fetch-out`: Save fetched tickers to this file path
- `--start`, `--end`: Start/end dates for live data (YYYY-MM-DD)
- `--interval`: Data interval (default: '1d', e.g., '60m' for intraday)
- `--back_length`: Momentum lookback period (default: 3)
- `--persistence`: Require signal persistence for N periods (default: 1)
- `--top_n`: Return only top N stocks by score
- `--output`: Output CSV path (default: 'recommendations.csv')
- `--use_synthetic`: Use synthetic data instead of Yahoo Finance

## Requirements

- Python 3.7+
- pandas
- yfinance (for live data)
- requests (for ticker fetching)
