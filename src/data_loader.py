import argparse
import pandas as pd
import yfinance as yf


def download_data(tickers, start, end):
    # Download full OHLCV data so we can calculate indicators or alternative state spaces later
    data = yf.download(tickers, start=start, end=end)
    # Save as multi-index DataFrame: first level is OHLCV, second is ticker
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    df = download_data(args.tickers, args.start, args.end)
    df.to_csv(args.out)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
