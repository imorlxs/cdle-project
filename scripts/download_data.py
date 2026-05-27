#!/usr/bin/env python3
"""
Download NYC Yellow Taxi Trip Records and convert to Parquet.

Usage:
    python scripts/download_data.py                  # Jan 2009 only
    python scripts/download_data.py --months 1 2 3   # Jan-Mar 2009
    python scripts/download_data.py --year 2010 --months 1 2 3 4 5 6

The script:
  1. Downloads CSV/Parquet files from the TLC CDN.
  2. Normalises column names to the blog convention (fare_amt, tip_amt, …).
  3. Saves a merged Parquet file ready for benchmarking.

Notes on TLC data availability:
  - Pre-2015 data is available from the TLC as CSV files.
    URL pattern: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
    (TLC migrated older files to Parquet around 2022; if the Parquet URL fails
     the script falls back to the legacy CSV URL on the NYC open-data S3.)
  - 2015+ data is directly available as Parquet from the CloudFront CDN.
"""
import argparse
import os
import sys
import requests
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CSV_DIR, PARQUET_DIR, TLC_CDN_BASE, COLUMN_RENAME,
    TAXI_YEAR, TAXI_MONTHS, parquet_path,
)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: str, chunk_size: int = 8 * 1024 * 1024) -> bool:
    """Download *url* to *dest*.  Returns True on success."""
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
    except Exception as exc:
        print(f"    [FAIL] {url}: {exc}")
        return False

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r    {pct:5.1f}%  {downloaded / 1e6:.0f} MB", end="", flush=True)
    print()
    return True


def download_month(year: int, month: int) -> str:
    """
    Download one month of Yellow Taxi data.  Returns the local file path.
    Tries Parquet from the CDN first, then CSV from the legacy S3 bucket.
    """
    os.makedirs(PARQUET_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    tag = f"yellow_tripdata_{year:04d}-{month:02d}"
    parquet_url = f"{TLC_CDN_BASE}/{tag}.parquet"
    csv_url = f"https://nyc-tlc.s3.amazonaws.com/trip+data/{tag}.csv"

    # Try Parquet first
    dest_parquet = os.path.join(PARQUET_DIR, f"{tag}.parquet")
    if os.path.exists(dest_parquet):
        print(f"  [SKIP] {dest_parquet} already exists.")
        return dest_parquet

    print(f"  Downloading {parquet_url} …")
    if _download_file(parquet_url, dest_parquet):
        return dest_parquet

    # Fallback: CSV
    dest_csv = os.path.join(CSV_DIR, f"{tag}.csv")
    if os.path.exists(dest_csv):
        print(f"  [SKIP] {dest_csv} already exists.")
    else:
        print(f"  Parquet not available, trying CSV: {csv_url} …")
        if not _download_file(csv_url, dest_csv):
            raise RuntimeError(f"Could not download {tag} from either URL.")

    # Convert CSV → Parquet
    print(f"  Converting CSV → Parquet …")
    df = pd.read_csv(dest_csv, low_memory=False)
    df = normalise_columns(df)
    df = clean_dataframe(df)
    df.to_parquet(dest_parquet, index=False, engine="pyarrow")
    print(f"  Saved {dest_parquet}")
    return dest_parquet


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the blog-style lowercase names (fare_amt, tip_amt, …)."""
    rename = {c: COLUMN_RENAME[c] for c in df.columns if c in COLUMN_RENAME}
    df = df.rename(columns=rename)
    # Lowercase any remaining columns not in the map
    df.columns = [c.lower() for c in df.columns]
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop obvious bad rows (negative fares, null coords, etc.)."""
    numeric_cols = ["fare_amt", "tip_amt", "trip_distance"]
    for col in numeric_cols:
        if col in df.columns:
            df = df[df[col] >= 0]
    df = df.dropna(subset=[c for c in numeric_cols if c in df.columns])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merge all downloaded months into one Parquet
# ---------------------------------------------------------------------------

def merge_months(year: int, months: list[int], output_name: str = "yellow_taxi.parquet"):
    """Concatenate individual month Parquets into one file for benchmarking."""
    frames = []
    for m in months:
        tag = f"yellow_tripdata_{year:04d}-{m:02d}"
        src = os.path.join(PARQUET_DIR, f"{tag}.parquet")
        if not os.path.exists(src):
            print(f"  [WARN] {src} not found – skipping.")
            continue
        df = pd.read_parquet(src)
        df = normalise_columns(df)
        df = clean_dataframe(df)
        frames.append(df)
        print(f"  Loaded {src}: {len(df):,} rows")

    if not frames:
        raise RuntimeError("No data files found to merge.")

    merged = pd.concat(frames, ignore_index=True)
    out = parquet_path(output_name)
    print(f"\n  Saving merged file → {out}  ({len(merged):,} rows total)")
    merged.to_parquet(out, index=False, engine="pyarrow")
    print("  Done.")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download & prepare NYC Taxi data.")
    parser.add_argument("--year",   type=int, default=TAXI_YEAR,
                        help="Year to download (default: 2009)")
    parser.add_argument("--months", type=int, nargs="+", default=TAXI_MONTHS,
                        help="Months to download (default: 1)")
    parser.add_argument("--output", type=str, default="yellow_taxi.parquet",
                        help="Output Parquet filename")
    args = parser.parse_args()

    print(f"=== Downloading Yellow Taxi {args.year} months {args.months} ===")
    for m in args.months:
        download_month(args.year, m)

    print("\n=== Merging into single Parquet ===")
    out = merge_months(args.year, args.months, args.output)
    print(f"\nBenchmark-ready file: {out}")


if __name__ == "__main__":
    main()
