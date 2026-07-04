"""
FRED API and Yahoo Finance data fetchers.

Used in Phase 2 for live data fetching.
For MVP, sample_data.py provides all data.

Supported sources:
  - FRED (Federal Reserve Economic Data): requires API key
  - Yahoo Finance: no key required (rate-limited)
"""

from typing import Optional
from datetime import datetime

import pandas as pd


class FREDFetcher:
    """
    Fetch economic data from the FRED API.

    Usage:
        fetcher = FREDFetcher(api_key="your_key")
        df = fetcher.fetch("DGS10", start="2020-01-01")
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(
        self,
        series_id: str,
        start: str = "2020-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch a FRED series. Returns DataFrame with date index and 'value' column.
        """
        import urllib.request
        import json

        end = end or datetime.today().strftime("%Y-%m-%d")
        url = (
            f"{self.BASE_URL}?series_id={series_id}"
            f"&api_key={self.api_key}"
            f"&observation_start={start}"
            f"&observation_end={end}"
            f"&file_type=json"
        )

        with urllib.request.urlopen(url) as resp:
            raw = json.loads(resp.read())

        records = []
        for obs in raw.get("observations", []):
            val = obs["value"]
            if val == ".":
                continue  # FRED uses "." for missing
            records.append({"date": obs["date"], "value": float(val)})

        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()


class YahooFetcher:
    """
    Fetch market data from Yahoo Finance.

    Usage:
        fetcher = YahooFetcher()
        df = fetcher.fetch("^VIX", start="2020-01-01")
    """

    def fetch(
        self,
        ticker: str,
        start: str = "2020-01-01",
        end: Optional[str] = None,
        column: str = "Close",
    ) -> pd.DataFrame:
        """
        Fetch historical prices. Returns DataFrame with date index and one value column.
        """
        import yfinance as yf

        end = end or datetime.today().strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

        if data.empty:
            return pd.DataFrame()

        series = data[column] if column in data.columns else data.iloc[:, 0]
        return series.rename("value").to_frame()
