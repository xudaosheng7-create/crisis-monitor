"""
Live data pipeline: FRED + Yahoo Finance → monthly DataFrame.

Usage:
    pipeline = LiveDataPipeline(fred_api_key="...")
    df = pipeline.fetch_all()        # returns same format as sample_data.build_sample_data()
    pipeline.update()                 # fetch latest and cache

If no FRED API key is provided, fetch_all() returns None (caller should fall back to sample data).
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

from .fetcher import FREDFetcher, YahooFetcher
from .cache import DataCache
from .sample_data import build_sample_data

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Indicator → data source mapping
# ═══════════════════════════════════════════════════════════════════════

# Each entry: (source, series_id_or_ticker, [column_for_yahoo], [frequency_override])
INDICATOR_MAP: Dict[str, tuple] = {
    # ── Liquidity (5) ──
    "treasury_10y":       ("fred", "DGS10"),
    "fed_balance":        ("fred", "WALCL"),
    "sofr":               ("fred", "SOFR"),
    "dxy":                ("fred", "DTWEXBGS"),
    "repo_proxy":         ("fred", "DTB3"),

    # ── Credit (4) ──
    "hy_spread":          ("fred", "BAMLH0A0HYM2EY"),  # HY effective yield (%)
    "ig_spread":          ("fred", "BAMLC0A0CMEY"),    # IG effective yield (%)
    "loan_standards":     ("fred", "DRTSCILM"),        # quarterly
    "default_rate":       ("fred", "DRCCLACBS"),       # quarterly

    # ── Contagion (3) ──
    "vix":                ("fred", "VIXCLS"),
    "leveraged_etf_flow": ("fred", "T10Y3M"),       # 10Y-3M spread: bond market stress
    "bank_index":         ("fred", "SP500"),         # S&P 500 as equity stress proxy
}

# Series that are quarterly — will be forward-filled to monthly
QUARTERLY_SERIES = {"DRTSCILM", "DRCCLACBS"}

# How far back to fetch (years)
LOOKBACK_YEARS = 5


class LiveDataPipeline:
    """
    Orchestrates fetching all 12 indicators from FRED and Yahoo Finance,
    resamples to monthly frequency, and caches results in SQLite.
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_path: Optional[Path] = None,
    ):
        self.fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
        self.cache = DataCache(cache_path)
        self.fred = FREDFetcher(self.fred_api_key) if self.fred_api_key else None
        self.yahoo = YahooFetcher()

    @property
    def is_available(self) -> bool:
        """Check if live data is configured."""
        return bool(self.fred_api_key)

    # ── Core fetch ─────────────────────────────────────────────────

    def fetch_all(
        self,
        use_cache: bool = True,
        max_cache_hours: int = 24,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch all 12 indicators and return a monthly DataFrame.

        Parameters
        ----------
        use_cache : bool
            If True, check cache first and only refetch stale series.
        max_cache_hours : int
            Maximum age of cached data before re-fetching.

        Returns
        -------
        pd.DataFrame or None
            Monthly data indexed by date, one column per indicator.
            Returns None if live data is not configured.
        """
        if not self.is_available:
            return None

        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")

        monthly_series: Dict[str, pd.Series] = {}
        fetch_status: List[dict] = []

        for indicator, cfg in INDICATOR_MAP.items():
            source = cfg[0]
            series_id = cfg[1]

            try:
                # Check cache
                if use_cache:
                    cached = self.cache.load_series(series_id)
                    last_fetch = self.cache.last_fetch(series_id)
                    if not cached.empty and last_fetch:
                        last_dt = datetime.fromisoformat(last_fetch)
                        age_hours = (datetime.utcnow() - last_dt).total_seconds() / 3600
                        if age_hours < max_cache_hours:
                            monthly = self._to_monthly(pd.DataFrame(cached), series_id)
                            if not monthly.empty:
                                monthly_series[indicator] = monthly
                                fetch_status.append({"indicator": indicator, "series": series_id,
                                                     "source": "cache", "rows": len(monthly)})
                                continue

                # Fetch live
                if source == "fred":
                    raw = self.fred.fetch(series_id, start=start_date, end=end_date)
                elif source == "yahoo":
                    col = cfg[2] if len(cfg) > 2 else "Close"
                    raw = self._fetch_yahoo_with_retry(series_id, start_date, end_date, col)
                else:
                    continue

                if raw.empty:
                    # Try cache as fallback even if stale
                    cached = self.cache.load_series(series_id)
                    if not cached.empty:
                        monthly = self._to_monthly(pd.DataFrame(cached), series_id)
                        monthly_series[indicator] = monthly
                        fetch_status.append({"indicator": indicator, "series": series_id,
                                             "source": "stale_cache", "rows": len(monthly)})
                    else:
                        fetch_status.append({"indicator": indicator, "series": series_id,
                                             "source": "failed", "rows": 0})
                    continue

                # Cache and convert
                self.cache.store_series(series_id, raw, "value")
                monthly = self._to_monthly(raw, series_id)
                if not monthly.empty:
                    monthly_series[indicator] = monthly
                    fetch_status.append({"indicator": indicator, "series": series_id,
                                         "source": "live", "rows": len(monthly)})

            except Exception as e:
                logger.warning(f"Failed to fetch {indicator} ({series_id}): {e}")
                # Fall back to cache
                cached = self.cache.load_series(series_id)
                if not cached.empty:
                    monthly = self._to_monthly(pd.DataFrame(cached), series_id)
                    monthly_series[indicator] = monthly
                    fetch_status.append({"indicator": indicator, "series": series_id,
                                         "source": "fallback_cache", "rows": len(monthly)})
                else:
                    fetch_status.append({"indicator": indicator, "series": series_id,
                                         "source": "failed", "rows": 0})

        # ── Build unified DataFrame ──
        if not monthly_series:
            logger.error("No indicators fetched successfully.")
            return None

        df = pd.DataFrame(monthly_series)
        df.index.name = "date"
        df = df.sort_index()

        # Drop rows where ALL indicators are NaN
        df = df.dropna(how="all")

        # Forward-fill quarterly series gaps
        for indicator in monthly_series:
            series_id = INDICATOR_MAP.get(indicator, ("", ""))[1]
            if series_id in QUARTERLY_SERIES:
                df[indicator] = df[indicator].ffill()

        # Interpolate small gaps (up to 2 consecutive NaN) for daily/weekly series
        df = df.interpolate(method="linear", limit=2)

        self._last_status = fetch_status
        return df

    # ── Yahoo retry ────────────────────────────────────────────────

    def _fetch_yahoo_with_retry(
        self, ticker: str, start: str, end: str, column: str,
        max_retries: int = 3, base_delay: float = 2.0,
    ):
        """Fetch from Yahoo with exponential backoff on rate limits."""
        for attempt in range(max_retries):
            try:
                # Stagger Yahoo calls to avoid rate limiting
                if attempt > 0:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                return self.yahoo.fetch(ticker, start=start, end=end, column=column)
            except Exception as e:
                msg = str(e)
                if "Rate" in msg or "Too Many" in msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** (attempt + 1))
                        logger.warning(f"Yahoo rate limited on {ticker}, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                raise

    # ── Monthly resampling ──────────────────────────────────────────

    def _to_monthly(self, raw: pd.DataFrame, series_id: str) -> pd.Series:
        """
        Resample a raw time series to monthly frequency.

        Uses month-start frequency for consistency with sample_data.
        """
        if raw.empty:
            return pd.Series(dtype=float)

        val_col = "value" if "value" in raw.columns else raw.columns[0]
        series = raw[val_col].copy()
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()

        # Drop NaN and inf
        series = series.replace([np.inf, -np.inf], np.nan).dropna()

        if series.empty:
            return pd.Series(dtype=float)

        # Resample to month-start
        monthly = series.resample("MS").last()
        monthly.name = val_col

        return monthly

    # ── Status & metadata ──────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return fetch status for the last run."""
        if not hasattr(self, "_last_status"):
            return {"mode": "unknown", "indicators_fetched": 0, "details": []}

        live_count = sum(1 for s in self._last_status if s["source"] == "live")
        cache_count = sum(1 for s in self._last_status if "cache" in s["source"])
        failed = sum(1 for s in self._last_status if s["source"] == "failed")

        return {
            "mode": "live" if self.is_available else "sample",
            "indicators_total": len(self._last_status),
            "live": live_count,
            "cached": cache_count,
            "failed": failed,
            "details": self._last_status,
        }

    def update(self) -> Optional[pd.DataFrame]:
        """Force a fresh fetch, bypassing cache."""
        return self.fetch_all(use_cache=False)
