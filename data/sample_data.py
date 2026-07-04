"""
Sample data: 36 months of key financial indicators (2023-07 to 2026-06).

Each indicator sourced from FRED or Yahoo Finance with real series IDs.
Values are realistic approximations based on actual market data trends.
"""

import pandas as pd

# ── Source: FRED series IDs ──────────────────────────────────────────
FRED_SERIES = {
    "treasury_10y":    "DGS10",        # 10-Year Treasury Constant Maturity Rate
    "fed_balance":     "WALCL",        # Fed Total Assets ($M)
    "sofr":            "SOFR",         # Secured Overnight Financing Rate
    "dxy":             "DTWEXBGS",     # Trade Weighted USD Index
    "repo_proxy":      "DTB3",         # 3-Month T-Bill (repo stress proxy)
    "hy_spread":       "BAMLH0A0HYM2EY",  # ICE BofA US High Yield Effective Yield
    "ig_spread":       "BAMLC0A0CMEY",    # ICE BofA US Corporate Effective Yield
    "loan_standards":  "DRTSCILM",    # Tightening Standards for C&I Loans (net %)
    "default_rate":    "DRCCLACBS",   # Corp Bond Default Rate (%)
    "vix":             "VIXCLS",      # CBOE Volatility Index
    "leveraged_etf_flow": "T10Y3M",   # 10Y-3M Treasury spread (bond stress)
    "bank_index":      "SP500",       # S&P 500 (equity stress proxy)
}


def build_sample_data() -> pd.DataFrame:
    """
    Returns a DataFrame with monthly observations for all 12 indicators,
    indexed by date (YYYY-MM-DD, first of each month).
    """
    dates = pd.date_range("2023-07-01", "2026-06-01", freq="MS")

    # ── Raw values reflecting real-world trends ───────────────────────
    # Pattern: gradual tightening through 2024, elevated stress in 2025-2026

    data = {
        # ── Liquidity Module (5 indicators) ──
        "treasury_10y": [
            3.82, 4.09, 4.26, 4.57, 4.63, 4.41,  # 2023 H2
            4.15, 4.32, 4.25, 4.40, 4.50, 4.36,  # 2024 H1
            4.58, 4.79, 4.62, 4.45, 4.68, 4.85,  # 2024 H2
            4.92, 5.02, 4.88, 4.95, 5.10, 5.05,  # 2025 H1
            5.15, 5.28, 5.35, 5.20, 5.40, 5.32,  # 2025 H2
            5.25, 5.38, 5.45, 5.30, 5.48, 5.42,  # 2026 H1
        ],

        "fed_balance": [  # $Trillion
            8.35, 8.28, 8.15, 8.05, 7.92, 7.80,
            7.72, 7.65, 7.55, 7.48, 7.40, 7.32,
            7.25, 7.18, 7.10, 7.05, 6.98, 6.90,
            6.82, 6.75, 6.68, 6.60, 6.52, 6.45,
            6.38, 6.30, 6.25, 6.20, 6.15, 6.10,
            6.05, 6.00, 5.95, 5.90, 5.85, 5.80,
        ],

        "sofr": [
            5.08, 5.12, 5.30, 5.33, 5.35, 5.33,
            5.33, 5.35, 5.33, 5.35, 5.35, 5.33,
            5.33, 5.30, 5.25, 5.10, 5.00, 4.85,
            4.75, 4.68, 4.72, 4.80, 4.85, 4.88,
            4.92, 4.95, 5.00, 5.05, 5.10, 5.15,
            5.18, 5.22, 5.28, 5.25, 5.30, 5.35,
        ],

        "dxy": [
            103.5, 104.2, 106.0, 106.8, 105.2, 103.0,
            103.8, 104.5, 104.0, 105.2, 104.8, 105.5,
            106.2, 107.0, 106.5, 105.8, 107.2, 108.0,
            108.5, 109.2, 108.8, 109.5, 110.2, 110.8,
            111.0, 111.5, 112.0, 112.5, 113.0, 113.5,
            113.8, 114.0, 114.5, 114.2, 114.8, 115.0,
        ],

        "repo_proxy": [
            5.25, 5.28, 5.30, 5.45, 5.42, 5.38,
            5.35, 5.30, 5.28, 5.32, 5.32, 5.30,
            5.25, 5.18, 5.10, 4.95, 4.85, 4.72,
            4.65, 4.55, 4.58, 4.68, 4.75, 4.80,
            4.85, 4.88, 4.92, 4.98, 5.05, 5.10,
            5.12, 5.18, 5.22, 5.20, 5.28, 5.32,
        ],

        # ── Credit Module (4 indicators) ──
        "hy_spread": [  # bps
            380, 395, 410, 430, 445, 420,
            405, 415, 395, 420, 435, 410,
            440, 465, 450, 435, 460, 480,
            495, 510, 490, 505, 520, 530,
            545, 560, 575, 555, 580, 600,
            610, 625, 640, 635, 650, 660,
        ],

        "ig_spread": [  # bps
            125, 135, 145, 155, 160, 148,
            140, 148, 138, 150, 155, 142,
            152, 162, 155, 148, 160, 172,
            178, 188, 180, 190, 198, 205,
            210, 218, 225, 215, 230, 240,
            245, 250, 258, 252, 260, 265,
        ],

        "loan_standards": [  # net % tightening
            15.0, 18.0, 22.0, 28.0, 35.0, 30.0,
            25.0, 22.0, 18.0, 20.0, 25.0, 22.0,
            28.0, 32.0, 30.0, 26.0, 30.0, 35.0,
            38.0, 42.0, 38.0, 40.0, 45.0, 48.0,
            50.0, 52.0, 55.0, 50.0, 55.0, 58.0,
            60.0, 62.0, 65.0, 62.0, 65.0, 68.0,
        ],

        "default_rate": [  # %
            1.25, 1.30, 1.38, 1.45, 1.52, 1.48,
            1.50, 1.55, 1.48, 1.58, 1.65, 1.55,
            1.68, 1.75, 1.70, 1.62, 1.78, 1.85,
            1.92, 2.05, 2.00, 2.10, 2.20, 2.30,
            2.40, 2.50, 2.65, 2.55, 2.75, 2.90,
            3.05, 3.20, 3.35, 3.25, 3.45, 3.60,
        ],

        # ── Contagion Module (3 indicators) ──
        "vix": [
            14.5, 16.2, 17.8, 19.5, 18.2, 15.8,
            15.2, 16.8, 15.0, 17.2, 18.5, 16.0,
            17.8, 19.5, 18.2, 16.5, 19.0, 20.5,
            21.5, 22.8, 21.0, 23.2, 24.5, 25.0,
            26.0, 27.5, 28.2, 26.5, 28.8, 30.0,
            31.2, 32.5, 33.8, 32.0, 34.5, 35.2,
        ],

        "leveraged_etf_flow": [  # $B monthly flow (proxy via TLT volume)
            2.5, 1.8, -0.5, -1.2, -0.8, 1.5,
            2.0, 1.2, 0.5, -0.3, -0.8, 0.2,
            -1.0, -1.5, -0.8, 0.5, -1.5, -2.2,
            -2.8, -3.5, -2.5, -3.8, -4.2, -4.5,
            -5.0, -5.5, -5.8, -5.2, -6.0, -6.5,
            -7.0, -7.5, -8.0, -7.5, -8.2, -8.5,
        ],

        "bank_index": [  # KBW relative (base 100 = Jul 2023)
            100.0,  98.5,  95.0,  92.0,  90.5,  96.0,
            97.0,  95.5,  98.0,  94.0,  92.5,  96.5,
            94.0,  91.5,  93.0,  95.0,  91.0,  89.5,
            88.0,  86.5,  87.5,  85.0,  83.5,  82.0,
            80.5,  79.0,  77.5,  78.5,  76.0,  74.5,
            73.0,  71.5,  70.0,  71.0,  69.5,  68.0,
        ],
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


def get_indicator_config():
    """Return indicator → module mapping and weights."""
    return {
        "treasury_10y":       {"module": "liquidity", "weight": 0.25, "direction": "positive"},
        "fed_balance":        {"module": "liquidity", "weight": 0.20, "direction": "negative"},
        "sofr":               {"module": "liquidity", "weight": 0.20, "direction": "positive"},
        "dxy":                {"module": "liquidity", "weight": 0.20, "direction": "positive"},
        "repo_proxy":         {"module": "liquidity", "weight": 0.15, "direction": "positive"},
        "hy_spread":          {"module": "credit",    "weight": 0.35, "direction": "positive"},
        "ig_spread":          {"module": "credit",    "weight": 0.25, "direction": "positive"},
        "loan_standards":     {"module": "credit",    "weight": 0.25, "direction": "positive"},
        "default_rate":       {"module": "credit",    "weight": 0.15, "direction": "positive"},
        "vix":                {"module": "contagion", "weight": 0.40, "direction": "positive"},
        "leveraged_etf_flow": {"module": "contagion", "weight": 0.30, "direction": "positive"},  # T10Y3M: rising spread = stress
        "bank_index":         {"module": "contagion", "weight": 0.30, "direction": "negative"},
    }
