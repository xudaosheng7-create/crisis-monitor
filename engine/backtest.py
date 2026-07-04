"""
Backtesting framework: walk-forward evaluation against historical crises.

Known crisis periods used for validation:
  - 2000-2002: Dot-com bubble burst
  - 2007-2009: Global Financial Crisis
  - 2020-03: COVID-19 crash
  - 2023-03: Regional bank crisis (SVB, etc.)

The backtest runs the model on historical data and checks:
  1. Did the model signal BEFORE the crisis? (lead time)
  2. What was the peak probability?
  3. Any false positives in calm periods?
"""

from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from engine.scoring import (
    crisis_probability_series,
    classify_risk,
    RISK_LEVELS,
)

# Known crisis windows for validation
CRISIS_PERIODS = [
    {
        "name": "2000 Dot-com Bubble",
        "start": "2000-03-01",
        "peak": "2000-09-01",
        "end": "2002-10-01",
        "type": "资产泡沫破裂",
    },
    {
        "name": "2008 Global Financial Crisis",
        "start": "2007-08-01",
        "peak": "2008-09-15",
        "end": "2009-03-01",
        "type": "信用 + 流动性双杀",
    },
    {
        "name": "2020 COVID-19 Crash",
        "start": "2020-02-15",
        "peak": "2020-03-16",
        "end": "2020-04-30",
        "type": "流动性瞬间冻结",
    },
    {
        "name": "2023 Banking Crisis",
        "start": "2023-03-08",
        "peak": "2023-03-13",
        "end": "2023-05-01",
        "type": "利率冲击 + 银行资产错配",
    },
]


def run_backtest(
    stress_df: pd.DataFrame,
    offset: float = 50.0,
    scale: float = 10.0,
    alert_threshold: float = 45.0,
) -> pd.DataFrame:
    """
    Run a walk-forward backtest on historical stress data.

    Parameters
    ----------
    stress_df : pd.DataFrame
        Columns: liquidity_stress, credit_stress, contagion_stress.
        Index: datetime.
    offset, scale : float
        Sigmoid parameters.
    alert_threshold : float
        Probability threshold for generating an alert.

    Returns
    -------
    pd.DataFrame
        Same index, with added columns:
        - crisis_probability
        - risk_level
        - alert (bool)
        - composite_stress
    """
    df = stress_df.copy()

    # Compute crisis probability
    df["crisis_probability"] = crisis_probability_series(df, offset, scale)

    # Classify risk level
    df["risk_level"] = df["crisis_probability"].apply(
        lambda p: classify_risk(p)[0]
    )

    # Composite stress (for reference)
    df["composite_stress"] = (
        0.40 * df["liquidity_stress"]
        + 0.35 * df["credit_stress"]
        + 0.25 * df["contagion_stress"]
    )

    # Alert signal
    df["alert"] = df["crisis_probability"] >= alert_threshold

    return df


def find_signal_windows(
    backtest_df: pd.DataFrame,
    min_consecutive: int = 2,
) -> List[Dict]:
    """
    Find periods where the model generated sustained alert signals.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output from run_backtest().
    min_consecutive : int
        Minimum consecutive months of alerts to count as a signal.

    Returns
    -------
    List of signal dicts: {start, end, peak_prob, peak_date, duration_months}
    """
    signals = []
    in_signal = False
    signal_start = None
    signal_probs = []

    for date, row in backtest_df.iterrows():
        if row["alert"] and not in_signal:
            in_signal = True
            signal_start = date
            signal_probs = [row["crisis_probability"]]
        elif row["alert"] and in_signal:
            signal_probs.append(row["crisis_probability"])
        elif not row["alert"] and in_signal:
            duration = (date - signal_start).days / 30.0
            if len(signal_probs) >= min_consecutive:
                peak_idx = np.argmax(signal_probs)
                signals.append({
                    "start": signal_start,
                    "end": date,
                    "duration_months": round(duration, 1),
                    "peak_prob": round(max(signal_probs), 1),
                    "peak_date": backtest_df.index[
                        backtest_df.index.get_loc(signal_start) + peak_idx
                    ],
                    "avg_prob": round(np.mean(signal_probs), 1),
                })
            in_signal = False
            signal_start = None
            signal_probs = []

    # Handle signal still active at end
    if in_signal and len(signal_probs) >= min_consecutive:
        peak_idx = np.argmax(signal_probs)
        signals.append({
            "start": signal_start,
            "end": backtest_df.index[-1],
            "duration_months": round((backtest_df.index[-1] - signal_start).days / 30.0, 1),
            "peak_prob": round(max(signal_probs), 1),
            "peak_date": backtest_df.index[
                backtest_df.index.get_loc(signal_start) + peak_idx
            ],
            "avg_prob": round(np.mean(signal_probs), 1),
        })

    return signals


def evaluate_crisis_match(
    signals: List[Dict],
    crisis_periods: Optional[List[Dict]] = None,
    lead_months: int = 6,
) -> List[Dict]:
    """
    Check how many known crisis periods were preceded by model signals.

    A "hit" = at least one signal started within `lead_months` before the crisis.

    Returns a list of matches with lead time info.
    """
    if crisis_periods is None:
        crisis_periods = CRISIS_PERIODS

    results = []
    for crisis in crisis_periods:
        crisis_start = pd.Timestamp(crisis["start"])
        matched = None
        best_lead = float("inf")

        for sig in signals:
            sig_start = pd.Timestamp(sig["start"])
            lead = (crisis_start - sig_start).days / 30.0
            if 0 <= lead <= lead_months and lead < best_lead:
                best_lead = lead
                matched = sig

        if matched:
            results.append({
                "crisis": crisis["name"],
                "type": crisis["type"],
                "crisis_start": crisis["start"],
                "detected": True,
                "lead_months": round(best_lead, 1),
                "signal_peak_prob": matched["peak_prob"],
                "signal_start": str(matched["start"].date()),
            })
        else:
            results.append({
                "crisis": crisis["name"],
                "type": crisis["type"],
                "crisis_start": crisis["start"],
                "detected": False,
                "lead_months": None,
                "signal_peak_prob": None,
                "signal_start": None,
            })

    return results


def backtest_summary(
    stress_df: pd.DataFrame,
    crisis_periods: Optional[List[Dict]] = None,
) -> Dict:
    """
    Run a full backtest and return a summary report.

    Returns
    -------
    Dict with keys:
        - current: latest risk level info
        - signals: list of signal windows found
        - crisis_matches: evaluation against known crises
        - summary_stats: detection rate, avg lead time, etc.
    """
    bt_df = run_backtest(stress_df)
    signals = find_signal_windows(bt_df)
    matches = evaluate_crisis_match(signals, crisis_periods)

    detected = [m for m in matches if m["detected"]]
    detection_rate = len(detected) / len(matches) * 100 if matches else 0
    avg_lead = (
        np.mean([m["lead_months"] for m in detected])
        if detected
        else 0
    )

    return {
        "current": {
            "date": str(bt_df.index[-1].date()),
            "probability": round(bt_df["crisis_probability"].iloc[-1], 1),
            "risk_level": bt_df["risk_level"].iloc[-1],
            "liquidity_stress": round(bt_df["liquidity_stress"].iloc[-1], 1),
            "credit_stress": round(bt_df["credit_stress"].iloc[-1], 1),
            "contagion_stress": round(bt_df["contagion_stress"].iloc[-1], 1),
        },
        "signals": signals,
        "crisis_matches": matches,
        "summary_stats": {
            "total_crises": len(matches),
            "detected": len(detected),
            "detection_rate": round(detection_rate, 1),
            "avg_lead_months": round(avg_lead, 1),
            "total_signals": len(signals),
        },
    }
