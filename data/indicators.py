"""
Core indicator computation: Z-score normalization → 3 stress indices.

Pipeline:
  raw_data → z_score() → stress_index() for each module → (liquidity, credit, contagion)
"""

import numpy as np
import pandas as pd

from .sample_data import get_indicator_config

INDICATOR_CONFIG = get_indicator_config()


def z_score_normalize(series: pd.Series, lookback: int = 36) -> pd.Series:
    """
    Rolling Z-score: (x - μ) / σ over a rolling window.

    For each point, μ and σ are computed from the *trailing* window.
    The first `lookback` rows will be NaN (insufficient history).
    """
    roll = series.rolling(window=lookback, min_periods=max(lookback // 2, 6))
    mu = roll.mean()
    sigma = roll.std().replace(0, np.nan)  # avoid divide-by-zero
    z = (series - mu) / sigma
    return z.clip(-3, 3)  # cap outliers


def stress_from_z(z_series: pd.Series, direction: str = "positive") -> pd.Series:
    """
    Map Z-score to [0, 100] stress scale.

    direction='positive': higher raw value → more stress (z>0 → stress>50)
    direction='negative': lower raw value → more stress  (z<0 → stress>50)

    Uses: stress = 50 + sign * z * 16.67  →  roughly [0, 100] for z in [-3, 3]
    """
    sign = 1 if direction == "positive" else -1
    raw = 50 + sign * z_series * 16.67
    return raw.clip(0, 100)


def compute_stress_indices(df: pd.DataFrame, lookback: int = 36) -> pd.DataFrame:
    """
    Compute the 3 module-level stress indices from raw indicator data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw indicator values, indexed by date, one column per indicator.
    lookback : int
        Rolling window size for Z-score calculation (months).

    Returns
    -------
    pd.DataFrame
        Columns: liquidity_stress, credit_stress, contagion_stress
        Index: same dates as input
    """
    results = pd.DataFrame(index=df.index)

    # Per-module accumulator
    module_scores = {"liquidity": [], "credit": [], "contagion": []}

    for col in df.columns:
        if col not in INDICATOR_CONFIG:
            continue

        cfg = INDICATOR_CONFIG[col]
        z = z_score_normalize(df[col], lookback)
        stress = stress_from_z(z, cfg["direction"])

        weighted = stress * cfg["weight"]
        module_scores[cfg["module"]].append(weighted)

    # Sum of weighted scores per module (weights already applied)
    for module, scores in module_scores.items():
        if scores:
            # Stack arrays and sum across indicators
            stacked = np.column_stack([s.values for s in scores])
            # Re-normalize: each module's total = sum of weighted stresses
            # Since weights within each module sum to 1.0, result is in [0, 100]
            results[f"{module}_stress"] = stacked.sum(axis=1)

    return results


def compute_leading_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward-looking leading indicators from raw data.

    Leading indicators detect stress BEFORE it appears in the main modules,
    giving 1–6 month advance warning.

    Indicators:
      - sofr_ois_spread: SOFR - 3M T-Bill spread (funding stress)
      - hy_momentum: 3-month HY spread acceleration (credit cracks)
      - vix_momentum: 3-month VIX acceleration (fear buildup)

    Returns DataFrame with _lead suffix columns.
    """
    result = pd.DataFrame(index=df.index)

    # ── 1. SOFR-OIS proxy: SOFR minus 3M T-Bill ──
    # Widening = bank funding stress. Historically spikes before crises.
    if "sofr" in df.columns and "repo_proxy" in df.columns:
        result["sofr_ois_spread"] = df["sofr"] - df["repo_proxy"]
        # Z-score it
        spread = result["sofr_ois_spread"]
        spread_z = z_score_normalize(spread, lookback=24)
        result["funding_stress_lead"] = stress_from_z(spread_z, "positive")

    # ── 2. HY spread momentum (3-month change) ──
    # Credit market is first to smell trouble.
    if "hy_spread" in df.columns:
        hy_delta = df["hy_spread"].diff(3)  # 3-month change
        hy_delta_z = z_score_normalize(hy_delta, lookback=24)
        result["credit_momentum_lead"] = stress_from_z(hy_delta_z, "positive")

    # ── 3. VIX momentum (3-month change) ──
    # VIX acceleration captures fear before it becomes sustained.
    if "vix" in df.columns:
        vix_delta = df["vix"].diff(3)
        vix_delta_z = z_score_normalize(vix_delta, lookback=24)
        result["fear_momentum_lead"] = stress_from_z(vix_delta_z, "positive")

    # ── 4. Composite leading signal ──
    lead_cols = [c for c in ["funding_stress_lead", "credit_momentum_lead",
                               "fear_momentum_lead"] if c in result.columns]
    if lead_cols:
        result["leading_signal"] = result[lead_cols].mean(axis=1)

    return result


def compute_all_stress(df: pd.DataFrame, lookback: int = 36) -> pd.DataFrame:
    """
    Full pipeline: raw data → stress indices.

    Returns a DataFrame with the original data plus 3 stress columns.
    """
    stress_df = compute_stress_indices(df, lookback)
    lead_df = compute_leading_indicators(df)
    combined = pd.concat([df, stress_df, lead_df], axis=1)
    return combined


def evaluate_alerts(
    stress_df: pd.DataFrame,
    crisis_prob: pd.Series,
    dual_credit: float = 55,
    dual_liquidity: float = 55,
    extreme: float = 75,
    prob_threshold: float = 45,
) -> list[dict]:
    """
    Evaluate alert rules against the latest data point.

    Returns a list of alert dicts with keys: date, type, severity, message.
    """
    alerts = []
    latest_date = stress_df.index[-1]
    latest = stress_df.iloc[-1]
    latest_prob = crisis_prob.iloc[-1]

    # ── Dual stress alert ──
    credit_ok = latest.get("credit_stress", 0) >= dual_credit
    liquidity_ok = latest.get("liquidity_stress", 0) >= dual_liquidity

    if credit_ok and liquidity_ok:
        alerts.append({
            "date": latest_date,
            "type": "⚠️ 双压力信号 Dual Stress",
            "severity": "🔴 HIGH",
            "message": (
                f"信用压力({latest['credit_stress']:.1f}) + "
                f"流动性压力({latest['liquidity_stress']:.1f}) 同时升高 —— "
                f"3-9个月风险窗口可能打开"
            ),
        })

    # ── Extreme single-module alert ──
    for module in ["liquidity", "credit", "contagion"]:
        col = f"{module}_stress"
        val = latest.get(col, 0)
        if val >= extreme:
            labels = {"liquidity": "流动性", "credit": "信用", "contagion": "传染"}
            alerts.append({
                "date": latest_date,
                "type": f"🚨 极端压力 Extreme {labels.get(module, module)}",
                "severity": "🔴 CRITICAL",
                "message": f"{labels.get(module, module)}压力指数 {val:.1f}，超过极端阈值 {extreme}",
            })

    # ── Crisis probability alert ──
    if latest_prob >= prob_threshold:
        alerts.append({
            "date": latest_date,
            "type": "📊 危机概率阈值突破",
            "severity": "🟠 WARNING",
            "message": f"P(危机 3-9个月) = {latest_prob:.1f}%，超过预警线 {prob_threshold}%",
        })

    # ── Trend alerts: 3-month delta ──
    if len(stress_df) >= 3:
        three_months_ago = stress_df.iloc[-4]  # index -4 because -1 is current
        for module in ["liquidity", "credit", "contagion"]:
            col = f"{module}_stress"
            delta = latest[col] - three_months_ago[col]
            if abs(delta) >= 10:
                direction = "↑ 上升" if delta > 0 else "↓ 下降"
                labels = {"liquidity": "流动性", "credit": "信用", "contagion": "传染"}
                alerts.append({
                    "date": latest_date,
                    "type": f"📈 趋势变化 Trend {labels.get(module, module)}",
                    "severity": "🟡 INFO",
                    "message": (
                        f"{labels.get(module, module)}压力 3个月{direction} {abs(delta):.1f} 点"
                    ),
                })

    return alerts
