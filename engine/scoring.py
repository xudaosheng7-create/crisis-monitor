"""
Crisis probability model.

Converts 3 stress indices → single crisis probability using a sigmoid (logistic) function.

Formula:
    composite = 0.40 * liquidity_stress
              + 0.35 * credit_stress
              + 0.25 * contagion_stress

    P(Crisis 3-9mo) = σ(composite - 50) * 100

where σ(x) = 1 / (1 + e^(-x / 10))

The offset of -50 means: when all 3 modules are at 50 (neutral), probability is 50%.
The divisor 10 controls steepness — changing a module by 10 points shifts probability by ~25%.
"""

from typing import Tuple

import numpy as np
import pandas as pd

# Module weights (must match settings.yaml)
LIQUIDITY_WEIGHT = 0.40
CREDIT_WEIGHT = 0.35
CONTAGION_WEIGHT = 0.25

# Risk level thresholds
RISK_LEVELS = [
    (20, "🟢 正常 Normal", "#2ECC40"),
    (40, "🟡 注意 Watch", "#FFDC00"),
    (60, "🟠 脆弱 Fragile", "#FF851B"),
    (100, "🔴 危险 Danger", "#FF4136"),
]


def sigmoid(x: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """Sigmoid (logistic) function: 1 / (1 + e^(-x/scale))."""
    return 1.0 / (1.0 + np.exp(-x / scale))


def composite_stress(liquidity: float, credit: float, contagion: float) -> float:
    """Weighted composite stress score."""
    return (
        LIQUIDITY_WEIGHT * liquidity
        + CREDIT_WEIGHT * credit
        + CONTAGION_WEIGHT * contagion
    )


def crisis_probability(
    liquidity: float,
    credit: float,
    contagion: float,
    offset: float = 50.0,
    scale: float = 10.0,
) -> float:
    """
    Compute P(Crisis 3-9 months) from 3 stress indices.

    Parameters
    ----------
    liquidity, credit, contagion : float
        Stress indices in [0, 100].
    offset : float
        Shift parameter. Higher = lower probabilities overall.
    scale : float
        Steepness parameter. Lower = sharper transition.

    Returns
    -------
    float
        Probability in [0, 100].
    """
    comp = composite_stress(liquidity, credit, contagion)
    prob = sigmoid(np.array([comp - offset]), scale)[0]
    return float(prob * 100)


def crisis_probability_series(
    df: pd.DataFrame,
    offset: float = 50.0,
    scale: float = 10.0,
) -> pd.Series:
    """
    Compute crisis probability for an entire time series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: liquidity_stress, credit_stress, contagion_stress.
    """
    comp = (
        LIQUIDITY_WEIGHT * df["liquidity_stress"]
        + CREDIT_WEIGHT * df["credit_stress"]
        + CONTAGION_WEIGHT * df["contagion_stress"]
    )
    prob = sigmoid((comp - offset).values, scale) * 100
    return pd.Series(prob, index=df.index, name="crisis_probability")


def classify_risk(probability: float) -> Tuple[str, str, str]:
    """
    Classify a crisis probability into risk level.

    Returns
    -------
    (label, emoji+name, hex_color)
    """
    for threshold, label, color in RISK_LEVELS:
        if probability <= threshold:
            return label, label.split(" ", 1)[1] if " " in label else label, color
    return RISK_LEVELS[-1][0], RISK_LEVELS[-1][1], RISK_LEVELS[-1][2]


def get_risk_level(probability: float) -> dict:
    """Return full risk level info as a dict."""
    label, name, color = classify_risk(probability)
    return {
        "probability": round(probability, 1),
        "status_label": label,
        "status_name": name,
        "color": color,
    }


# ═══════════════════════════════════════════════════════════════════════
# Regime Detection — simple rule-based market phase classifier
# ═══════════════════════════════════════════════════════════════════════

REGIMES = [
    {
        "name": "📗 扩张 Expansion",
        "color": "#2ECC40",
        "desc": "系统稳定，流动性充裕，信用环境宽松，波动率正常。维持正常风险配置。",
    },
    {
        "name": "📒 后周期 Late Cycle",
        "color": "#FFDC00",
        "desc": "部分指标开始承压但整体可控。关注边际变化，不宜激进加仓。",
    },
    {
        "name": "📙 压力积累 Stress Build-up",
        "color": "#FF851B",
        "desc": "多模块同步恶化，系统性风险窗口打开。建议降低风险敞口，增加对冲。",
    },
    {
        "name": "📕 危机 Crisis",
        "color": "#FF4136",
        "desc": "系统面临断裂风险，流动性枯竭或信用冻结正在进行。防御优先。",
    },
]


def detect_regime(
    liquidity: float,
    credit: float,
    contagion: float,
    leading_signal: float | None = None,
    probability: float | None = None,
) -> dict:
    """
    Rule-based market regime classifier.

    Logic (ordered, first match wins):
      Crisis:        2+ modules > 70, or probability > 60%
      Stress Build:  2+ modules > 50, or leading_signal > 60
      Late Cycle:    any module > 30, or leading_signal > 40
      Expansion:     everything else
    """
    modules_above = lambda t: sum([
        1 if liquidity > t else 0,
        1 if credit > t else 0,
        1 if contagion > t else 0,
    ])

    lead = leading_signal if leading_signal is not None else 0
    prob = probability if probability is not None else 0

    if modules_above(70) >= 2 or prob > 60:
        return REGIMES[3]  # Crisis

    if modules_above(50) >= 2 or lead > 60:
        return REGIMES[2]  # Stress Build-up

    if modules_above(30) >= 1 or lead > 40:
        return REGIMES[1]  # Late Cycle

    return REGIMES[0]  # Expansion


def get_regime_series(df: pd.DataFrame) -> pd.Series:
    """
    Compute regime label for each row of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: liquidity_stress, credit_stress, contagion_stress.
        Optional: leading_signal, crisis_probability.

    Returns
    -------
    pd.Series of regime dicts.
    """
    regimes = []
    for i in range(len(df)):
        row = df.iloc[i]
        r = detect_regime(
            liquidity=float(row["liquidity_stress"]),
            credit=float(row["credit_stress"]),
            contagion=float(row["contagion_stress"]),
            leading_signal=float(row.get("leading_signal", 0)),
            probability=float(row.get("crisis_probability", 0)),
        )
        regimes.append(r["name"])
    return pd.Series(regimes, index=df.index, name="regime")
