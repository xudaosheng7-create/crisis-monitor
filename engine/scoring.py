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


# ═══════════════════════════════════════════════════════════════════════
# Action Recommendation Layer
# ═══════════════════════════════════════════════════════════════════════

# Posture levels
POSTURES = {
    1: {"name": "正常配置 Normal", "color": "#2ECC40",
        "desc": "系统稳定，维持战略资产配置"},
    2: {"name": "谨慎 Cautious", "color": "#FFDC00",
        "desc": "边际收紧，审查风险敞口，削减高风险暴露"},
    3: {"name": "防御 Defensive", "color": "#FF851B",
        "desc": "显著降风险：减仓权益、缩短久期、增持现金"},
    4: {"name": "保护 Protective", "color": "#FF4136",
        "desc": "资本保全优先：最低权益敞口、避险资产为主"},
    5: {"name": "危机应对 Crisis", "color": "#8B0000",
        "desc": "流动性为王：现金+国债，等待系统性风险释放"},
}


def get_action_plan(
    probability: float,
    regime_name: str,
    liquidity: float,
    credit: float,
    contagion: float,
    trend_delta: float = 0.0,
) -> dict:
    """
    Generate risk posture and action recommendations.

    Parameters
    ----------
    probability : float
        P(Crisis 3-9mo) in [0, 100].
    regime_name : str
        Regime label from detect_regime().
    liquidity, credit, contagion : float
        Current stress indices in [0, 100].
    trend_delta : float
        Month-over-month change in probability (positive = worsening).

    Returns
    -------
    dict with keys: posture, posture_level, asset_allocation, actions, lead_driver
    """
    # ── Determine posture level ──
    if probability >= 60:
        posture_level = 5
    elif probability >= 45:
        posture_level = 4
    elif probability >= 25:
        posture_level = 3 if trend_delta > 3 else 2
    elif probability >= 10:
        posture_level = 2
    else:
        posture_level = 1

    posture = POSTURES[posture_level]

    # ── Determine leading stress driver ──
    drivers = [
        ("credit", credit, "信用收缩 Credit Tightening"),
        ("liquidity", liquidity, "流动性收紧 Liquidity Squeeze"),
        ("contagion", contagion, "市场传染 Market Contagion"),
    ]
    drivers.sort(key=lambda x: x[1], reverse=True)
    lead_module, lead_value, lead_label = drivers[0]

    # ── Generate actions based on posture + driver ──
    base_actions = {
        1: [
            "维持战略资产配置，无需调整",
            "定期再平衡，保持目标权重",
            "关注信用利差和流动性指标的边际变化",
        ],
        2: [
            "削减高 beta 权益敞口（科技股、小盘股）",
            "缩短固定收益久期，降低利率敏感度",
            "检查组合流动性，确保可快速减仓",
            "增持投资级信用债，减持垃圾债",
        ],
        3: [
            "权益仓位降至配置下限，优先卖出周期股",
            "信用敞口集中于 AA 及以上，完全退出高收益债",
            "现金比例提升至 15-25%",
            "增加黄金/美债等避险资产配置",
            "审查衍生品对冲有效性",
        ],
        4: [
            "权益敞口降至最低（≤ 配置下限的 50%）",
            "清仓高收益债和杠杆贷款",
            "现金+短期国债 ≥ 组合的 30%",
            "增持美元现金、黄金、长期美债",
            "暂停所有新增风险敞口",
            "设置止损/尾部风险对冲",
        ],
        5: [
            "资本保全为唯一目标",
            "权益敞口接近零或已对冲",
            "持有现金 + 短期国债为主（≥ 50%）",
            "避免所有信用风险敞口",
            "等待体制切换回 Stress Build-up 以下再考虑入场",
            "监控美联储/央行紧急政策信号",
        ],
    }

    actions = base_actions.get(posture_level, base_actions[1])

    # ── Driver-specific actions ──
    driver_actions = {
        "credit": [
            f"领先压力源: {lead_label} ({lead_value:.0f}/100)",
            "信用危机通常发展较慢（3-6个月窗口），有减仓时间",
            "重点监控 HY 有效收益率和银行贷款标准",
        ],
        "liquidity": [
            f"领先压力源: {lead_label} ({lead_value:.0f}/100)",
            "流动性危机发展极快（数天至数周），需快速反应",
            "确保现金储备充足，避免流动性错配",
        ],
        "contagion": [
            f"领先压力源: {lead_label} ({lead_value:.0f}/100)",
            "传染阶段通常已接近或进入危机，防御优先",
            "关注 VIX > 30 的持续时间和银行股表现",
        ],
    }

    driver_info = driver_actions.get(lead_module, driver_actions["credit"])

    # ── Trend warning ──
    trend_warning = ""
    if trend_delta > 10:
        trend_warning = f"⚠️ 风险快速恶化中（月增 {trend_delta:.0f}%），建议加速调整"
    elif trend_delta > 5:
        trend_warning = f"⚡ 风险边际上升（月增 {trend_delta:.0f}%），保持警惕"
    elif trend_delta < -5:
        trend_warning = f"✅ 风险边际改善（月降 {abs(trend_delta):.0f}%），可考虑逐步回归"

    # ── Asset allocation guide ──
    allocations = {
        1: {"权益 Equities": "标配", "投资级信用 IG": "标配", "高收益 HY": "标配",
            "现金 Cash": "低配", "黄金/美债 Safe Haven": "低配"},
        2: {"权益 Equities": "标配偏轻", "投资级信用 IG": "标配", "高收益 HY": "低配",
            "现金 Cash": "标配", "黄金/美债 Safe Haven": "标配"},
        3: {"权益 Equities": "低配", "投资级信用 IG": "标配偏轻", "高收益 HY": "回避",
            "现金 Cash": "超配", "黄金/美债 Safe Haven": "超配"},
        4: {"权益 Equities": "最低", "投资级信用 IG": "低配", "高收益 HY": "清仓",
            "现金 Cash": "重仓", "黄金/美债 Safe Haven": "重仓"},
        5: {"权益 Equities": "清仓", "投资级信用 IG": "回避", "高收益 HY": "清仓",
            "现金 Cash": "绝对重仓", "黄金/美债 Safe Haven": "重仓"},
    }

    return {
        "posture": posture,
        "posture_level": posture_level,
        "asset_allocation": allocations.get(posture_level, allocations[1]),
        "actions": actions,
        "lead_driver": lead_label,
        "lead_value": lead_value,
        "driver_info": driver_info,
        "trend_warning": trend_warning,
        "probability": probability,
        "regime": regime_name,
    }
