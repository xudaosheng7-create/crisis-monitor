"""Action recommendation layer for the crisis monitor."""

POSTURES = {
    1: {"name": "Normal", "color": "#2ECC40",
        "desc": "System stable, maintain strategic allocation"},
    2: {"name": "Cautious", "color": "#FFDC00",
        "desc": "Marginal tightening, review risk exposures"},
    3: {"name": "Defensive", "color": "#FF851B",
        "desc": "Reduce risk: cut equities, shorten duration, raise cash"},
    4: {"name": "Protective", "color": "#FF4136",
        "desc": "Capital preservation: minimal equities, safe havens"},
    5: {"name": "Crisis", "color": "#8B0000",
        "desc": "Liquidity is king: cash + Treasuries, wait for reset"},
}


def get_action_plan(probability, regime_name, liquidity, credit, contagion, trend_delta=0.0):
    """Generate risk posture and action recommendations.

    Returns dict with: posture, posture_level, asset_allocation, actions, lead_driver
    """
    # Determine posture level
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

    # Leading stress driver
    drivers = [
        ("credit", credit, "Credit Tightening"),
        ("liquidity", liquidity, "Liquidity Squeeze"),
        ("contagion", contagion, "Market Contagion"),
    ]
    drivers.sort(key=lambda x: x[1], reverse=True)
    lead_module, lead_value, lead_label = drivers[0]

    # Actions per posture level
    base_actions = {
        1: [
            "Maintain strategic asset allocation",
            "Regular rebalancing, keep target weights",
            "Monitor credit spreads and liquidity indicators",
        ],
        2: [
            "Trim high-beta equity exposure (tech, small-caps)",
            "Shorten fixed income duration",
            "Ensure portfolio liquidity for rapid de-risking",
            "Rotate from high-yield to investment-grade credit",
        ],
        3: [
            "Reduce equities to allocation floor, sell cyclicals first",
            "Credit exposure: AA and above only, exit all high yield",
            "Raise cash to 15-25% of portfolio",
            "Add gold / long-dated Treasuries as hedges",
            "Review derivatives hedge effectiveness",
        ],
        4: [
            "Minimize equity exposure (<= 50% of floor allocation)",
            "Liquidate all high-yield and leveraged loans",
            "Cash + short-term Treasuries >= 30% of portfolio",
            "Add USD cash, gold, long-dated Treasuries",
            "Halt all new risk exposure",
            "Set stop-losses / tail-risk hedges",
        ],
        5: [
            "Capital preservation is the only objective",
            "Equity exposure near zero or fully hedged",
            "Cash + short-term Treasuries >= 50%",
            "Avoid all credit risk",
            "Wait for regime to shift below Stress Build-up before re-entering",
            "Monitor Fed / central bank emergency policy signals",
        ],
    }

    actions = base_actions.get(posture_level, base_actions[1])

    # Driver-specific notes
    driver_actions = {
        "credit": [
            "Leading driver: %s (%.0f/100)" % (lead_label, lead_value),
            "Credit crises develop slowly (3-6 month window), time to de-risk",
            "Watch HY effective yield and bank loan standards closely",
        ],
        "liquidity": [
            "Leading driver: %s (%.0f/100)" % (lead_label, lead_value),
            "Liquidity crises develop fast (days to weeks), react quickly",
            "Ensure ample cash reserves, avoid liquidity mismatches",
        ],
        "contagion": [
            "Leading driver: %s (%.0f/100)" % (lead_label, lead_value),
            "Contagion phase is usually near or in crisis, defense first",
            "Watch VIX > 30 duration and bank stock performance",
        ],
    }

    driver_info = driver_actions.get(lead_module, driver_actions["credit"])

    # Trend warning
    trend_warning = ""
    if trend_delta > 10:
        trend_warning = "WARNING: Risk deteriorating rapidly (monthly +%.0f%%), accelerate adjustments" % trend_delta
    elif trend_delta > 5:
        trend_warning = "Risk rising marginally (monthly +%.0f%%), stay vigilant" % trend_delta
    elif trend_delta < -5:
        trend_warning = "Risk improving (monthly %.0f%%), consider staged re-entry" % trend_delta

    # Asset allocation guide
    allocations = {
        1: {"Equities": "Overweight", "IG Credit": "Normal", "High Yield": "Normal",
            "Cash": "Underweight", "Gold / Bonds": "Underweight"},
        2: {"Equities": "Normal", "IG Credit": "Normal", "High Yield": "Underweight",
            "Cash": "Normal", "Gold / Bonds": "Normal"},
        3: {"Equities": "Underweight", "IG Credit": "Light", "High Yield": "Avoid",
            "Cash": "Overweight", "Gold / Bonds": "Overweight"},
        4: {"Equities": "Minimal", "IG Credit": "Underweight", "High Yield": "Exit",
            "Cash": "Heavy", "Gold / Bonds": "Heavy"},
        5: {"Equities": "Exit", "IG Credit": "Avoid", "High Yield": "Exit",
            "Cash": "Max", "Gold / Bonds": "Heavy"},
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
