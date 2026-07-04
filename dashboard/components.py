"""
Reusable dashboard components for the crisis monitor.

All charts use Plotly for interactivity; stat cards use Streamlit native metrics.
"""

from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Color palette ─────────────────────────────────────────────────────
COLORS = {
    "green":  "#2ECC40",
    "yellow": "#FFDC00",
    "orange": "#FF851B",
    "red":    "#FF4136",
    "blue":   "#0074D9",
    "purple": "#B10DC9",
    "dark":   "#1a1a2e",
    "grey":   "#6c757d",
}

MODULE_COLORS = {
    "liquidity": "#0074D9",   # blue — money/water
    "credit":    "#B10DC9",   # purple — trust/premium
    "contagion": "#FF851B",   # orange — fire/spread
}

MODULE_LABELS = {
    "liquidity": "💰 流动性 Liquidity",
    "credit":    "🏦 信用 Credit",
    "contagion": "🌊 传染 Contagion",
}

RISK_ZONES = [
    (0, 20, COLORS["green"]),
    (20, 40, COLORS["yellow"]),
    (40, 60, COLORS["orange"]),
    (60, 100, COLORS["red"]),
]


# ═══════════════════════════════════════════════════════════════════════
# Gauge — Crisis Probability Donut
# ═══════════════════════════════════════════════════════════════════════

def render_gauge(probability: float) -> go.Figure:
    """
    A donut chart showing crisis probability with a big number in the center.
    """
    prob_pct = probability / 100
    remaining = 1 - prob_pct

    # Determine color
    color = COLORS["green"]
    for lo, hi, c in RISK_ZONES:
        if probability <= hi:
            color = c
            break

    fig = go.Figure()

    # Background ring
    fig.add_trace(go.Pie(
        values=[prob_pct, remaining],
        labels=["", ""],
        hole=0.78,
        marker=dict(colors=[color, "#1a1a30"]),
        textinfo="none",
        hoverinfo="none",
        sort=False,
        showlegend=False,
        rotation=90,
    ))

    # Center number
    fig.add_annotation(
        text=f"<b style='font-size:46px;color:{color}'>{probability:.1f}%</b>",
        x=0.5, y=0.52, showarrow=False,
    )
    # Subtitle
    fig.add_annotation(
        text="<span style='font-size:11px;color:#777;'>P(Crisis 3-9mo)</span>",
        x=0.5, y=0.32, showarrow=False,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Radar Cards — 3 stress modules
# ═══════════════════════════════════════════════════════════════════════

def render_radar_card(
    module: str,
    value: float,
    prev_value: Optional[float] = None,
) -> str:
    """
    Returns HTML for a small radar card showing module stress + delta.

    (Streamlit's st.metric handles this natively — this is a fallback.)
    """
    color = MODULE_COLORS[module]
    label = MODULE_LABELS[module]

    delta = None
    if prev_value is not None:
        delta = round(value - prev_value, 1)

    return f"""
    <div style="
        background: linear-gradient(135deg, #1e1e30 0%, #252540 100%);
        border: 1px solid {color}44;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    ">
        <div style="font-size:13px;color:#888;margin-bottom:6px;">{label}</div>
        <div style="font-size:32px;font-weight:700;color:{color};">{value:.1f}</div>
        <div style="font-size:12px;color:#666;">/ 100</div>
        {f'<div style="font-size:14px;color:{"#FF4136" if delta and delta > 0 else "#2ECC40"};">{"↑" if delta and delta > 0 else "↓"} {abs(delta or 0):.1f}</div>' if delta is not None else ''}
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════
# Trend Chart — Crisis Probability over time
# ═══════════════════════════════════════════════════════════════════════

def render_probability_trend(
    df: pd.DataFrame,
    days: int = 90,
) -> go.Figure:
    """
    Line chart of crisis probability with colored risk zones.
    """
    # Take last N rows
    subset = df.tail(days) if len(df) > days else df

    fig = go.Figure()

    # ── Risk zone bands ──
    zone_colors = [
        (0, 20, "rgba(46,204,64,0.08)"),
        (20, 40, "rgba(255,220,0,0.08)"),
        (40, 60, "rgba(255,133,27,0.08)"),
        (60, 100, "rgba(255,65,54,0.08)"),
    ]
    for lo, hi, color in zone_colors:
        fig.add_hrect(
            y0=lo, y1=hi,
            fillcolor=color,
            layer="below",
            line_width=0,
        )

    # ── Probability line with gradient fill ──
    fig.add_trace(go.Scatter(
        x=subset.index,
        y=subset["crisis_probability"],
        mode="lines",
        name="P(Crisis)",
        line=dict(color=COLORS["red"], width=2.5, shape="spline", smoothing=0.3),
        fill="tozeroy",
        fillcolor="rgba(255,65,54,0.06)",
        hovertemplate="%{x|%Y-%m-%d}<br>Probability: %{y:.1f}%<extra></extra>",
    ))

    # ── Threshold lines ──
    for thresh, label, lcolor in [(20, "Watch", COLORS["yellow"]),
                                    (40, "Fragile", COLORS["orange"]),
                                    (60, "Danger", COLORS["red"])]:
        fig.add_hline(
            y=thresh,
            line_dash="dot",
            line_color=lcolor,
            opacity=0.3, line_width=1,
            annotation_text=f" {label} ",
            annotation_position="right",
            annotation_font=dict(size=10, color=lcolor),
        )

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, color="#444"),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#1a1a30",
                    zeroline=False, color="#444", title=None),
        height=360,
        margin=dict(l=0, r=10, t=5, b=5),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Module Trend Chart — 3 stress indices over time
# ═══════════════════════════════════════════════════════════════════════

def render_module_trends(df: pd.DataFrame, days: int = 90) -> go.Figure:
    """
    Three line charts (one per module) stacked vertically.
    """
    subset = df.tail(days) if len(df) > days else df

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            MODULE_LABELS["liquidity"],
            MODULE_LABELS["credit"],
            MODULE_LABELS["contagion"],
        ),
    )

    modules = [
        ("liquidity_stress", MODULE_COLORS["liquidity"], "💰"),
        ("credit_stress", MODULE_COLORS["credit"], "🏦"),
        ("contagion_stress", MODULE_COLORS["contagion"], "🌊"),
    ]

    fill_colors = {
        "liquidity_stress": "rgba(0,116,217,0.06)",
        "credit_stress": "rgba(177,13,201,0.06)",
        "contagion_stress": "rgba(255,133,27,0.06)",
    }
    for i, (col, color, icon) in enumerate(modules, 1):
        fig.add_trace(
            go.Scatter(
                x=subset.index,
                y=subset[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=2, shape="spline", smoothing=0.3),
                fill="tozeroy",
                fillcolor=fill_colors.get(col, "rgba(128,128,128,0.04)"),
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
            ),
            row=i, col=1,
        )
        fig.add_hline(y=50, line_dash="dot", line_color="#333355",
                       opacity=0.5, row=i, col=1)

    fig.update_xaxes(showgrid=False, color="#444", row=3, col=1)
    fig.update_yaxes(range=[0, 100], showgrid=True, gridcolor="#1a1a30", color="#444")
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=10, t=30, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3-Radar Bar Chart — Current snapshot
# ═══════════════════════════════════════════════════════════════════════

def render_radar_bars(liquidity: float, credit: float, contagion: float) -> go.Figure:
    """
    Horizontal bar chart showing current stress per module.
    """
    modules = ["💰 流动性", "🏦 信用", "🌊 传染"]
    values = [liquidity, credit, contagion]
    colors = [
        MODULE_COLORS["liquidity"],
        MODULE_COLORS["credit"],
        MODULE_COLORS["contagion"],
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=modules,
        x=values,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.05)", width=1),
            cornerradius=4,
        ),
        text=[f" {v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=16, color="white", family="sans-serif"),
        hovertemplate="%{y}: %{x:.1f}/100<extra></extra>",
        width=0.5,
    ))

    # Subtle zone reference lines
    for thresh in [25, 50, 75]:
        fig.add_vline(
            x=thresh, line_dash="solid", line_color="#333355",
            opacity=0.4, line_width=1,
        )

    fig.update_layout(
        xaxis=dict(range=[0, 105], showgrid=False, showticklabels=False, title=None),
        yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=14, color="#aaa")),
        height=160,
        margin=dict(l=0, r=40, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        barmode="group",
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Alert List
# ═══════════════════════════════════════════════════════════════════════

def render_alert_table(alerts: List[Dict]) -> str:
    """
    Returns an HTML table of alerts.
    """
    if not alerts:
        return "<p style='color:#2ECC40;text-align:center;'>✅ 当前无预警信号</p>"

    rows = []
    for a in alerts:
        date_str = a["date"].strftime("%Y-%m-%d") if hasattr(a["date"], "strftime") else str(a["date"])
        rows.append(f"""
        <tr style="border-bottom:1px solid #333;">
            <td style="padding:6px 10px;white-space:nowrap;color:#888;">{date_str}</td>
            <td style="padding:6px 10px;white-space:nowrap;">{a['severity']}</td>
            <td style="padding:6px 10px;">{a['type']}</td>
            <td style="padding:6px 10px;color:#ccc;font-size:13px;">{a['message']}</td>
        </tr>
        """)

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
            <tr style="border-bottom:2px solid #444;">
                <th style="padding:8px 10px;text-align:left;color:#aaa;">日期</th>
                <th style="padding:8px 10px;text-align:left;color:#aaa;">严重性</th>
                <th style="padding:8px 10px;text-align:left;color:#aaa;">类型</th>
                <th style="padding:8px 10px;text-align:left;color:#aaa;">描述</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


# ═══════════════════════════════════════════════════════════════════════
# Status badge
# ═══════════════════════════════════════════════════════════════════════

def get_status_info(probability: float) -> Dict:
    """Get status label and color for a given probability."""
    for lo, hi, color in RISK_ZONES:
        if probability <= hi:
            labels = {
                20: ("🟢 正常 Normal", COLORS["green"]),
                40: ("🟡 注意 Watch", COLORS["yellow"]),
                60: ("🟠 脆弱 Fragile", COLORS["orange"]),
                100: ("🔴 危险 Danger", COLORS["red"]),
            }
            return {"label": labels[hi][0], "color": labels[hi][1]}
    return {"label": "🔴 危险 Danger", "color": COLORS["red"]}


# ═══════════════════════════════════════════════════════════════════════
# Leading Signal Lights — 4 forward-looking indicators
# ═══════════════════════════════════════════════════════════════════════

def _light(value: float, label: str) -> str:
    """Single signal light HTML."""
    if value >= 65:
        color, emoji, status = COLORS["red"], "●", "HIGH"
    elif value >= 45:
        color, emoji, status = COLORS["orange"], "●", "ELEVATED"
    elif value >= 30:
        color, emoji, status = COLORS["yellow"], "●", "WATCH"
    else:
        color, emoji, status = COLORS["green"], "●", "LOW"

    bar_pct = min(value, 100)
    return f"""
    <div style="
        background: linear-gradient(145deg, #12122a 0%, #18183a 100%);
        border: 1px solid #252545; border-radius: 12px;
        padding: 14px 16px; text-align: center;
    ">
        <div style="font-size:10px;color:#777;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">{label}</div>
        <div style="font-size:28px;font-weight:700;color:{color};margin:4px 0;line-height:1;">{value:.0f}<span style="font-size:14px;color:#555;">/100</span></div>
        <div style="
            background:#1a1a30; border-radius:4px; height:3px; margin:8px 0; overflow:hidden;
        "><div style="
            background:{color}; height:100%; width:{bar_pct}%; border-radius:4px; transition:width 0.5s;
        "></div></div>
        <div style="font-size:10px;font-weight:600;color:{color};text-transform:uppercase;letter-spacing:0.05em;">{status}</div>
    </div>
    """


def render_signal_lights(
    funding_lead: float,
    credit_lead: float,
    fear_lead: float,
    leading_signal: float,
) -> str:
    """
    Returns HTML for a row of 4 leading signal lights.
    """
    lights = [
        _light(funding_lead, "💰 融资压力 Lead"),
        _light(credit_lead, "🏦 信用动能 Lead"),
        _light(fear_lead, "😰 恐慌动能 Lead"),
        _light(leading_signal, "📡 综合领先信号"),
    ]
    return f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
        {''.join(lights)}
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════
# Regime Badge — Market phase indicator
# ═══════════════════════════════════════════════════════════════════════

def render_regime_badge(regime: dict, changed_from: str = None) -> str:
    """
    Returns HTML for a prominent market regime badge.

    Parameters
    ----------
    regime : dict with keys name, color, desc
    changed_from : previous regime name if regime just shifted
    """
    shift_note = ""
    if changed_from and changed_from != regime["name"]:
        shift_note = (
            f'<div style="font-size:11px;color:#FF851B;margin-top:4px;">'
            f'⚠️ 体制转换: {changed_from} → {regime["name"]}'
            f'</div>'
        )

    return f"""
    <div style="
        background: {regime['color']}15;
        border: 2px solid {regime['color']};
        border-radius: 12px;
        padding: 14px 0;
        text-align: center;
        margin-top: 8px;
    ">
        <div style="font-size:12px;color:#888;">市场周期 · Market Regime</div>
        <div style="font-size:22px;font-weight:700;color:{regime['color']};margin:4px 0;">
            {regime['name']}
        </div>
        <div style="font-size:11px;color:#aaa;padding:0 10px;line-height:1.4;">
            {regime['desc']}
        </div>
        {shift_note}
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════
# Regime timeline mini-chart
# ═══════════════════════════════════════════════════════════════════════

def render_regime_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar showing regime transitions over time.
    Each regime gets a colored block.
    """
    if "regime" not in df.columns:
        return go.Figure()

    regime_color_map = {
        "📗 扩张 Expansion":        COLORS["green"],
        "📒 后周期 Late Cycle":      COLORS["yellow"],
        "📙 压力积累 Stress Build-up": COLORS["orange"],
        "📕 危机 Crisis":            COLORS["red"],
    }

    # Build contiguous segments
    segments = []
    current_regime = None
    current_start = None

    for date, regime in df["regime"].items():
        if regime != current_regime:
            if current_regime is not None:
                segments.append({
                    "start": current_start,
                    "end": date,
                    "regime": current_regime,
                    "color": regime_color_map.get(current_regime, COLORS["grey"]),
                })
            current_regime = regime
            current_start = date

    # Last segment
    if current_regime is not None:
        segments.append({
            "start": current_start,
            "end": df.index[-1],
            "regime": current_regime,
            "color": regime_color_map.get(current_regime, COLORS["grey"]),
        })

    fig = go.Figure()

    for seg in segments:
        duration = (seg["end"] - seg["start"]).days
        fig.add_trace(go.Bar(
            x=[duration],
            y=[""],
            orientation="h",
            name=seg["regime"],
            marker=dict(color=seg["color"], line=dict(color="rgba(255,255,255,0.05)", width=1)),
            text=seg["regime"].split(" ")[1] if " " in seg["regime"] else seg["regime"],
            textposition="inside",
            textfont=dict(size=11, color="white"),
            insidetextanchor="middle",
            hovertemplate=f"{seg['start'].strftime('%Y-%m')} → {seg['end'].strftime('%Y-%m')}<br>{seg['regime']}<extra></extra>",
            showlegend=False,
            base=0,
            width=[duration],
        ))

    fig.update_layout(
        barmode="stack",
        height=50,
        margin=dict(l=10, r=10, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Action Recommendation Card
# ═══════════════════════════════════════════════════════════════════════

def render_action_card(action_plan: dict) -> str:
    """
    Returns HTML for a prominent action recommendation card.
    Shows: posture, asset allocation table, specific actions, driver info.
    """
    p = action_plan

    # Posture badge
    posture = p["posture"]
    posture_color = posture["color"]

    # Asset allocation rows
    alloc_rows = ""
    for asset, guidance in p["asset_allocation"].items():
        g_color = {
            "Overweight": "#2ECC40", "Normal": "#2ECC40",
            "Light": "#FFDC00", "Underweight": "#FF851B",
            "Minimal": "#FF4136", "Avoid": "#FF4136",
            "Exit": "#8B0000", "Heavy": "#FFDC00", "Max": "#FFDC00",
        }.get(guidance, "#888")
        alloc_rows += f"""
        <tr style="border-bottom:1px solid #333;">
            <td style="padding:5px 12px;color:#ccc;">{asset}</td>
            <td style="padding:5px 12px;color:{g_color};font-weight:600;">{guidance}</td>
        </tr>
        """

    # Actions
    action_items = ""
    for a in p["actions"]:
        action_items += f'<li style="margin:4px 0;color:#ddd;">{a}</li>'

    # Driver info
    driver_items = ""
    for d in p["driver_info"]:
        driver_items += f'<li style="margin:2px 0;color:#aaa;font-size:13px;">{d}</li>'

    trend_html = ""
    if p["trend_warning"]:
        tcolor = "#FF4136" if "deteriorating" in p["trend_warning"].lower() else (
            "#2ECC40" if "improving" in p["trend_warning"].lower() else "#FFDC00")
        trend_html = f"""
        <div style="
            background:{tcolor}15; border-left:3px solid {tcolor};
            padding:8px 12px; margin:10px 0; border-radius:4px;
            font-size:13px; color:{tcolor};
        ">{p["trend_warning"]}</div>
        """

    return f"""
    <div style="
        background: linear-gradient(135deg, #1a1a30 0%, #1e1e35 100%);
        border: 1px solid {posture_color}44;
        border-left: 5px solid {posture_color};
        border-radius: 12px;
        overflow: hidden;
    ">
        <!-- Header -->
        <div style="
            background: {posture_color}15;
            padding: 14px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div>
                <div style="font-size:12px;color:#888;text-transform:uppercase;letter-spacing:1px;">
                    Risk Posture · 风险姿态
                </div>
                <div style="font-size:22px;font-weight:700;color:{posture_color};margin-top:2px;">
                    Level {p["posture_level"]} — {posture["name"]}
                </div>
                <div style="font-size:13px;color:#aaa;margin-top:4px;">
                    {posture["desc"]}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px;color:#888;">P(Crisis 3-9mo)</div>
                <div style="font-size:28px;font-weight:700;color:{posture_color};">{p["probability"]:.1f}%</div>
                <div style="font-size:12px;color:#888;">{p["regime"]}</div>
            </div>
        </div>

        {trend_html}

        <!-- Body: 2 columns -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0;">
            <!-- Left: Actions -->
            <div style="padding:16px 20px;border-right:1px solid #2a2a40;">
                <div style="font-size:14px;font-weight:600;color:#fff;margin-bottom:8px;">
                    Suggested Actions
                </div>
                <ul style="padding-left:18px;margin:0;line-height:1.5;">
                    {action_items}
                </ul>
            </div>
            <!-- Right: Allocation + Driver -->
            <div style="padding:16px 20px;">
                <div style="font-size:14px;font-weight:600;color:#fff;margin-bottom:8px;">
                    Asset Allocation Guide
                </div>
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                    {alloc_rows}
                </table>
                <div style="margin-top:12px;padding-top:10px;border-top:1px solid #2a2a40;">
                    <div style="font-size:12px;color:#888;margin-bottom:4px;">Driver Analysis</div>
                    <ul style="padding-left:14px;margin:0;line-height:1.5;">
                        {driver_items}
                    </ul>
                </div>
            </div>
        </div>
    </div>
    """
