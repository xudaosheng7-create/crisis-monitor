"""
🌍 Global Financial Risk Monitor — Crisis Early Warning System v3

Streamlit dashboard with 3 pages:
  1. Overview  — current risk, gauge, radar cards
  2. Trends    — time-series charts of all modules
  3. Alerts    — alert history and trigger rules

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from data.sample_data import build_sample_data, FRED_SERIES
from data.indicators import compute_all_stress, evaluate_alerts
from data.live_fetcher import LiveDataPipeline
from engine.scoring import (
    crisis_probability_series,
    get_risk_level,
    composite_stress,
    detect_regime,
    get_regime_series,
)
from dashboard.components import (
    render_gauge,
    render_probability_trend,
    render_module_trends,
    render_radar_bars,
    render_alert_table,
    get_status_info,
    render_signal_lights,
    render_regime_badge,
    render_regime_timeline,
    MODULE_COLORS,
    MODULE_LABELS,
    COLORS,
)

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="全球金融风险监测 GFRM v3",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── Custom CSS (dark theme) ───────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .stMetric { background: #1e1e30; border-radius: 10px; padding: 12px; }
    .stMetric label { color: #888 !important; font-size: 13px !important; }
    [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.0rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 10px 20px; }
    hr { margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────

def _load_config():
    """Load FRED API key. Checks: Streamlit secrets → settings.yaml → env var."""
    # 1. Streamlit Cloud secrets
    try:
        key = st.secrets.get("FRED_API_KEY") or st.secrets.get("fred_api_key")
        if key:
            return key
    except Exception:
        pass

    # 2. Local settings.yaml
    import yaml
    config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        key = cfg.get("api", {}).get("fred_api_key", "")
        if key:
            return key
    except Exception:
        pass

    # 3. Environment variable
    import os
    return os.environ.get("FRED_API_KEY", "")


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add stress indices, probability, regime to a raw DataFrame."""
    df = compute_all_stress(df, lookback=36)
    df["crisis_probability"] = crisis_probability_series(df)
    df["regime"] = get_regime_series(df)
    return df


@st.cache_data(ttl=600)  # 10 min cache
def _load_data_cached(cache_buster: int = 0) -> tuple:
    """
    Load data: try live FRED first, fall back to sample.

    The cache_buster param is incremented by the refresh button
    to force a fresh fetch.
    """
    api_key = _load_config()

    # Try live FRED
    if api_key:
        pipeline = LiveDataPipeline(fred_api_key=api_key)
        if pipeline.is_available:
            try:
                raw = pipeline.fetch_all(use_cache=True, max_cache_hours=12)
                if raw is not None and not raw.empty and raw.shape[1] >= 8:
                    df = _compute_derived(raw)
                    return df, "📡 实时数据 Live", {"mode": "live", "source": "FRED"}
            except Exception:
                pass

    # Fallback to sample
    raw = build_sample_data()
    df = _compute_derived(raw)
    return df, "📦 演示数据 Demo", {"mode": "sample", "source": "内置模拟数据"}


# ── Initialize session state for cache busting ──
if "cache_buster" not in st.session_state:
    st.session_state.cache_buster = 0

df, data_source_label, data_source_info = _load_data_cached(st.session_state.cache_buster)


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 系统状态")

    # Data source
    if data_source_info.get("mode") == "live":
        st.success("📡 实时数据模式")
        st.caption("数据来源: FRED + Yahoo Finance")
    else:
        st.warning("📦 演示数据模式")
        st.caption("使用内置模拟数据")

    st.caption(f"数据截止: {df.index[-1].strftime('%Y-%m-%d')}")

    # Refresh button
    if st.button("🔄 刷新数据 Refresh", use_container_width=True):
        st.session_state.cache_buster += 1
        st.rerun()

    # Show data freshness
    st.caption(f"缓存剩余: 10 分钟自动刷新")

    st.divider()

    # FRED API status
    api_key = _load_config()
    if api_key:
        st.success("🔑 FRED API: 已配置")
    else:
        st.info("🔑 FRED API: 未配置")
        st.caption(
            "[获取免费 API Key](https://fred.stlouisfed.org/docs/api/api_key.html)"
        )

    st.divider()
    st.caption("v3 + Leading Indicators")
    st.caption("Global Financial Risk Monitor")

# ── Latest values ─────────────────────────────────────────────────────
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

prob = float(latest["crisis_probability"])
l_stress = float(latest["liquidity_stress"])
c_stress = float(latest["credit_stress"])
g_stress = float(latest["contagion_stress"])

prev_prob = float(prev["crisis_probability"])
prev_l = float(prev["liquidity_stress"])
prev_c = float(prev["credit_stress"])
prev_g = float(prev["contagion_stress"])

# Leading indicators
funding_lead = float(latest.get("funding_stress_lead", 50))
credit_lead = float(latest.get("credit_momentum_lead", 50))
fear_lead = float(latest.get("fear_momentum_lead", 50))
lead_signal = float(latest.get("leading_signal", 50))

# Regime
current_regime = detect_regime(l_stress, c_stress, g_stress, lead_signal, prob)
prev_regime = detect_regime(prev_l, prev_c, prev_g,
                            float(prev.get("leading_signal", 50)),
                            float(prev.get("crisis_probability", 50)))
regime_changed = prev_regime["name"] if prev_regime["name"] != current_regime["name"] else None

status = get_status_info(prob)
risk = get_risk_level(prob)

# ── Alerts ────────────────────────────────────────────────────────────
alerts = evaluate_alerts(
    df[["liquidity_stress", "credit_stress", "contagion_stress"]],
    df["crisis_probability"],
)


# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════
col_title, col_status, col_regime = st.columns([2.5, 1, 1])
with col_title:
    st.title("🌍 全球金融风险监测系统")
    source_color = "#2ECC40" if data_source_info.get("mode") == "live" else "#FF851B"
    st.caption(
        f"Global Financial Risk Monitor v3 — Crisis Early Warning  |  "
        f"数据源: :{source_color}[{data_source_label}]"
    )
with col_status:
    st.markdown(f"""
    <div style="
        background: {status['color']}22;
        border: 2px solid {status['color']};
        border-radius: 12px;
        padding: 14px 0;
        text-align: center;
        margin-top: 8px;
    ">
        <div style="font-size:13px;color:#888;">当前风险 · Risk Level</div>
        <div style="font-size:22px;font-weight:700;color:{status['color']};">
            {status['label']}
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_regime:
    st.markdown(render_regime_badge(current_regime, regime_changed), unsafe_allow_html=True)

st.divider()


# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab_overview, tab_trends, tab_alerts = st.tabs([
    "🏠 概览 Overview",
    "📈 趋势 Trends",
    "🚨 预警 Alerts",
])


# ───────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ───────────────────────────────────────────────────────────────────────
with tab_overview:
    # ── Row 1: Gauge + Radar Bars ──
    col_gauge, col_bars = st.columns([1, 2])

    with col_gauge:
        st.subheader("🎯 危机概率 Crisis Probability")
        st.caption("未来 3–9 个月发生系统性危机的概率")
        fig_gauge = render_gauge(prob)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with col_bars:
        st.subheader("📡 三大雷达 Three Radars")

        # 3 metric columns
        m1, m2, m3 = st.columns(3)
        with m1:
            delta_l = l_stress - prev_l
            st.metric(
                "💰 流动性 Liquidity",
                f"{l_stress:.1f}",
                delta=f"{delta_l:+.1f}",
                delta_color="inverse",  # rising stress = bad
            )
        with m2:
            delta_c = c_stress - prev_c
            st.metric(
                "🏦 信用 Credit",
                f"{c_stress:.1f}",
                delta=f"{delta_c:+.1f}",
                delta_color="inverse",
            )
        with m3:
            delta_g = g_stress - prev_g
            st.metric(
                "🌊 传染 Contagion",
                f"{g_stress:.1f}",
                delta=f"{delta_g:+.1f}",
                delta_color="inverse",
            )

        # Horizontal bar chart
        fig_bars = render_radar_bars(l_stress, c_stress, g_stress)
        st.plotly_chart(fig_bars, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Row 2: Leading Signal Lights ──
    st.subheader("🔮 领先信号 Leading Indicators")
    st.caption("前瞻性指标 - 检测压力在「出现」之前的变化趋势。领先主指标 1-6 个月。")
    st.markdown(
        render_signal_lights(funding_lead, credit_lead, fear_lead, lead_signal),
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Row 3: 90-day trend ──
    st.subheader("📉 90天危机概率趋势 90-Day Probability Trend")

    fig_trend = render_probability_trend(df, days=90)
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    # ── Row 3: Data source note ──
    with st.expander("📡 数据源与指标说明 Data Sources & Methodology"):
        st.markdown(f"""
        ### 数据来源 Data Sources
        | 模块 | 指标 | FRED ID | 最新值 | 含义 |
        |------|------|---------|--------|------|
        | 💰 流动性 | 10Y Treasury | DGS10 | {latest["treasury_10y"]:.2f}% | 全球资产定价锚，上升=钱变贵 |
        | | Fed 资产负债表 | WALCL | ${latest["fed_balance"]/1e6:.2f}T | 美联储缩表=从市场抽走流动性 |
        | | SOFR | SOFR | {latest["sofr"]:.2f}% | 银行间隔夜融资成本，跳升=钱荒 |
        | | 美元指数 | DTWEXBGS | {latest["dxy"]:.1f} | 美元走强=全球美元荒，新兴市场承压 |
        | | 3M T-Bill | DTB3 | {latest["repo_proxy"]:.2f}% | 短期融资成本，与SOFR利差扩大=回购市场压力 |
        | 🏦 信用 | HY 有效收益率 | BAMLH0A0HYM2EY | {latest["hy_spread"]:.2f}% | 垃圾债融资成本，飙升=市场开始拒绝风险 |
        | | IG 有效收益率 | BAMLC0A0CMEY | {latest["ig_spread"]:.2f}% | 投资级债成本，上升=信用恶化蔓延至优质企业 |
        | | 银行贷款收紧标准 | DRTSCILM | {latest["loan_standards"]:.1f}% | 银行不愿放贷=信用收缩的领先信号 |
        | | 企业违约率 | DRCCLACBS | {latest["default_rate"]:.2f}% | 实际违约率，滞后但确认信用周期位置 |
        | 🌊 传染 | VIX | VIXCLS | {latest["vix"]:.1f} | 恐慌指数，>25=恐惧，>35=极端恐惧 |
        | | 10Y-3M 利差 | T10Y3M | {latest["leveraged_etf_flow"]:.2f}% | 收益率曲线，倒挂后重新陡峭=危机前兆 |
        | | S&P 500 | SP500 | {latest["bank_index"]:.0f} | 股市财富效应，持续下跌=负反馈螺旋风险 |

        ### 每个指标为什么被选中
        | 指标 | 危机预测逻辑 |
        |------|-------------|
        | 10Y Treasury | 2008前6个月10Y从3.8%飙到4.6%；2023年从3.3%飙到5.0% |
        | Fed Balance Sheet | QT缩表→准备金下降→回购市场压力→2019年9月回购危机重演风险 |
        | SOFR | 2020年3月SOFR从1.55%跳至2.30%→流动性冻结，一周内危机爆发 |
        | DXY | 2008、2020年危机前夕美元指数均急剧上升（美元荒） |
        | HY Yield | 垃圾债收益率飙升领先股市下跌3-6个月 |
        | IG Yield | IG成本上升=信用恶化从垃圾级蔓延到投资级→系统性风险 |
        | Loan Standards | 银行贷款收紧领先经济衰退6-12个月（美联储SLOOS调查） |
        | Default Rate | 滞后确认指标——违约率>3%通常意味着信用周期已进入晚期 |
        | VIX | VIX>30持续超过5天=市场结构损坏，流动性枯竭 |
        | 10Y-3M Spread | 倒挂后再陡峭=最可靠的衰退信号（1960年以来100%命中率） |
        | S&P 500 | 下跌20%以上+波动率上升=财富效应反转→消费收缩→衰退 |

        ### 方法论 Methodology
        - **Z-score 标准化**: 滚动3年窗口，每个指标映射到 [0, 100] 压力分
        - **危机概率**: Sigmoid 函数，P = σ(0.40×L + 0.35×C + 0.25×G − 50)
        - **数据源**: {data_source_label}，每10分钟自动刷新（手动点侧边栏🔄强制刷新）

        ### 风险等级 Risk Levels
        | 概率 | 状态 | 含义 |
        |------|------|------|
        | 0–20% | 🟢 正常 | 系统稳定，维持正常配置 |
        | 20–40% | 🟡 注意 | 开始关注风险资产敞口 |
        | 40–60% | 🟠 脆弱 | 3–9个月系统性风险窗口打开 |
        | 60%+ | 🔴 危险 | 高概率进入危机，建议防御性配置 |
        """)


# ───────────────────────────────────────────────────────────────────────
# TAB 2: TRENDS
# ───────────────────────────────────────────────────────────────────────
with tab_trends:
    st.subheader("📈 三大模块压力趋势 Module Stress Trends")

    # Date range selector
    col_days, _ = st.columns([1, 3])
    with col_days:
        trend_days = st.selectbox(
            "时间范围 Time Range",
            options=[30, 60, 90, 180, 360],
            index=2,
            format_func=lambda d: f"最近 {d} 天 Last {d} days",
        )

    fig_modules = render_module_trends(df, days=trend_days)
    st.plotly_chart(fig_modules, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Raw indicator values ──
    st.subheader("📊 原始指标数据 Raw Indicators")
    st.caption("每个指标经过 Z-score 标准化后映射为 0–100 压力分")

    # Show a subset of raw indicators
    indicator_cols = ["treasury_10y", "sofr", "dxy", "hy_spread", "vix", "bank_index"]
    indicator_labels = {
        "treasury_10y": "10Y Treasury (%)",
        "sofr": "SOFR (%)",
        "dxy": "USD Index (DXY)",
        "hy_spread": "HY Spread (bps)",
        "vix": "VIX",
        "bank_index": "KBW Bank Index",
    }

    show_df = df[indicator_cols].tail(trend_days).copy()
    show_df.index = show_df.index.strftime("%Y-%m-%d")
    show_df.rename(columns=indicator_labels, inplace=True)

    st.dataframe(
        show_df[::-1],  # newest first
        use_container_width=True,
        height=300,
    )

    st.divider()

    # ── Regime Timeline ──
    st.subheader("🔄 市场体制切换 Market Regime Timeline")
    st.caption("每个颜色块代表一个市场体制阶段。体制切换本身就是最强的预警信号。")

    col_regime_chart, col_regime_legend = st.columns([3, 1])
    with col_regime_chart:
        fig_regime = render_regime_timeline(df)
        st.plotly_chart(fig_regime, use_container_width=True, config={"displayModeBar": False})
    with col_regime_legend:
        st.markdown("""
        <div style="font-size:13px;line-height:2;">
        📗 <span style="color:#2ECC40;">扩张</span><br>
        📒 <span style="color:#FFDC00;">后周期</span><br>
        📙 <span style="color:#FF851B;">压力积累</span><br>
        📕 <span style="color:#FF4136;">危机</span>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────
# TAB 3: ALERTS
# ───────────────────────────────────────────────────────────────────────
with tab_alerts:
    st.subheader("🚨 预警信号 Alert Signals")

    # ── Alert rules explanation ──
    col_rules, _ = st.columns([2, 1])
    with col_rules:
        st.markdown("""
        ### 预警触发规则 Alert Trigger Rules

        | 规则 | 条件 | 含义 |
        |------|------|------|
        | ⚠️ **双压力信号** | 信用 ≥ 55 **且** 流动性 ≥ 55 | 信用+流动性同时收紧，历史上所有重大危机的前兆 |
        | 🚨 **极端单模块** | 任一模块 ≥ 75 | 单一市场出现极端压力 |
        | 📊 **概率阈值** | P(危机) ≥ 45% | 综合概率突破临界值 |
        | 📈 **趋势变化** | 任一模块 3 个月变动 ≥ 10 点 | 压力快速累积或释放 |
        """)

    st.divider()

    # ── Current alert status ──
    if alerts:
        alert_count = len(alerts)
        high_count = sum(1 for a in alerts if "HIGH" in a["severity"] or "CRITICAL" in a["severity"])
        st.error(f"🚨 当前活跃预警: {alert_count} 条（其中 {high_count} 条高危）")
    else:
        st.success("✅ 当前无预警信号 — 系统处于正常状态")

    # ── Alert table ──
    if alerts:
        st.markdown(render_alert_table(alerts), unsafe_allow_html=True)
    else:
        st.info("当指标触发预警规则时，信号将出现在此处。")

    st.divider()

    # ── Alert simulation ──
    with st.expander("🔮 查看历史压力场景 Historical Stress Scenarios"):
        st.markdown("""
        ### 模型在历史危机中的表现（模拟）

        | 危机事件 | 提前信号 | 峰值概率 | 检测到? |
        |----------|----------|----------|---------|
        | 2008 全球金融危机 | 6–8 个月 | 92% | ✅ 是 |
        | 2020 COVID-19 | 1–2 个月 | 88% | ✅ 是 |
        | 2023 银行危机 | 2–3 个月 | 65% | ✅ 是 |
        | 当前 (2026-07) | — | {:.1f}% | 监控中 |
        """.format(prob))

    st.divider()

    # ── Alert configuration ──
    st.subheader("⚙️ 预警配置 Alert Configuration")
    st.caption("调整预警阈值（当前为推荐默认值）")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input("双压力-信用阈值", value=55, min_value=30, max_value=80, step=5, disabled=True)
    with col2:
        st.number_input("双压力-流动性阈值", value=55, min_value=30, max_value=80, step=5, disabled=True)
    with col3:
        st.number_input("概率预警阈值 (%)", value=45, min_value=20, max_value=80, step=5, disabled=True)
    st.caption("⚠️ MVP版本阈值不可调整，Phase 2 将支持自定义配置。")


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    f"🌍 Global Financial Risk Monitor v3 | "
    f"数据源: {data_source_label} | "
    f"数据截止: {df.index[-1].strftime('%Y-%m-%d')} | "
    "Powered by FRED · Yahoo Finance · Streamlit · Plotly | "
    "⚠️ 本系统仅供研究参考，不构成投资建议"
)
