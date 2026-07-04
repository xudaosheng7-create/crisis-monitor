# 🌍 全球金融风险监测系统 v3 (MVP)

**Global Financial Risk Monitor — Crisis Early Warning System**

一个基于 Streamlit 的全球金融危机早期预警仪表盘，监控三大核心模块：

| 模块 | 指标数 | 权重 | 含义 |
|------|--------|------|------|
| 💰 流动性 Liquidity | 5 | 40% | 钱够不够？ |
| 🏦 信用 Credit | 4 | 35% | 市场敢不敢借钱？ |
| 🌊 传染 Contagion | 3 | 25% | 问题有没有扩散？ |

## 快速启动

```bash
cd crisis-monitor
pip install -r requirements.txt
streamlit run dashboard/app.py
```

浏览器打开 `http://localhost:8501`

## 项目结构

```
crisis-monitor/
├── config/settings.yaml      # 配置文件（API keys, 阈值, 权重）
├── data/
│   ├── sample_data.py        # MVP内置36个月历史数据
│   ├── indicators.py         # Z-score + 压力指数计算
│   ├── fetcher.py            # FRED/Yahoo API 数据抓取器（Phase 2）
│   └── cache.py              # SQLite 缓存层（Phase 2）
├── engine/
│   ├── scoring.py            # Sigmoid 危机概率模型
│   └── backtest.py           # 历史回测框架
├── dashboard/
│   ├── app.py                # Streamlit 主应用（3页面）
│   └── components.py         # 可复用图表组件
└── requirements.txt
```

## 三页面仪表盘

1. **🏠 概览 Overview** — 危机概率大圆环 + 三大雷达卡片 + 90天趋势
2. **📈 趋势 Trends** — 三大模块压力曲线 + 原始指标数据表
3. **🚨 预警 Alerts** — 预警触发规则 + 活跃信号列表

## 核心算法

```
P(Crisis 3-9 months) = σ(0.40×L + 0.35×C + 0.25×G − 50) × 100
```

- **Z-score 标准化**: 滚动3年窗口 → [0, 100] 压力分
- **Sigmoid 概率**: 将综合压力映射为 0-100% 危机概率

## 风险分层

| 概率 | 状态 | 颜色 |
|------|------|------|
| 0–20% | 🟢 正常 Normal | Green |
| 20–40% | 🟡 注意 Watch | Yellow |
| 40–60% | 🟠 脆弱 Fragile | Orange |
| 60%+ | 🔴 危险 Danger | Red |

## 预警规则

- ⚠️ **双压力信号**: 信用≥55 AND 流动性≥55 → 3-9月风险窗口
- 🚨 **极端单模块**: 任一模块≥75 → 单一市场极端压力
- 📊 **概率阈值**: P(Crisis)≥45% → 综合概率突破

## Phase 2 路线图

- [ ] FRED/Yahoo Finance 实时数据接入
- [ ] Telegram/微信预警机器人
- [ ] FastAPI 后端 API
- [ ] 动态权重 ML 学习 (2000-2025 回测)
- [ ] HMM 市场体制转换检测

## 免责声明

⚠️ 本系统仅供研究参考，不构成投资建议。MVP版本使用模拟历史数据演示模型逻辑。
