import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="港股通 VCP / Cheat 掃描儀", layout="wide")
st.title("港股通 VCP / Cheat / Low Cheat 型態監控 Dashboard")


@st.cache_data(ttl=1800)
def load_scan_results():
    try:
        return pd.read_csv("data/scan_results.csv")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_price_history(ticker):
    df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA150"] = df["Close"].rolling(150).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    return df.tail(140)


df = load_scan_results()

if df.empty:
    st.warning("尚未有掃描結果，請先在 GitHub Actions 手動觸發一次 Daily Scan。")
    st.stop()

st.caption(f"最後更新時間 (HKT)：{df['Updated'].iloc[0]}")

# 側邊欄過濾器
with st.sidebar:
    st.header("篩選條件")
    patterns = st.multiselect(
        "型態分類", ["VCP", "Cheat", "Low Cheat"], default=["VCP", "Cheat", "Low Cheat"]
    )
    max_range = st.slider("近 10 日最大波幅 (%)", 2.0, 15.0, 10.0, 0.5)
    max_vol_ratio = st.slider("5日/50日量比上限", 0.2, 1.0, 0.75, 0.05)

filtered = df[
    (df["Pattern"].isin(patterns))
    & (df["Range_10D_%"] <= max_range)
    & (df["Vol_Ratio_5_50"] <= max_vol_ratio)
]

# 頂部統計指標
c1, c2, c3, c4 = st.columns(4)
c1.metric("總篩選數量", len(filtered))
c2.metric("VCP", int((filtered["Pattern"] == "VCP").sum()))
c3.metric("Cheat", int((filtered["Pattern"] == "Cheat").sum()))
c4.metric("Low Cheat", int((filtered["Pattern"] == "Low Cheat").sum()))

st.dataframe(
    filtered[
        [
            "Ticker", "Name", "Pattern", "Close", "Pivot",
            "Dist_Pivot_%", "Dist_52W_High_%", "Range_10D_%",
            "Vol_Ratio_5_50", "RS_Rank",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

# 互動 K 線圖
if not filtered.empty:
    st.subheader("個股型態檢視")
    options = [f"{r.Ticker} - {r.Name} ({r.Pattern})" for r in filtered.itertuples()]
    selected = st.selectbox("選擇股份查看日線圖", options)
    sel_ticker = selected.split(" - ")[0]
    sel_row = filtered[filtered["Ticker"] == sel_ticker].iloc[0]

    hist = load_price_history(sel_ticker)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25]
    )

    fig.add_trace(
        go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"], name="K線",
        ),
        row=1, col=1,
    )
    for ma, color in [("MA50", "#1f77b4"), ("MA150", "#ff7f0e"), ("MA200", "#d62728")]:
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist[ma], mode="lines", name=ma, line=dict(width=1.4, color=color)),
            row=1, col=1,
        )

    fig.add_hline(
        y=sel_row["Pivot"], line_dash="dash", line_color="green",
        annotation_text=f"Pivot: {sel_row['Pivot']}", row=1, col=1,
    )

    fig.add_trace(
        go.Bar(x=hist.index, y=hist["Volume"], name="成交量", marker_color="rgba(100,149,237,0.5)"),
        row=2, col=1,
    )
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
