import os
import time
import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf


def get_ggt_tickers():
    """取得最新港股通成分股並轉換為 Yahoo Finance 代碼格式"""
    try:
        df = ak.stock_hk_ggt_components_em()
        codes = df["代码"].astype(str).tolist()
        names = df["名称"].astype(str).tolist()
        mapping = {}
        for c, n in zip(codes, names):
            num = int(c)
            yf_code = f"{num:04d}.HK" if num < 10000 else f"{num:05d}.HK"
            mapping[yf_code] = n
        return mapping
    except Exception as e:
        print(f"akshare 抓取失敗，改用備用清單: {e}")
        if os.path.exists("data/ggt_list.csv"):
            backup = pd.read_csv("data/ggt_list.csv")
            return dict(zip(backup["Ticker"], backup["Name"]))
        return {
            "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴-W", "3690.HK": "美團-W",
            "1810.HK": "小米集團-W", "0981.HK": "中芯國際", "1211.HK": "比亞迪股份",
            "9660.HK": "地平線機器人-W", "1686.HK": "新意網集團", "1133.HK": "哈爾濱電氣",
        }


def analyze_stock(df, ticker, name):
    if df is None or len(df) < 210:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    curr_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    if curr_close < 1.0:
        return None

    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    c_ma50 = float(ma50.iloc[-1])
    c_ma150 = float(ma150.iloc[-1])
    c_ma200 = float(ma200.iloc[-1])
    ma200_20d_ago = float(ma200.iloc[-21])

    # 1. Stage 2 基礎趨勢過濾
    if not (curr_close > c_ma150 and curr_close > c_ma200 and c_ma200 > ma200_20d_ago):
        return None

    high_52w = float(high.tail(252).max())
    low_52w = float(low.tail(252).min())
    dist_from_high = (high_52w - curr_close) / high_52w
    above_low_52w = (curr_close - low_52w) / low_52w

    if above_low_52w < 0.25 or dist_from_high > 0.35:
        return None

    high_120 = float(high.tail(120).max())
    low_120 = float(low.tail(120).min())
    base_span = max(high_120 - low_120, 1e-6)
    pos_in_base = (curr_close - low_120) / base_span

    # 2. 用「昨日及之前」計算收縮，確保今日放量突破不會被誤殺
    prev_10_high = float(high.iloc[-11:-1].max())
    prev_10_low = float(low.iloc[-11:-1].min())
    range_10d = (prev_10_high - prev_10_low) / prev_close
    range_40d = (float(high.iloc[-41:-1].max()) - float(low.iloc[-41:-1].min())) / prev_close
    contraction_ratio = range_10d / max(range_40d, 1e-6)

    # 3. 成交量計算
    vol_50d = float(vol.iloc[-51:-1].mean())
    if vol_50d * curr_close < 10_000_000:
        return None

    prev_vol_5d = float(vol.iloc[-6:-1].mean())
    vol_ratio_5_50 = prev_vol_5d / max(vol_50d, 1.0)
    today_vol_ratio = float(vol.iloc[-1]) / max(vol_50d, 1.0)
    day_chg_pct = (curr_close / prev_close - 1) * 100

    is_tight = (range_10d <= 0.12) and (contraction_ratio <= 0.60)
    is_dry_vol = vol_ratio_5_50 <= 0.85

    if not (is_tight and is_dry_vol):
        return None

    # 4. 區分型態 (VCP / Cheat / Low Cheat)
    pattern = None
    pivot_price = prev_10_high  # 以過去 10 日整理平台頂部作為 Pivot

    if dist_from_high <= 0.08 and curr_close >= c_ma50:
        pattern = "VCP"
    elif 0.08 < dist_from_high <= 0.18 and pos_in_base >= 0.55 and curr_close >= c_ma50 * 0.97:
        pattern = "Cheat"
    elif 0.18 < dist_from_high <= 0.35 and 0.30 <= pos_in_base < 0.65:
        if float(low.tail(10).min()) >= float(low.tail(25).min()) * 0.99:
            pattern = "Low Cheat"

    if pattern is None:
        return None

    # 5. 即日收市訊號判定
    dist_pivot_pct = (pivot_price - curr_close) / curr_close * 100
    if curr_close > pivot_price and today_vol_ratio >= 1.3 and day_chg_pct > 0:
        signal = "🔥 即日放量突破"
    elif curr_close > pivot_price and day_chg_pct > 0:
        signal = "⚡ 即日突破(縮量)"
    elif -0.5 <= dist_pivot_pct <= 3.0:
        signal = "👀 逼近突破"
    else:
        signal = "💤 蓄勢收縮"

    ret_6m = (curr_close / float(close.iloc[-126]) - 1) * 100
    ret_3m = (curr_close / float(close.iloc[-63]) - 1) * 100
    rs_raw = 0.6 * ret_3m + 0.4 * ret_6m

    return {
        "Ticker": ticker,
        "Name": name,
        "Signal": signal,
        "Pattern": pattern,
        "Close": round(curr_close, 2),
        "Day_Chg_%": round(day_chg_pct, 2),
        "Pivot": round(pivot_price, 2),
        "Dist_Pivot_%": round(dist_pivot_pct, 2),
        "Today_Vol_Ratio": round(today_vol_ratio, 2),
        "Vol_Ratio_5_50": round(vol_ratio_5_50, 2),
        "Dist_52W_High_%": round(dist_from_high * 100, 2),
        "Range_10D_%": round(range_10d * 100, 2),
        "RS_Raw": round(rs_raw, 2),
        "Updated": pd.Timestamp.now(tz="Asia/Hong_Kong").strftime("%Y-%m-%d %H:%M"),
    }


def main():
    os.makedirs("data", exist_ok=True)
    ticker_map = get_ggt_tickers()
    tickers = list(ticker_map.keys())
    print(f"準備掃描 {len(tickers)} 隻港股通股份...")

    results = []
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            data = yf.download(
                batch,
                period="14mo",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            for t in batch:
                try:
                    df_t = data[t].dropna() if len(batch) > 1 else data.dropna()
                    res = analyze_stock(df_t, t, ticker_map[t])
                    if res:
                        results.append(res)
                except Exception:
                    continue
        except Exception as e:
            print(f"Batch {i} 下載失敗: {e}")
        time.sleep(1)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df["RS_Rank"] = (res_df["RS_Raw"].rank(pct=True) * 99).round(0).astype(int)
        # 依訊號優先順序排序：放量突破最前
        sig_order = {"🔥 即日放量突破": 1, "⚡ 即日突破(縮量)": 2, "👀 逼近突破": 3, "💤 蓄勢收縮": 4}
        res_df["Sig_Order"] = res_df["Signal"].map(sig_order).fillna(9)
        res_df = res_df.sort_values(by=["Sig_Order", "RS_Rank"], ascending=[True, False]).drop(columns=["Sig_Order"])
    else:
        res_df = pd.DataFrame(
            columns=[
                "Ticker", "Name", "Signal", "Pattern", "Close", "Day_Chg_%", "Pivot",
                "Dist_Pivot_%", "Today_Vol_Ratio", "Vol_Ratio_5_50", "Dist_52W_High_%",
                "Range_10D_%", "RS_Raw", "RS_Rank", "Updated",
            ]
        )

    res_df.to_csv("data/scan_results.csv", index=False, encoding="utf-8-sig")
    print(f"掃描完成，共篩選出 {len(res_df)} 隻符合型態股份。")


if __name__ == "__main__":
    main()
