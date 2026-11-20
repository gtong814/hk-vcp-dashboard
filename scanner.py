import os
import time
import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf


def get_ggt_tickers():
    """取得最新港股通成分股並轉換為 Yahoo Finance 代碼格式 (如 0700.HK)"""
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
    if curr_close < 1.0:  # 過濾仙股
        return None

    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    c_ma50 = float(ma50.iloc[-1])
    c_ma150 = float(ma150.iloc[-1])
    c_ma200 = float(ma200.iloc[-1])
    ma200_20d_ago = float(ma200.iloc[-21])

    # 1. 基本趨勢過濾 (Stage 2 基礎)
    if not (curr_close > c_ma150 and curr_close > c_ma200 and c_ma200 > ma200_20d_ago):
        return None

    # 52 週高低點與 120 日基底位置
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

    # 2. 波動收縮 (Volatility Contraction)
    range_10d = (float(high.tail(10).max()) - float(low.tail(10).min())) / curr_close
    range_40d = (float(high.tail(40).max()) - float(low.tail(40).min())) / curr_close
    contraction_ratio = range_10d / max(range_40d, 1e-6)

    # 3. 成交量乾涸 (Volume Dry-Up)
    vol_5d = float(vol.tail(5).mean())
    vol_50d = float(vol.tail(50).mean())
    if vol_50d * curr_close < 10_000_000:  # 日均成交額低於 1,000 萬港元略過
        return None
    vol_ratio = vol_5d / max(vol_50d, 1.0)

    # 收縮與量縮條件
    is_tight = (range_10d <= 0.12) and (contraction_ratio <= 0.60)
    is_dry_vol = vol_ratio <= 0.80

    if not (is_tight and is_dry_vol):
        return None

    # 4. 區分 VCP / Cheat / Low Cheat
    pattern = None
    pivot_price = float(high.tail(10).max())

    if dist_from_high <= 0.08 and curr_close >= c_ma50:
        pattern = "VCP"
    elif 0.08 < dist_from_high <= 0.18 and pos_in_base >= 0.55 and curr_close >= c_ma50 * 0.97:
        pattern = "Cheat"
    elif 0.18 < dist_from_high <= 0.35 and 0.30 <= pos_in_base < 0.65:
        # Low Cheat 需確認右側企穩 (近 10 日低點未破近 25 日低點)
        if float(low.tail(10).min()) >= float(low.tail(25).min()) * 0.99:
            pattern = "Low Cheat"

    if pattern is None:
        return None

    # 計算 6 個月動能分數 (用於之後計算 RS 百分位)
    ret_6m = (curr_close / float(close.iloc[-126]) - 1) * 100
    ret_3m = (curr_close / float(close.iloc[-63]) - 1) * 100
    rs_raw = 0.6 * ret_3m + 0.4 * ret_6m

    return {
        "Ticker": ticker,
        "Name": name,
        "Pattern": pattern,
        "Close": round(curr_close, 2),
        "Pivot": round(pivot_price, 2),
        "Dist_Pivot_%": round((pivot_price - curr_close) / curr_close * 100, 2),
        "Dist_52W_High_%": round(dist_from_high * 100, 2),
        "Range_10D_%": round(range_10d * 100, 2),
        "Contraction": round(contraction_ratio, 2),
        "Vol_Ratio_5_50": round(vol_ratio, 2),
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
        res_df = res_df.sort_values(by=["Pattern", "RS_Rank"], ascending=[True, False])
    else:
        res_df = pd.DataFrame(
            columns=[
                "Ticker", "Name", "Pattern", "Close", "Pivot", "Dist_Pivot_%",
                "Dist_52W_High_%", "Range_10D_%", "Contraction", "Vol_Ratio_5_50",
                "RS_Raw", "RS_Rank", "Updated",
            ]
        )

    res_df.to_csv("data/scan_results.csv", index=False, encoding="utf-8-sig")
    print(f"掃描完成，共篩選出 {len(res_df)} 隻符合型態股份。")


if __name__ == "__main__":
    main()
