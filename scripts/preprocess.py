"""
=============================================================
  Real-World Finance Data Project — Stock Market Analysis
  Script 1: preprocess.py
  Purpose : Load, clean, engineer features, save processed data
=============================================================
"""

import pandas as pd
import numpy as np
import os, pickle

def preprocess():
    print("=" * 60)
    print("  STEP 1: Data Loading & Feature Engineering")
    print("=" * 60)

    df = pd.read_csv("data/stock_data.csv", parse_dates=["Date"])
    print(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Tickers : {df['Ticker'].unique().tolist()}")
    print(f"   Date Range: {df['Date'].min().date()} → {df['Date'].max().date()}")

    print(f"\n🔍 Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicates   : {df.duplicated().sum()}")

    print("\n⚙️  Engineering Features...")
    processed = []
    for ticker, grp in df.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()

        grp["DailyReturn"]       = grp["Close"].pct_change() * 100
        grp["LogReturn"]         = np.log(grp["Close"] / grp["Close"].shift(1))
        grp["MA_7"]              = grp["Close"].rolling(7).mean()
        grp["MA_20"]             = grp["Close"].rolling(20).mean()
        grp["Volatility_7"]      = grp["DailyReturn"].rolling(7).std()
        grp["DayRange"]          = grp["High"] - grp["Low"]
        grp["DayRange_Pct"]      = (grp["DayRange"] / grp["Close"]) * 100
        grp["Momentum_5"]        = grp["Close"] - grp["Close"].shift(5)
        grp["VolumeChange"]      = grp["Volume"].pct_change() * 100

        delta = grp["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        grp["RSI_14"]            = 100 - (100 / (1 + rs))

        grp["Target_NextClose"]  = grp["Close"].shift(-1)
        grp["Target_Up"]         = (grp["Target_NextClose"] > grp["Close"]).astype(int)

        processed.append(grp)

    df_feat = pd.concat(processed).dropna().reset_index(drop=True)
    print(f"✅ Features engineered. Final shape: {df_feat.shape}")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    df_feat.to_csv("outputs/processed_stock_data.csv", index=False)
    with open("models/processed_df.pkl", "wb") as f:
        pickle.dump(df_feat, f)

    print("💾 Saved → outputs/processed_stock_data.csv")
    print("✅ Preprocessing Complete!\n")
    return df_feat

if __name__ == "__main__":
    preprocess()
