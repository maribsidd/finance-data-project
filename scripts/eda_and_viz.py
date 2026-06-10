"""
=============================================================
  Real-World Finance Data Project — Stock Market Analysis
  Script 2: eda_and_viz.py
  Purpose : 8 finance-grade charts for market analysis
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import seaborn as sns
import pickle, os

sns.set_theme(style="darkgrid")
plt.rcParams["figure.dpi"] = 130
os.makedirs("outputs", exist_ok=True)

with open("models/processed_df.pkl", "rb") as f:
    df = pickle.load(f)

raw = pd.read_csv("data/stock_data.csv", parse_dates=["Date"])

COLORS = {"AAPL": "#2196F3", "MSFT": "#4CAF50", "AMZN": "#FF9800",
          "JPM": "#9C27B0", "JNJ": "#F44336"}

print("=" * 60)
print("  Generating Finance Visualizations")
print("=" * 60)

# ── CHART 1: Stock Price History (all 5 tickers) ──────────────
print("\n📊 Chart 1: Stock Price History")
fig, ax = plt.subplots(figsize=(14, 6))
for ticker, grp in raw.groupby("Ticker"):
    grp = grp.sort_values("Date")
    ax.plot(grp["Date"], grp["Close"], label=ticker,
            color=COLORS[ticker], lw=2)
ax.set_title("Stock Price History — Jan to Mar 2024", fontsize=15, fontweight="bold")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Closing Price (USD)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=30)
ax.legend(title="Ticker", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/chart1_price_history.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart1_price_history.png")

# ── CHART 2: Daily Returns Distribution ───────────────────────
print("📊 Chart 2: Daily Returns Distribution")
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=False)
fig.suptitle("Daily Returns Distribution by Stock", fontsize=14, fontweight="bold")
for ax, (ticker, grp) in zip(axes, df.groupby("Ticker")):
    returns = grp["DailyReturn"].dropna()
    ax.hist(returns, bins=20, color=COLORS[ticker], alpha=0.8, edgecolor="white")
    ax.axvline(returns.mean(), color="black", lw=2, linestyle="--", label=f"Mean: {returns.mean():.2f}%")
    ax.set_title(ticker, fontsize=12, fontweight="bold", color=COLORS[ticker])
    ax.set_xlabel("Daily Return (%)", fontsize=10)
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("outputs/chart2_returns_distribution.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart2_returns_distribution.png")

# ── CHART 3: Candlestick-style OHLC for AAPL ─────────────────
print("📊 Chart 3: AAPL OHLC Price Chart")
aapl = raw[raw["Ticker"] == "AAPL"].sort_values("Date").tail(30).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(14, 6))
for i, row in aapl.iterrows():
    color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
    ax.plot([i, i], [row["Low"], row["High"]], color=color, lw=1.5)
    ax.bar(i, row["Close"] - row["Open"], bottom=min(row["Open"], row["Close"]),
           color=color, width=0.6, alpha=0.9)
ax.set_xticks(range(0, len(aapl), 5))
ax.set_xticklabels([aapl["Date"].iloc[j].strftime("%b %d")
                    for j in range(0, len(aapl), 5)], rotation=30)
ax.set_title("AAPL — OHLC Candlestick Chart (Last 30 Trading Days)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Price (USD)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.2f}"))
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#26a69a", label="Bullish (Close > Open)"),
                   Patch(color="#ef5350", label="Bearish (Close < Open)")],
          fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig("outputs/chart3_aapl_candlestick.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart3_aapl_candlestick.png")

# ── CHART 4: Moving Averages + RSI (MSFT) ────────────────────
print("📊 Chart 4: MSFT Moving Averages & RSI")
msft = df[df["Ticker"] == "MSFT"].sort_values("Date").copy()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(msft["Date"], msft["Close"], label="Close", color="#4CAF50", lw=2)
ax1.plot(msft["Date"], msft["MA_7"],  label="MA 7",  color="#FF9800", lw=1.5, linestyle="--")
ax1.plot(msft["Date"], msft["MA_20"], label="MA 20", color="#F44336", lw=1.5, linestyle="-.")
ax1.set_title("MSFT — Moving Averages & RSI Indicator", fontsize=14, fontweight="bold")
ax1.set_ylabel("Price (USD)", fontsize=12)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax1.legend(fontsize=11)
ax2.plot(msft["Date"], msft["RSI_14"], color="#9C27B0", lw=2, label="RSI 14")
ax2.axhline(70, color="red",   lw=1.2, linestyle="--", alpha=0.7, label="Overbought (70)")
ax2.axhline(30, color="green", lw=1.2, linestyle="--", alpha=0.7, label="Oversold (30)")
ax2.fill_between(msft["Date"], msft["RSI_14"], 70,
                 where=(msft["RSI_14"] >= 70), alpha=0.2, color="red")
ax2.fill_between(msft["Date"], msft["RSI_14"], 30,
                 where=(msft["RSI_14"] <= 30), alpha=0.2, color="green")
ax2.set_ylabel("RSI", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=10, loc="upper left")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/chart4_msft_ma_rsi.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart4_msft_ma_rsi.png")

# ── CHART 5: Volatility Comparison ────────────────────────────
print("📊 Chart 5: Volatility Comparison")
fig, ax = plt.subplots(figsize=(13, 5))
for ticker, grp in df.groupby("Ticker"):
    grp = grp.sort_values("Date")
    ax.plot(grp["Date"], grp["Volatility_7"],
            label=ticker, color=COLORS[ticker], lw=1.8, alpha=0.85)
ax.set_title("7-Day Rolling Volatility by Stock", fontsize=14, fontweight="bold")
ax.set_ylabel("Volatility (Std of Daily Return %)", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30)
ax.legend(title="Ticker", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/chart5_volatility.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart5_volatility.png")

# ── CHART 6: Correlation Heatmap of Returns ───────────────────
print("📊 Chart 6: Return Correlation Heatmap")
pivot = df.pivot_table(index="Date", columns="Ticker", values="DailyReturn")
corr  = pivot.corr()
fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            ax=ax, mask=mask, annot_kws={"size": 12})
ax.set_title("Daily Return Correlation Between Stocks", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/chart6_return_correlation.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart6_return_correlation.png")

# ── CHART 7: Volume vs Price Movement ────────────────────────
print("📊 Chart 7: Volume vs Price Movement")
aapl_full = df[df["Ticker"] == "AAPL"].sort_values("Date").copy()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(aapl_full["Date"], aapl_full["Close"], color="#2196F3", lw=2, label="Close Price")
ax1.set_title("AAPL — Price vs Volume Analysis", fontsize=14, fontweight="bold")
ax1.set_ylabel("Price (USD)", fontsize=12)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.2f}"))
ax1.legend(fontsize=11)
bar_colors = ["#26a69a" if r >= 0 else "#ef5350"
              for r in aapl_full["DailyReturn"].fillna(0)]
ax2.bar(aapl_full["Date"], aapl_full["Volume"] / 1e6,
        color=bar_colors, alpha=0.8, width=0.8)
ax2.set_ylabel("Volume (M shares)", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/chart7_volume_price.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart7_volume_price.png")

# ── CHART 8: Cumulative Returns ───────────────────────────────
print("📊 Chart 8: Cumulative Returns")
fig, ax = plt.subplots(figsize=(14, 6))
for ticker, grp in df.groupby("Ticker"):
    grp = grp.sort_values("Date").copy()
    grp["CumReturn"] = (1 + grp["DailyReturn"] / 100).cumprod() - 1
    ax.plot(grp["Date"], grp["CumReturn"] * 100,
            label=ticker, color=COLORS[ticker], lw=2.2)
ax.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.6)
ax.fill_between(grp["Date"], 0, 0, alpha=0)
ax.set_title("Cumulative Returns — All Stocks (Jan–Mar 2024)", fontsize=14, fontweight="bold")
ax.set_ylabel("Cumulative Return (%)", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30)
ax.legend(title="Ticker", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/chart8_cumulative_returns.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart8_cumulative_returns.png")

print("\n" + "=" * 60)
print("  ✅ All 8 Charts Generated → outputs/ folder")
print("=" * 60)
