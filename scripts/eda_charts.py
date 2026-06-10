"""
=============================================================
  Real-World Finance Data Project
  Script 2: eda_charts.py
  Purpose : Generate 8 professional finance visualizations
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

with open("models/enriched_data.pkl", "rb") as f:
    df = pickle.load(f)

COLORS = {
    "AAPL": "#2196F3", "MSFT": "#4CAF50",
    "JPM":  "#FF9800", "JNJ":  "#E91E63", "AMZN": "#9C27B0"
}
TICKERS = list(COLORS.keys())

print("=" * 60)
print("  Generating Finance Visualizations")
print("=" * 60)

# ── CHART 1: Stock Price Trends (All 5 tickers) ───────────────
print("\n📊 Chart 1: Stock Price Trends")
fig, ax = plt.subplots(figsize=(13, 6))
for ticker in TICKERS:
    sub = df[df["Ticker"] == ticker]
    ax.plot(sub["Date"], sub["Close"], label=ticker,
            color=COLORS[ticker], lw=2)
ax.set_title("Stock Closing Price Trends (Jan–Mar 2024)",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Closing Price (USD)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=30)
ax.legend(fontsize=11, loc="upper left")
plt.tight_layout()
plt.savefig("outputs/chart1_price_trends.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart1_price_trends.png")

# ── CHART 2: AAPL Candlestick-style with MA lines ─────────────
print("📊 Chart 2: AAPL Price + Moving Averages")
aapl = df[df["Ticker"] == "AAPL"].copy()
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(aapl["Date"], aapl["Close"], color="#2196F3", lw=2, label="Close Price")
ax.plot(aapl["Date"], aapl["MA7"],   color="#FF9800", lw=1.5, linestyle="--", label="MA7")
ax.plot(aapl["Date"], aapl["MA20"],  color="#E91E63", lw=1.5, linestyle="-.", label="MA20")
ax.fill_between(aapl["Date"], aapl["MA7"], aapl["MA20"],
                where=aapl["MA7"] >= aapl["MA20"],
                alpha=0.15, color="green", label="Bullish Zone")
ax.fill_between(aapl["Date"], aapl["MA7"], aapl["MA20"],
                where=aapl["MA7"] < aapl["MA20"],
                alpha=0.15, color="red", label="Bearish Zone")
ax.set_title("AAPL — Close Price with MA7 & MA20",
             fontsize=15, fontweight="bold")
ax.set_ylabel("Price (USD)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("outputs/chart2_aapl_moving_averages.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart2_aapl_moving_averages.png")

# ── CHART 3: Daily Returns Distribution ───────────────────────
print("📊 Chart 3: Daily Returns Distribution")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
fig.suptitle("Daily Return Distribution by Stock", fontsize=15, fontweight="bold")
for i, ticker in enumerate(TICKERS):
    sub = df[df["Ticker"] == ticker]["DailyReturn"]
    axes[i].hist(sub, bins=25, color=COLORS[ticker], edgecolor="white", alpha=0.85)
    axes[i].axvline(sub.mean(), color="black", lw=2, linestyle="--",
                    label=f"Mean: {sub.mean():.2f}%")
    axes[i].axvline(0, color="red", lw=1, linestyle=":", alpha=0.7)
    axes[i].set_title(ticker, fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Daily Return (%)")
    axes[i].legend(fontsize=9)
axes[5].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/chart3_return_distributions.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart3_return_distributions.png")

# ── CHART 4: Volatility Comparison ────────────────────────────
print("📊 Chart 4: Rolling Volatility Comparison")
fig, ax = plt.subplots(figsize=(13, 5))
for ticker in TICKERS:
    sub = df[df["Ticker"] == ticker]
    ax.plot(sub["Date"], sub["Volatility7"],
            label=ticker, color=COLORS[ticker], lw=1.8)
ax.set_title("7-Day Rolling Volatility (Std of Daily Returns)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Volatility (%)", fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("outputs/chart4_volatility.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart4_volatility.png")

# ── CHART 5: Correlation Heatmap of Close Prices ──────────────
print("📊 Chart 5: Stock Price Correlation Heatmap")
pivot = df.pivot_table(values="Close", index="Date", columns="Ticker")
corr  = pivot.corr()
fig, ax = plt.subplots(figsize=(7, 5))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, linewidths=0.5, ax=ax, mask=mask,
            annot_kws={"size": 12})
ax.set_title("Stock Price Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/chart5_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart5_correlation_heatmap.png")

# ── CHART 6: RSI Chart for AAPL & MSFT ────────────────────────
print("📊 Chart 6: RSI Indicator")
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, ticker in zip(axes, ["AAPL", "MSFT"]):
    sub = df[df["Ticker"] == ticker]
    ax.plot(sub["Date"], sub["RSI14"], color=COLORS[ticker], lw=2, label=f"{ticker} RSI14")
    ax.axhline(70, color="red",   lw=1.5, linestyle="--", alpha=0.7, label="Overbought (70)")
    ax.axhline(30, color="green", lw=1.5, linestyle="--", alpha=0.7, label="Oversold (30)")
    ax.fill_between(sub["Date"], 70, sub["RSI14"],
                    where=sub["RSI14"] >= 70, color="red",   alpha=0.15)
    ax.fill_between(sub["Date"], 30, sub["RSI14"],
                    where=sub["RSI14"] <= 30, color="green", alpha=0.15)
    ax.set_ylabel("RSI", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title(f"{ticker} — RSI (14-day)", fontsize=12, fontweight="bold")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30)
fig.suptitle("Relative Strength Index (RSI) — Momentum Indicator",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/chart6_rsi_indicator.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart6_rsi_indicator.png")

# ── CHART 7: Risk vs Return Scatter ───────────────────────────
print("📊 Chart 7: Risk vs Return")
risk_ret = df.groupby("Ticker")["DailyReturn"].agg(
    Return="mean", Risk="std"
).reset_index()

fig, ax = plt.subplots(figsize=(8, 6))
for _, row in risk_ret.iterrows():
    ax.scatter(row["Risk"], row["Return"],
               color=COLORS[row["Ticker"]], s=200, zorder=5, edgecolors="white", lw=1.5)
    ax.annotate(row["Ticker"],
                xy=(row["Risk"], row["Return"]),
                xytext=(8, 5), textcoords="offset points",
                fontsize=12, fontweight="bold", color=COLORS[row["Ticker"]])
ax.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.5)
ax.set_xlabel("Risk — Std Dev of Daily Returns (%)", fontsize=12)
ax.set_ylabel("Avg Daily Return (%)", fontsize=12)
ax.set_title("Risk vs Return — Portfolio Overview", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/chart7_risk_vs_return.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart7_risk_vs_return.png")

# ── CHART 8: Cumulative Return Since Jan 1 ────────────────────
print("📊 Chart 8: Cumulative Return")
fig, ax = plt.subplots(figsize=(13, 6))
for ticker in TICKERS:
    sub = df[df["Ticker"] == ticker].sort_values("Date")
    cumret = (1 + sub["DailyReturn"] / 100).cumprod() - 1
    ax.plot(sub["Date"], cumret * 100,
            label=ticker, color=COLORS[ticker], lw=2)
ax.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.5)
ax.set_title("Cumulative Return Since Jan 2024 (%)",
             fontsize=15, fontweight="bold")
ax.set_ylabel("Cumulative Return (%)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: f"{v:+.1f}%"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("outputs/chart8_cumulative_return.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart8_cumulative_return.png")

print("\n" + "=" * 60)
print("  ✅ All 8 Charts Generated! → outputs/ folder")
print("=" * 60)
