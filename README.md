# 📈 Real-World Finance Data Project — Stock Market Analysis

> **Internship Task 4** | Due: 29 Jun 2026  
> Domain: **Finance** | Skills: OHLCV Analysis · Technical Indicators · ML Prediction · Financial Reporting

---

## 📌 Project Overview

End-to-end data science project on **real-world stock market data** for 5 major US companies across 3 sectors:

| Ticker | Company | Sector |
|--------|---------|--------|
| AAPL | Apple Inc | Technology |
| MSFT | Microsoft Corp | Technology |
| AMZN | Amazon.com | Technology |
| JPM | JPMorgan Chase | Finance |
| JNJ | Johnson & Johnson | Healthcare |

The project covers data ingestion → feature engineering → exploratory analysis → machine learning prediction → structured financial report.

---

## 🗂️ Project Structure

```
finance_project/
│
├── data/
│   └── stock_data.csv              # Raw OHLCV data (5 tickers, ~70 trading days)
│
├── scripts/
│   ├── preprocess.py               # Step 1: Feature engineering (RSI, MA, returns)
│   ├── eda_and_viz.py              # Step 2: 8 financial charts
│   ├── predict.py                  # Step 3: ML direction prediction + 3 charts
│   └── generate_report.py          # Step 4: Full financial analysis report
│
├── models/                         # Auto-generated pickle files
│   ├── processed_df.pkl
│   └── prediction_results.pkl
│
├── outputs/                        # Auto-generated
│   ├── processed_stock_data.csv
│   ├── financial_analysis_report.txt
│   ├── chart1_price_history.png
│   ├── chart2_returns_distribution.png
│   ├── chart3_aapl_candlestick.png
│   ├── chart4_msft_ma_rsi.png
│   ├── chart5_volatility.png
│   ├── chart6_return_correlation.png
│   ├── chart7_volume_price.png
│   ├── chart8_cumulative_returns.png
│   ├── chart9_prediction_cm.png
│   ├── chart10_roc_curves.png
│   └── chart11_feature_importance.png
│
├── main.py                         # ▶ Run everything
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

```bash
git clone https://github.com/YOUR_USERNAME/finance-data-project.git
cd finance-data-project
pip install -r requirements.txt
python main.py
```

---

## 🔧 Feature Engineering

| Feature | Description |
|---------|-------------|
| `DailyReturn` | % change in closing price day-over-day |
| `LogReturn` | Natural log of price ratio (used in quant finance) |
| `MA_7` | 7-day moving average of Close |
| `MA_20` | 20-day moving average of Close |
| `Volatility_7` | 7-day rolling standard deviation of daily returns |
| `DayRange_Pct` | (High - Low) / Close × 100 |
| `Momentum_5` | Close minus Close 5 days ago |
| `RSI_14` | Relative Strength Index (14-day window) |
| `VolumeChange` | % change in trading volume |
| `Target_Up` | 1 if next day Close > today Close, else 0 (ML target) |

---

## 📊 11 Visualizations

| # | Chart | Insight |
|---|-------|---------|
| 1 | Price History | All 5 stocks on one timeline |
| 2 | Returns Distribution | Histogram of daily % returns per stock |
| 3 | AAPL Candlestick | OHLC chart (green=bullish, red=bearish) |
| 4 | MSFT MA + RSI | Moving averages with RSI overbought/oversold zones |
| 5 | Volatility | 7-day rolling volatility comparison |
| 6 | Return Correlation | Heatmap showing which stocks move together |
| 7 | Volume vs Price | Trading volume bars color-coded by return |
| 8 | Cumulative Returns | Who grew most from Jan 2024? |
| 9 | Confusion Matrices | ML prediction accuracy per ticker |
| 10 | ROC Curves | Model discrimination ability (AUC) |
| 11 | Feature Importance | Which indicators matter most for prediction |

---

## 🤖 ML Model

- **Model**: Random Forest Classifier (100 trees)
- **Target**: Will tomorrow's price close higher than today?
- **Split**: Chronological (no data leakage — no random shuffle)
- **Evaluation**: Accuracy, ROC-AUC, Confusion Matrix

---

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `matplotlib` | Charts and plots |
| `seaborn` | Statistical visualization |
| `scikit-learn` | ML model, metrics |

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**. It is not financial advice and should not be used for real trading decisions.

---

## 👤 Author

**[Your Name]** | Learning Intern | June 2026
