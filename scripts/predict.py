"""
=============================================================
  Real-World Finance Data Project — Stock Market Analysis
  Script 3: predict.py
  Purpose : Train Random Forest to predict next-day direction
=============================================================
"""

import pandas as pd
import numpy as np
import pickle, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

with open("models/processed_df.pkl", "rb") as f:
    df = pickle.load(f)

print("=" * 60)
print("  STEP 3: Stock Direction Prediction (ML)")
print("=" * 60)
print("  Goal: Predict whether next day Close > today Close")

FEATURES = ["DailyReturn", "LogReturn", "MA_7", "MA_20",
            "Volatility_7", "DayRange_Pct", "Momentum_5",
            "RSI_14", "VolumeChange", "Volume"]

results_all = {}

for ticker in ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]:
    sub = df[df["Ticker"] == ticker].sort_values("Date").dropna(subset=FEATURES + ["Target_Up"])
    X   = sub[FEATURES]
    y   = sub["Target_Up"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False  # time-series: no shuffle
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else 0.5
    cm      = confusion_matrix(y_test, y_pred)

    results_all[ticker] = {
        "model": model, "scaler": scaler,
        "acc": acc, "auc": auc, "cm": cm,
        "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
        "features": FEATURES
    }
    print(f"\n  {ticker}: Accuracy={acc*100:.1f}%  AUC={auc:.3f}")

# ── CHART 9: Confusion Matrices ───────────────────────────────
print("\n📊 Chart 9: Confusion Matrices (all tickers)")
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Prediction Confusion Matrices — Next Day Direction",
             fontsize=14, fontweight="bold", y=1.02)
cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds"]
for ax, (ticker, r), cmap in zip(axes, results_all.items(), cmaps):
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(r["cm"], display_labels=["Down", "Up"])
    disp.plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(f"{ticker}\nAcc: {r['acc']*100:.1f}%", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/chart9_prediction_cm.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart9_prediction_cm.png")

# ── CHART 10: ROC Curves ─────────────────────────────────────
print("📊 Chart 10: ROC Curves")
COLORS = {"AAPL":"#2196F3","MSFT":"#4CAF50","AMZN":"#FF9800","JPM":"#9C27B0","JNJ":"#F44336"}
fig, ax = plt.subplots(figsize=(8, 6))
for ticker, r in results_all.items():
    if len(r["y_test"].unique()) > 1:
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_proba"])
        ax.plot(fpr, tpr, lw=2, color=COLORS[ticker],
                label=f"{ticker} (AUC={r['auc']:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1.5,label="Random (AUC=0.500)")
ax.set_title("ROC Curves — Next Day Price Direction", fontsize=14, fontweight="bold")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.legend(fontsize=11, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/chart10_roc_curves.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart10_roc_curves.png")

# ── CHART 11: Feature Importance ─────────────────────────────
print("📊 Chart 11: Feature Importance (AAPL model)")
fi = results_all["AAPL"]["model"].feature_importances_
fi_df = pd.Series(fi, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(fi_df)))
ax.barh(fi_df.index, fi_df.values, color=colors, edgecolor="white")
for i, (v, name) in enumerate(zip(fi_df.values, fi_df.index)):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=10)
ax.set_title("Feature Importance — AAPL Direction Prediction", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/chart11_feature_importance.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved → outputs/chart11_feature_importance.png")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  📊 PREDICTION SUMMARY")
print("=" * 60)
print(f"  {'Ticker':<8} {'Accuracy':>10} {'ROC-AUC':>10}")
print("  " + "─" * 32)
for ticker, r in results_all.items():
    print(f"  {ticker:<8} {r['acc']*100:>9.1f}% {r['auc']:>10.3f}")

with open("models/prediction_results.pkl", "wb") as f:
    pickle.dump(results_all, f)
print("\n💾 Saved models → models/prediction_results.pkl")
print("✅ Prediction Complete!\n")
