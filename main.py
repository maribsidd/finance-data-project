"""
=============================================================
  Real-World Finance Data Project — Stock Market Analysis
  main.py — Run the full pipeline in one command
=============================================================
  Usage: python main.py
"""

import sys, os, subprocess

def run(script, label):
    print(f"\n{'='*60}\n  ▶  {label}\n{'='*60}")
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        print(f"❌ Failed: {label}")
        sys.exit(1)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("\n🚀 Starting Finance Data Science Pipeline")
    print("   Stocks: AAPL | MSFT | AMZN | JPM | JNJ\n")
    run("scripts/preprocess.py",      "Step 1 — Data Loading & Feature Engineering")
    run("scripts/eda_and_viz.py",     "Step 2 — EDA & Visualizations (8 Charts)")
    run("scripts/predict.py",         "Step 3 — ML Price Direction Prediction (3 Charts)")
    run("scripts/generate_report.py", "Step 4 — Financial Analysis Report")
    print("\n🎉 PIPELINE COMPLETE!")
    print("📁 Check the 'outputs/' folder:")
    print("   Data   : processed_stock_data.csv, financial_analysis_report.txt")
    print("   Charts : chart1 to chart8 (EDA), chart9-11 (ML Prediction)\n")
