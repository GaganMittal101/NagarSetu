from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REQUIRED = ["report_id","created_at","category","description","city","area","latitude","longitude","status"]

def main():
    files = list(RAW.rglob("*.csv"))
    if not files:
        print("No CSV files found in data/raw/. Add a permitted dataset first.")
        return
    for path in files:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        print(f"\n{path}")
        print(f"Rows: {len(df):,}")
        print("Missing required:", missing if missing else "None")
        print(f"Duplicate rows: {df.duplicated().sum()}")

if __name__ == "__main__":
    main()
