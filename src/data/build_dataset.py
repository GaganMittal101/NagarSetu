from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "processed" / "clean_complaints.csv"
OUT = ROOT / "data" / "processed" / "NagarSetu_master.csv"

def main():
    if not IN.exists():
        print("Run clean_complaints.py first.")
        return
    df = pd.read_csv(IN)
    if "created_at" in df:
        dt = pd.to_datetime(df["created_at"], errors="coerce")
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day_of_week"] = dt.dt.dayofweek
        df["hour"] = dt.dt.hour
    if "description" in df:
        df["description_length"] = df["description"].fillna("").astype(str).str.len()
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"Created {OUT} | rows={len(df):,} | cols={len(df.columns)}")

if __name__ == "__main__":
    main()

