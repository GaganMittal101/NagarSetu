from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUTDIR = ROOT / "data" / "processed"
OUT = OUTDIR / "clean_complaints.csv"

def main():
    files = list(RAW.rglob("*.csv"))
    if not files:
        print("No CSV files found in data/raw/.")
        return
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = df.drop_duplicates()
    for c in ["created_at"]:
        if c in df: df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["latitude","longitude"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"Saved {OUT} | rows={len(df):,}")

if __name__ == "__main__":
    main()
