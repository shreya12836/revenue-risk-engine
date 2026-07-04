"""Quick check: which sheets does the Online Retail II xlsx contain and
what date range does each cover?"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XL_PATH = ROOT / "data" / "online_retail_II.xlsx"

xl = pd.ExcelFile(XL_PATH)
print("Sheets:", xl.sheet_names)
for s in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=s, usecols=["InvoiceDate"])
    print(f"  {s}: {len(df):,} rows, {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")