#!/usr/bin/env python3
import os
import sys
import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

PII_PATTERNS = [
    re.compile(r"email", re.I),
    re.compile(r"e-mail", re.I),
    re.compile(r"name", re.I),
    re.compile(r"phone", re.I),
    re.compile(r"contact", re.I),
    re.compile(r"address", re.I),
]

def is_pii_column(col: str) -> bool:
    return any(p.search(col) for p in PII_PATTERNS)

def audit_csv(path: Path):
    print("=" * 80)
    print(f"FILE: {path.name}")
    try:
        df = pd.read_csv(path, dtype=str, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding='latin-1', on_bad_lines='skip')
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns):,}")
    print("Columns:")
    for c in df.columns:
        pii_flag = " (PII?)" if is_pii_column(c) else ""
        print(f"  - {c}{pii_flag}")

    # Sample rows (head)
    print("\nHead (first 3 rows):")
    print(df.head(3).to_markdown(index=False))

    # Null/empty summary
    nulls = df.isna().sum().sum()
    empties = (df == "").sum().sum()
    print(f"\nNull values total: {nulls:,} | Empty strings total: {empties:,}")
    print("=" * 80 + "\n")

def main():
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)
    csvs = sorted(p for p in DATA_DIR.glob("*.csv"))
    if not csvs:
        print("No CSV files in data/raw", file=sys.stderr)
        sys.exit(2)
    for p in csvs:
        audit_csv(p)

if __name__ == "__main__":
    main()
