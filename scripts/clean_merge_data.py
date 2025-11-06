#!/usr/bin/env python3
"""
Clean & merge your two Google Form response CSVs for a live dashboard.

Inputs (must exist from Step 2.2):
  data/processed/student_form_responses.csv
  data/processed/medical_form_responses.csv

Outputs:
  data/processed/student_clean.csv
  data/processed/medical_clean.csv
  data/processed/combined_responses.csv
  data/processed/meta/student_columns_map.csv
  data/processed/meta/medical_columns_map.csv
  data/processed/meta/pii_columns_removed.csv
"""

from pathlib import Path
import re
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IN_STUD = REPO / "data" / "processed" / "student_form_responses.csv"
IN_MED  = REPO / "data" / "processed" / "medical_form_responses.csv"

OUT_DIR = REPO / "data" / "processed"
META_DIR = OUT_DIR / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)

# Columns that look like PII (case-insensitive, substring match)
PII_PATTERNS = [
    r"email", r"e-mail", r"\bname\b", r"first[-_\s]*name", r"last[-_\s]*name",
    r"contact", r"phone", r"address"
]
PII_REGEXES = [re.compile(pat, re.I) for pat in PII_PATTERNS]

def is_pii(col: str) -> bool:
    col_s = col or ""
    return any(rx.search(col_s) for rx in PII_REGEXES)

def normalize_col(col: str) -> str:
    """Normalize to snake_case: lowercase, trim, collapse spaces, remove non-word chars."""
    if col is None:
        return "unnamed"
    s = col.strip().lower()
    # replace fancy quotes and odd spaces
    s = s.replace("’", "'").replace("‘", "'").replace("“","\"").replace("”","\"")
    s = re.sub(r"\s+", " ", s)          # collapse whitespace
    s = re.sub(r"[^\w\s]", "", s)       # remove punctuation
    s = s.strip().replace(" ", "_")     # snake
    return s or "unnamed"

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1")

def clean_one(df: pd.DataFrame, respondent_type: str):
    """Return: cleaned_df, columns_map_df, removed_pii_list"""
    if df.empty:
        return df, pd.DataFrame(columns=["original","normalized"]), []

    # Save map of original->normalized
    cols_map = []
    for c in df.columns:
        cols_map.append((c, normalize_col(c)))
    cols_map_df = pd.DataFrame(cols_map, columns=["original","normalized"])

    # Drop PII columns
    pii_cols = [c for c in df.columns if is_pii(c)]
    df_nopii = df.drop(columns=pii_cols, errors="ignore")

    # Rename to normalized names
    rename_map = dict(cols_map)
    df_nopii = df_nopii.rename(columns=rename_map)

    # Standard add-ons
    df_nopii["respondent_type"] = respondent_type

    # Normalize empty strings to NaN for safer stats
    df_nopii = df_nopii.replace({"": pd.NA})

    return df_nopii, cols_map_df, pii_cols

def main():
    stud_raw = load_csv(IN_STUD)
    med_raw  = load_csv(IN_MED)

    stud_clean, stud_map, stud_pii = clean_one(stud_raw, "student")
    med_clean,  med_map,  med_pii  = clean_one(med_raw,  "medical")

    # Save per-dataset outputs
    if not stud_clean.empty:
        stud_clean.to_csv(OUT_DIR / "student_clean.csv", index=False)
        stud_map.to_csv(META_DIR / "student_columns_map.csv", index=False)
    if not med_clean.empty:
        med_clean.to_csv(OUT_DIR / "medical_clean.csv", index=False)
        med_map.to_csv(META_DIR / "medical_columns_map.csv", index=False)

    # Save PII removed list (for transparency)
    removed_records = []
    for name, removed in [("student", stud_pii), ("medical", med_pii)]:
        for col in removed:
            removed_records.append({"dataset": name, "column_removed": col})
    pd.DataFrame(removed_records).to_csv(META_DIR / "pii_columns_removed.csv", index=False)

    # Merge with safe union of columns
    combined = pd.concat([stud_clean, med_clean], axis=0, ignore_index=True).fillna(pd.NA)
    combined.to_csv(OUT_DIR / "combined_responses.csv", index=False)

    # Quick health prints
    print(f"[OK] student_clean rows: {0 if stud_clean is None else len(stud_clean)}")
    print(f"[OK] medical_clean rows: {0 if med_clean is None else len(med_clean)}")
    print(f"[OK] combined rows: {len(combined)}")
    print(f"[OK] outputs → {OUT_DIR}")

if __name__ == "__main__":
    main()

