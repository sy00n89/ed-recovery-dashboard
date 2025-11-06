#!/usr/bin/env python3
"""
Pull TWO Google Form response sheets and save each to CSV, tolerating duplicate headers.

Prereqs:
- credentials/service_account.json  (your JSON key)
- Share BOTH Google Sheets with your service account email
- pip install gspread google-auth pandas
"""

from pathlib import Path
import gspread
import pandas as pd
from collections import defaultdict

# ====== EDIT THESE: paste each Google Sheet URL and (optionally) the tab name ======
SOURCES = [
    {
        "name": "students",  # used in output filenames
        "url": "https://docs.google.com/spreadsheets/d/1qin5S0V2beHcj3A2oV48nF_TX5pW73_M8IdIqx3HIVY/edit?usp=sharing",
        "worksheet": "Form Responses 1",  # or None to use first tab
        "outfile": "student_form_responses.csv",
    },
    {
        "name": "medical",   # used in output filenames
        "url": "https://docs.google.com/spreadsheets/d/13lY6kHhiJCJP6CBP2CQbtVQzuffS2mn-vXCXC9CtlYE/edit?usp=sharing",
        "worksheet": "Form Responses 1",  # or None to use first tab
        "outfile": "medical_form_responses.csv",
    },
]
# ================================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
CRED_PATH = REPO_ROOT / "credentials" / "service_account.json"
OUT_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def dedupe_headers(headers: list[str]) -> list[str]:
    """De-duplicate header names by appending ' (2)', ' (3)', ... and strip spaces."""
    counts = defaultdict(int)
    result = []
    for h in headers:
        base = (h or "").strip()
        counts[base] += 1
        if counts[base] == 1:
            result.append(base or "Unnamed")
        else:
            result.append(f"{base or 'Unnamed'} ({counts[base]})")
    return result

def fetch_ws_as_dataframe(gc, url: str, worksheet: str | None) -> pd.DataFrame:
    """Fetch a worksheet safely, handling duplicate headers."""
    sh = gc.open_by_url(url)
    ws = sh.worksheet(worksheet) if worksheet else sh.sheet1

    # Use values to avoid gspread's duplicate-header error with get_all_records
    values = ws.get_all_values()  # list of lists
    if not values:
        return pd.DataFrame()

    headers_raw = values[0]
    headers = dedupe_headers(headers_raw)
    rows = values[1:]

    df = pd.DataFrame(rows, columns=headers)

    # Normalize completely empty rows
    df = df.dropna(how="all")
    return df

def main():
    if not CRED_PATH.exists():
        raise FileNotFoundError(f"Missing credentials: {CRED_PATH}")

    gc = gspread.service_account(filename=str(CRED_PATH))

    for src in SOURCES:
        name = src["name"]
        url = src["url"]
        ws_name = src.get("worksheet")
        outfile = OUT_DIR / src["outfile"]

        df = fetch_ws_as_dataframe(gc, url, ws_name)

        # Save even if empty to confirm wiring
        df.to_csv(outfile, index=False)
        print(f"[{name}] Saved {len(df):,} rows to {outfile}")
        if not df.empty:
            print(f"[{name}] Columns: {list(df.columns)}")
            print(f"[{name}] Nulls: {int(df.replace('', pd.NA).isna().sum().sum())}")

if __name__ == "__main__":
    main()
