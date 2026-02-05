#!/usr/bin/env python3
"""
Pull THREE Google Form response sheets and save each to CSV, tolerating duplicate headers.

Works:
- Locally: uses credentials/service_account.json (if present)
- Streamlit Cloud: uses st.secrets["gcp_service_account"] (no file needed)

Prereqs:
- pip install gspread google-auth pandas streamlit
- Share ALL Google Sheets with your service account email (client_email)
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import gspread

# Streamlit is optional locally; safe import
try:
    import streamlit as st
except Exception:
    st = None

# ====== EDIT THESE: paste each Google Sheet URL and (optionally) tab name ======
SOURCES = [
    {
        "name": "students",
        "url": "https://docs.google.com/spreadsheets/d/1qin5S0V2beHcj3A2oV48nF_TX5pW73_M8IdIqx3HIVY/edit?usp=sharing",
        "worksheet": "Form Responses 1",
        "outfile": "student_form_responses.csv",
    },
    {
        "name": "medical",
        "url": "https://docs.google.com/spreadsheets/d/13lY6kHhiJCJP6CBP2CQbtVQzuffS2mn-vXCXC9CtlYE/edit?usp=sharing",
        "worksheet": "Form Responses 1",
        "outfile": "medical_form_responses.csv",
    },
    {
        "name": "future_users",
        "url": "https://docs.google.com/spreadsheets/d/1jAxJyBnzqIiLpFJbYtDAQ9D4PD6Dfw1xURH3ijucILI/edit?usp=sharing",
        "worksheet": "Form Responses 1",
        "outfile": "future_users_form_responses.csv",
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


def make_gspread_client():
    """
    Create gspread client from:
    1) Streamlit secrets (Cloud): st.secrets["gcp_service_account"]
    2) Local file: credentials/service_account.json
    """
    # Streamlit Cloud path
    if st is not None:
        try:
            # st.secrets behaves like a dict
            if "gcp_service_account" in st.secrets:
                return gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        except Exception as e:
            # if secrets exists but malformed, fall through to file
            print(f"[WARN] st.secrets present but could not be used: {e}")

    # Local dev path
    if CRED_PATH.exists():
        return gspread.service_account(filename=str(CRED_PATH))

    raise FileNotFoundError(
        "No Google credentials found.\n"
        "- On Streamlit Cloud: add Secrets with [gcp_service_account]\n"
        "- Locally: place credentials/service_account.json"
    )


def fetch_ws_as_dataframe(gc, url: str, worksheet: str | None) -> pd.DataFrame:
    """Fetch a worksheet safely, handling duplicate headers."""
    sh = gc.open_by_url(url)
    ws = sh.worksheet(worksheet) if worksheet else sh.sheet1

    values = ws.get_all_values()  # list of lists
    if not values:
        return pd.DataFrame()

    headers = dedupe_headers(values[0])
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)

    # Drop completely empty rows
    df = df.dropna(how="all")
    return df


def main():
    gc = make_gspread_client()

    for src in SOURCES:
        name = src["name"]
        url = src["url"]
        ws_name = src.get("worksheet")
        outfile = OUT_DIR / src["outfile"]

        df = fetch_ws_as_dataframe(gc, url, ws_name)

        df.to_csv(outfile, index=False)
        print(f"[{name}] Saved {len(df):,} rows → {outfile}")
        if not df.empty:
            print(f"[{name}] Columns ({len(df.columns)}): {list(df.columns)[:10]}{' ...' if len(df.columns) > 10 else ''}")


if __name__ == "__main__":
    main()
