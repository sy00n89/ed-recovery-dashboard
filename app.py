#!/usr/bin/env python3
# Streamlit Live Insight Dashboard for Mirror
# Reads data/processed/combined_responses.csv and provides interactive filters + charts.
# Optional: Click "Sync from Google Sheets" to pull latest data and re-clean.

import os
from pathlib import Path
import subprocess
import pandas as pd
import streamlit as st
import plotly.express as px

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data" / "processed"
COMBINED = DATA_DIR / "combined_responses.csv"
STUDENT = DATA_DIR / "student_clean.csv"
MEDICAL = DATA_DIR / "medical_clean.csv"

PII_KEYWORDS = ("email", "e-mail", "name", "first name", "last name", "contact", "phone", "address")

# Ensure service account exists when running in the cloud (Streamlit Secrets)
try:
    if "service_account_json" in st.secrets:
        cred_dir = REPO / "credentials"
        cred_dir.mkdir(exist_ok=True, parents=True)
        (cred_dir / "service_account.json").write_text(st.secrets["service_account_json"])
except Exception:
    pass


# Simple topic buckets for quick summaries
TOPIC_BUCKETS = {
    "Barriers": [
        "access","wait","cost","insurance","time","stigma","shame","privacy","confidential","schedule","distance"
    ],
    "Helpful supports": [
        "peer","community","group","friend","family","therap","counsel","dietitian","coach","mentor","support"
    ],
    "Desired features": [
        "anonymous","chat","journal","track","goal","reminder","resource","crisis","hotline","moderation",
        "identity","lgbt","bipoc","athlete","meal","plan"
    ],
}

st.set_page_config(page_title="Mirror — Live Insights", layout="wide")
st.title("Mirror — Live Insight Dashboard")

if st.button("🔁 Manual refresh"):
    st.rerun()

# --- Optional HTML auto-refresh every 60 seconds ---
st.markdown(
    "<meta http-equiv='refresh' content='60'>",
    unsafe_allow_html=True
)


# --- Helper: safe read with basic PII guard ---
def load_combined():
    if COMBINED.exists():
        df = pd.read_csv(COMBINED, dtype=str)
    else:
        # fallback: merge student + medical if combined not present
        frames = []
        for p in (STUDENT, MEDICAL):
            if p.exists():
                frames.append(pd.read_csv(p, dtype=str))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if df.empty:
        return df

    # PII guard: drop any suspicious columns by name
    cols_to_drop = [c for c in df.columns if any(k in c.lower() for k in PII_KEYWORDS)]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Try to parse a timestamp-like column automatically
    ts_candidates = [c for c in df.columns if "timestamp" in c.lower()]
    if ts_candidates:
        ts_col = ts_candidates[0]
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        except Exception:
            pass
    return df

# --- Sidebar controls ---
with st.sidebar:
    st.header("Controls")
    st.caption("Use these to filter and explore responses.")

    if st.button("🔄 Sync from Google Sheets now"):
        with st.spinner("Pulling latest Google Sheets data..."):
            # Run the two scripts you and I created
            pull = subprocess.run(["python", str(REPO / "scripts" / "pull_sheets.py")], capture_output=True, text=True)
            clean = subprocess.run(["python", str(REPO / "scripts" / "clean_merge_data.py")], capture_output=True, text=True)
        st.success("Sync complete. Scroll down for details.")
        with st.expander("View sync logs"):
            st.code("=== pull_sheets.py ===\n" + pull.stdout + "\n" + pull.stderr + "\n=== clean_merge_data.py ===\n" + clean.stdout + "\n" + clean.stderr)

    df = load_combined()
    if df.empty:
        st.error("No data found. Run the sync or check data/processed/*.csv")
        st.stop()

    # Respondent type filter (if present)
    type_col = "respondent_type" if "respondent_type" in df.columns else None
    if type_col:
        types = sorted([t for t in df[type_col].dropna().unique()])
        chosen_types = st.multiselect("Respondent type", options=types, default=types)
        if chosen_types:
            df = df[df[type_col].isin(chosen_types)]

    # Time range filter (if a timestamp column exists)
    ts_candidates = [c for c in df.columns if "timestamp" in c.lower()]
    ts_col = ts_candidates[0] if ts_candidates else None
    if ts_col and pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        min_dt = df[ts_col].min()
        max_dt = df[ts_col].max()
        start, end = st.date_input("Date range", value=(min_dt.date(), max_dt.date()))
        mask = (df[ts_col].dt.date >= start) & (df[ts_col].dt.date <= end)
        df = df[mask]

    # Column pickers
    st.markdown("---")
    st.subheader("Pick a question to analyze")
    # Candidate “question” columns: long text or select-multiple often have many unique values
    ignore_cols = {type_col, ts_col}
    candidates = [c for c in df.columns if c not in ignore_cols and df[c].notna().sum() > 0]
    question_col = st.selectbox("Question (column)", options=sorted(candidates))

    st.caption("If this question allows multiple selections, separate values by commas or semicolons for counting.")

# --- Main layout ---
col1, col2 = st.columns([2, 1])

def explode_multi(df: pd.DataFrame, col: str) -> pd.Series:
    """Split multi-select answers on commas/semicolons and count frequencies."""
    ser = df[col].dropna().astype(str)
    if ser.empty:
        return pd.Series(dtype=int)
    # heuristic: split on commas or semicolons
    parts = ser.str.split(r"[;,]", regex=True).explode().str.strip()
    parts = parts[parts != ""]
    return parts.value_counts()

with col1:
    st.subheader("Top Answers")
    counts = explode_multi(df, question_col) if question_col else pd.Series(dtype=int)
    if counts.empty:
        # Fall back to single-value counts
        counts = df[question_col].value_counts(dropna=True)
    topn = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)
    top_counts = counts.head(topn).reset_index()
    top_counts.columns = ["answer", "count"]
    if top_counts.empty:
        st.info("No data to display for this question.")
    else:
        fig = px.bar(top_counts, x="answer", y="count")
        fig.update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_counts)

with col2:
    st.subheader("Response Overview")
    st.metric("Total responses (filtered)", len(df))
    if ts_col and pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        by_day = df.groupby(df[ts_col].dt.date).size().reset_index(name="count")
        fig2 = px.line(by_day, x=ts_col, y="count", title="Submissions Over Time")
        fig2.update_layout(xaxis_title="Date", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

def _textify(series: pd.Series) -> pd.Series:
    if series.empty: return series
    s = series.dropna().astype(str).str.lower()
    # split multi-select cells so “a, b; c” counts separately
    s = s.str.split(r"[;,]", regex=True).explode().str.strip()
    return s[s != ""]

def topic_counts(df: pd.DataFrame, columns: list[str]) -> dict:
    """Count occurrences of keywords in TOPIC_BUCKETS across selected text columns."""
    text = pd.Series(dtype=str)
    for c in columns:
        if c in df.columns:
            text = pd.concat([text, _textify(df[c])], ignore_index=True)
    counts = {k: 0 for k in TOPIC_BUCKETS}
    if text.empty: return counts
    for topic, keys in TOPIC_BUCKETS.items():
        for k in keys:
            counts[topic] += int(text.str.contains(rf"\b{k}", regex=True).sum())
    return counts

def render_insight_cards(df: pd.DataFrame, primary_col: str):
    # Pick a small set of likely rich-text columns to scan
    candidate_cols = [primary_col]
    for hint in ["challenge","help","support","wish","explain","other","feature","tool"]:
        matches = [c for c in df.columns if hint in c.lower()]
        candidate_cols.extend(matches)
    candidate_cols = list(dict.fromkeys(candidate_cols))[:8]  # de-dup & cap

    counts = topic_counts(df, candidate_cols)
    total = max(sum(counts.values()), 1)

    colA, colB, colC = st.columns(3)
    with colA:
        st.caption("Top Theme")
        t = max(counts, key=counts.get)
        st.success(f"**{t}** appears most often in responses.")
    with colB:
        st.caption("Theme Mix")
        mix = ", ".join([f"{k}: {int(100*v/total)}%" for k,v in counts.items()])
        st.info(mix)
    with colC:
        st.caption("Analyst Tip")
        tip = {
            "Barriers": "Prioritize removing access friction (shorter waits, clearer referrals, privacy).",
            "Helpful supports": "Double down on peer groups + therapist touchpoints; show credible resources.",
            "Desired features": "Highlight anonymous chat, identity-based spaces, crisis tools, and gentle tracking."
        }[max(counts, key=counts.get)]
        st.warning(tip)

st.markdown("---")
st.subheader("Insight Cards")
render_insight_cards(df, question_col)
st.markdown("---")
st.subheader("Recent Responses")
preview_cols = [c for c in df.columns if not any(k in c.lower() for k in PII_KEYWORDS)]
st.dataframe(df[preview_cols].tail(25), use_container_width=True)
