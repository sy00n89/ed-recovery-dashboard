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
import re
import numpy as np
from collections import Counter, defaultdict


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
    
    st.markdown("---")
    st.subheader("Column mapping")

    guess = suggest(df)
    col_willing  = st.selectbox("Willingness column",  options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["willing"]) + 1) if guess["willing"] in df.columns else 0)
    col_features = st.selectbox("Desired app features", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["features"]) + 1) if guess["features"] in df.columns else 0)
    col_dx       = st.selectbox("ED diagnosis / type", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["diagnosis"]) + 1) if guess["diagnosis"] in df.columns else 0)

    col_pros_feat = st.selectbox("Professionals' ideal support (features/treatments)", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["pros_feat"]) + 1) if guess["pros_feat"] in df.columns else 0)
    col_pros_reco = st.selectbox("Professionals' recommended tools/resources", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["pros_reco"]) + 1) if guess["pros_reco"] in df.columns else 0)

    col_helpful  = st.selectbox("Individuals: what actually helped", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["helpful"]) + 1) if guess["helpful"] in df.columns else 0)
    col_wish     = st.selectbox("Individuals: what do you wish others had done", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["wish"]) + 1) if guess["wish"] in df.columns else 0)
    col_chal     = st.selectbox("Individuals: challenges", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["challenge"]) + 1) if guess["challenge"] in df.columns else 0)
    col_inef     = st.selectbox("Treatments that didn’t work", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["ineff"]) + 1) if guess["ineff"] in df.columns else 0)

    col_comm     = st.selectbox("Supportive community (yes/no)", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["community"]) + 1) if guess["community"] in df.columns else 0)
    col_age      = st.selectbox("Age", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["age"]) + 1) if guess["age"] in df.columns else 0)
    col_gender   = st.selectbox("Gender", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["gender"]) + 1) if guess["gender"] in df.columns else 0)
    col_stage    = st.selectbox("Recovery stage (optional)", options=["(none)"] + list(df.columns), index=(list(df.columns).index(guess["stage"]) + 1) if guess["stage"] in df.columns else 0)

    # turn "(none)" into None
    to_none = lambda x: None if x == "(none)" else x
    col_willing, col_features, col_dx, col_pros_feat, col_pros_reco, col_helpful, col_wish, col_chal, col_inef, col_comm, col_age, col_gender, col_stage = map(to_none, [col_willing, col_features, col_dx, col_pros_feat, col_pros_reco, col_helpful, col_wish, col_chal, col_inef, col_comm, col_age, col_gender, col_stage])


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

def find_col(df, *tokens, default=None, any_match=False):
    """
    Return the first column whose name contains ALL tokens (default) or ANY tokens (any_match=True).
    Tokens are case-insensitive substrings.
    """
    toks = [t.lower() for t in tokens if t]
    cols = list(df.columns)
    for c in cols:
        lc = c.lower()
        ok = any(any(t in lc for t in toks)) if any_match else all(t in lc for t in toks)
        if ok:
            return c
    return default

def suggest(df):
    s = {}
    s["willing"]  = find_col(df, "anonymous", "use", any_match=True) or find_col(df, "anonymous", "space", any_match=True)
    s["features"] = find_col(df, "include", "app", any_match=True) or find_col(df, "feature", any_match=True)
    s["diagnosis"]= find_col(df, "diagnos", any_match=True) or find_col(df, "type", "disorder", any_match=True)
    s["pros_feat"]= find_col(df, "treatment", "look", any_match=True) or find_col(df, "professional", any_match=True)
    s["pros_reco"]= find_col(df, "profession", "recommend", any_match=True) or find_col(df, "resources", "helpful", any_match=True)
    s["helpful"]  = find_col(df, "support", "helped", any_match=True) or find_col(df, "helpful", any_match=True)
    s["wish"]     = find_col(df, "wish", "help", any_match=True)
    s["challenge"]= find_col(df, "challenge", any_match=True) or find_col(df, "experience", any_match=True)
    s["ineff"]    = find_col(df, "didn", any_match=True) or find_col(df, "ineffective", any_match=True)
    s["community"]= find_col(df, "supportive", "community", any_match=True)
    s["age"]      = find_col(df, "how", "old", any_match=True) or find_col(df, "age", any_match=True)
    s["gender"]   = find_col(df, "gender", any_match=True)
    s["stage"]    = find_col(df, "recovery", "stage", any_match=True)
    return s

def split_multi_series(s: pd.Series) -> pd.Series:
    """Split multi-select text on commas/semicolons and trim."""
    if s is None or s.empty:
        return pd.Series(dtype=str)
    return (
        s.dropna().astype(str)
         .str.split(r"[;,]", regex=True)
         .explode()
         .str.strip()
         .replace("", np.nan)
         .dropna()
    )

def safe_counts(series: pd.Series) -> pd.DataFrame:
    if series is None or series.empty:
        return pd.DataFrame(columns=["value","count"])
    vc = series.value_counts(dropna=True)
    return vc.reset_index().rename(columns={"index":"value", series.name if series.name in vc.index else 0:"count"})

def multi_count(df, col):
    return safe_counts(split_multi_series(df[col])) if col in df.columns else pd.DataFrame(columns=["value","count"])


st.markdown("---")
st.subheader("1) Willingness vs Desired Features (size) by Diagnosis (color)")

try:
    if all(c is not None for c in [col_willing, col_features, col_dx]):
        # explode features row-wise with row_id alignment
        base = df[[col_willing, col_dx]].copy()
        base["row_id"] = base.index

        feats = df[[col_features]].copy()
        feats["feature"] = split_multi_series(df[col_features])
        feats["row_id"] = feats.index
        feats = feats.drop(columns=[col_features])

        tmp = feats.merge(base, on="row_id", how="inner").dropna(subset=["feature", col_willing, col_dx])
        agg = tmp.groupby([col_willing, "feature", col_dx]).size().reset_index(name="count")

        if not agg.empty:
            fig = px.scatter(
                agg, x=col_willing, y="feature", size="count", color=col_dx, size_max=40
            )
            fig.update_layout(xaxis_title="Willing to use anonymous space", yaxis_title="Desired feature")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for this view.")
    else:
        st.info("Pick the willingness, features, and diagnosis columns in the sidebar.")
except Exception as e:
    st.warning(f"Bubble chart skipped: {e}")

#Funnel — Pros’ ideal features → Individuals’ desired inclusions
st.markdown("---")
st.subheader("2) Ideal Support Funnel (Pros → Individuals)")

try:
    col_pros_feat = find_col(df, "professional", "treatment") or find_col(df, "what did your treatment look like")
    col_ind_feat  = find_col(df, "included", "app") or find_col(df, "would you like to see")

    pros = multi_count(df, col_pros_feat)
    inds = multi_count(df, col_ind_feat)

    pros["stage"] = "Professionals' ideal support"
    inds["stage"] = "Individuals' desired inclusions"
    pros.columns = ["label","count","stage"]
    inds.columns = ["label","count","stage"]
    funnel = pd.concat([pros, inds], ignore_index=True)
    if not funnel.empty:
        fig = px.funnel(funnel, x="count", y="stage", color="label", title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for funnel.")
except Exception as e:
    st.warning(f"Funnel skipped: {e}")

#Radar — Pros’ recommended vs Individuals’ found helpful
st.markdown("---")
st.subheader("3) Pros Recommended vs Individuals Found Helpful (Radar)")

try:
    col_pros_reco = find_col(df, "profession", "recommend") or find_col(df, "resources", "helpful")
    col_helpful   = find_col(df, "what kind of support actually helped") or find_col(df, "helpful", "healing")
    pros = multi_count(df, col_pros_reco)
    inds = multi_count(df, col_helpful)
    if not pros.empty and not inds.empty:
        domain = sorted(set(pros["value"]).union(set(inds["value"])))
        P = pd.Series(0, index=domain, dtype=int)
        I = pd.Series(0, index=domain, dtype=int)
        P.loc[pros["value"]] = pros["count"].values
        I.loc[inds["value"]] = inds["count"].values
        radar = pd.DataFrame({"feature": domain, "Professionals": P.values, "Individuals": I.values})
        fig = px.line_polar(radar.melt("feature", var_name="group", value_name="count"),
                            r="count", theta="feature", color="group", line_close=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need both professional recommendations and individual helpful tools.")
except Exception as e:
    st.warning(f"Radar skipped: {e}")

# Word cloud (bar approximation) — “What do you wish people had done?”
st.markdown("---")
st.subheader("4) Wishes for Support — Most Frequent Words (with optional grouping)")

try:
    # 1) Find the "wish" column robustly (token-based, works with snake_case)
    col_wish = find_col(df, "wish", "help", any_match=False) or find_col(df, "wish", any_match=True)

    if not col_wish:
        st.info("Pick a different column name pattern — couldn't find a 'wish' + 'help' question.")
    else:
        # 2) Optional grouping by demographics
        #    Suggest common demographics: respondent_type, identify..., gender, student...
        group_candidates = [c for c in df.columns if any(k in c.lower() for k in [
            "respondent_type", "identify", "student", "neuro", "gender", "major", "role"
        ])]
        group_col = st.selectbox(
            "Group by (optional)",
            options=["(none)"] + group_candidates,
            index=0
        )
        group_col = None if group_col == "(none)" else group_col

        # 3) Tokenize the wish text
        txt = (
            df[col_wish].dropna().astype(str).str.lower()
              .str.replace(r"[^a-z\s]", " ", regex=True)
        )
        # A slightly bigger stopword list
        stop = {
            "the","and","to","of","a","in","for","it","that","on","is","was","be","with","as","by","or","an",
            "i","you","your","my","me","we","they","their","our","from","at","this","those","these","had",
            "have","has","do","did","done","should","could","would","about","more","most","very"
        }

        def count_words(series: pd.Series, topn=25):
            tokens = series.str.split().explode()
            tokens = tokens[~tokens.isin(stop)]
            if tokens.empty:
                return pd.DataFrame(columns=["word","count"])
            out = tokens.value_counts().head(topn).reset_index()
            out.columns = ["word","count"]
            return out

        if not group_col:
            # overall frequencies
            top = count_words(txt, topn=25)
            if top.empty:
                st.info("Not enough text in the selected column.")
            else:
                fig = px.bar(top, x="word", y="count", title=None)
                fig.update_layout(xaxis_title="", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        else:
            # show up to 4 groups as tabs (keeps the UI readable)
            vals = [v for v in df[group_col].dropna().astype(str).unique()]
            vals = vals[:4]  # cap to 4 tabs
            if not vals:
                st.info("No non-empty groups found in the selected column.")
            else:
                tabs = st.tabs([f"{group_col}: {v}" for v in vals])
                for v, tab in zip(vals, tabs):
                    with tab:
                        sub = txt[df[group_col].astype(str) == v]
                        top = count_words(sub, topn=20)
                        if top.empty:
                            st.info("No text for this subgroup.")
                        else:
                            fig = px.bar(top, x="word", y="count")
                            fig.update_layout(xaxis_title="", yaxis_title="Count")
                            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Word summary skipped: {e}")

#Heatmap — Challenges × Ineffective treatments
st.markdown("---")
st.subheader("5) Challenges vs Ineffective Treatments (Heatmap)")

try:
    col_chal = find_col(df, "challenges", "experience")
    col_inef = find_col(df, "didn’t work") or find_col(df, "didnt work") or find_col(df, "ineffective")
    if col_chal and col_inef:
        A = split_multi_series(df[col_chal])
        B = split_multi_series(df[col_inef])
        # Build pair counts by joining on index (approximate co-occurrence by row)
        # Safer: explode both and merge back on original index
        eA = df[[col_chal]].copy()
        eA["a"] = split_multi_series(df[col_chal])
        eB = df[[col_inef]].copy()
        eB["b"] = split_multi_series(df[col_inef])
        joined = eA.dropna(subset=["a"]).join(eB.dropna(subset=["b"]), how="inner", lsuffix="_l", rsuffix="_r")
        if not joined.empty:
            mat = joined.pivot_table(index="a", columns="b", aggfunc="size", fill_value=0)
            fig = px.imshow(mat, aspect="auto", labels=dict(x="Ineffective", y="Challenge", color="Count"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough overlapping rows for heatmap.")
    else:
        st.info("Need both a challenges column and an 'ineffective' treatment column.")
except Exception as e:
    st.warning(f"Heatmap skipped: {e}")

#Stacked bar — Supportive community vs Helpful supports (filterable)
st.markdown("---")
st.subheader("6) Supportive Community vs Helpful Supports (Stacked)")

try:
    col_comm = find_col(df, "supportive community")
    col_help = find_col(df, "what kind of support actually helped")
    col_stage = find_col(df, "recovery stage") or find_col(df, "how old when")  # optional
    d = df.copy()
    if col_stage and col_stage in d.columns:
        stages = ["(All)"] + sorted([x for x in d[col_stage].dropna().unique()])
        pick = st.selectbox("Filter by recovery stage (optional):", stages, index=0)
        if pick != "(All)":
            d = d[d[col_stage] == pick]
    if col_comm and col_help:
        m = split_multi_series(d[col_help])
        # rebuild row-wise to align with community response
        tmp = d[[col_comm]].join(m.rename("help_item"))
        gr = tmp.groupby([col_comm, "help_item"]).size().reset_index(name="count")
        fig = px.bar(gr, x="help_item", y="count", color=col_comm, barmode="stack")
        fig.update_layout(xaxis_title="Helpful supports", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need supportive community yes/no and helpful supports.")
except Exception as e:
    st.warning(f"Stacked bar skipped: {e}")

#Scatter — Helpful tools vs “Wish existed”, axes: age & gender
st.markdown("---")
st.subheader("7) Helpful Tools vs Wish Existed (Scatter by Age/Gender)")

try:
    col_helpful = find_col(df, "helpful", "process") or find_col(df, "support", "helped")
    col_wish_ex = find_col(df, "wish had existed")
    col_age = find_col(df, "how old are you") or find_col(df, "age")
    col_gender = find_col(df, "what gender")
    if all(c in df.columns for c in [col_helpful, col_wish_ex, col_age, col_gender]):
        A = split_multi_series(df[col_helpful]).rename("helpful_item")
        B = df[col_wish_ex].fillna("").astype(str).str.slice(0, 40).rename("wish_excerpt")  # short label
        tmp = df[[col_age, col_gender]].join(A)
        tmp = tmp.join(B)
        # Coerce age to numeric (best-effort)
        tmp["age_num"] = pd.to_numeric(tmp[col_age].str.extract(r"(\d+)")[0], errors="coerce")
        tmp = tmp.dropna(subset=["age_num"])
        fig = px.scatter(tmp, x="age_num", y="helpful_item", color=col_gender,
                         hover_data=["wish_excerpt"], title=None)
        fig.update_layout(xaxis_title="Age", yaxis_title="Helpful resource/tool")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need helpful tools, wish existed, age, and gender columns.")
except Exception as e:
    st.warning(f"Scatter skipped: {e}")

#Horizontal bars — Reasons for delaying help (pros vs individuals)
st.markdown("---")
st.subheader("8) Reasons for Delaying / Avoiding Help (Pros vs Individuals)")

try:
    # Pros (from medical/professional sheet): look for “reasons” / “barriers”
    col_pro_delay = find_col(df, "professional", "reason") or find_col(df, "barrier") or find_col(df, "delay")
    # Individuals: use challenges question
    col_ind_delay = find_col(df, "challenges", "experience") or find_col(df, "avoid", "help")

    pros = multi_count(df, col_pro_delay)
    inds = multi_count(df, col_ind_delay)

    pros["who"] = "Professionals"
    inds["who"] = "Individuals"
    out = pd.concat([pros, inds], ignore_index=True).dropna()
    if not out.empty:
        fig = px.bar(out, x="count", y="value", color="who", orientation="h", barmode="group")
        fig.update_layout(xaxis_title="Count", yaxis_title="Reason / Challenge")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need reasons from professionals and challenges from individuals.")
except Exception as e:
    st.warning(f"Horizontal bars skipped: {e}")


st.markdown("---")
st.subheader("Recent Responses")
preview_cols = [c for c in df.columns if not any(k in c.lower() for k in PII_KEYWORDS)]
st.dataframe(df[preview_cols].tail(25), use_container_width=True)
