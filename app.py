#!/usr/bin/env python3
# Miirror — Live Insight Dashboard (Tabs + clear insights)
# Reads data/processed/combined_responses.csv (students + medical + future_app_user)
# Optional: Sync from Google Sheets (runs your pull + clean scripts).

from pathlib import Path
import re
import subprocess
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ───────────────────────────── Config ─────────────────────────────
REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data" / "processed"
COMBINED = DATA_DIR / "combined_responses.csv"

PII_FRAGMENTS = ("email", "e-mail", "name", "first name", "last name", "contact", "phone", "address")

SPLIT_PAT = re.compile(r"[;,/|•]\s*")

st.set_page_config(page_title="Miirror — Live Insights", layout="wide")
px.defaults.template = "plotly_white"

st.title("Miirror — Live Insight Dashboard")
st.caption("Executive-friendly insights across Students, Professionals, and Future App Users. Auto-refresh every 60s.")

# Auto-refresh every 60 seconds
st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

# ───────────────────────────── Helpers ─────────────────────────────
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    # Drop any suspicious PII columns just in case (your cleaner already removes)
    drop_cols = [c for c in df.columns if any(k in c.lower() for k in PII_FRAGMENTS)]
    df = df.drop(columns=drop_cols, errors="ignore")
    # Normalize blanks
    df = df.replace({"": pd.NA})
    return df

def find_col(df: pd.DataFrame, *tokens, any_match=False) -> str | None:
    """
    Find a column by substring tokens in the header (case-insensitive).
    - any_match=False: requires ALL tokens
    - any_match=True : requires ANY token
    """
    toks = [t.lower() for t in tokens if t]
    if not toks:
        return None
    for c in df.columns:
        lc = c.lower()
        ok = (any(t in lc for t in toks) if any_match else all(t in lc for t in toks))
        if ok:
            return c
    return None

def split_multi(series: pd.Series) -> pd.Series:
    """Split multi-select answers into individual items."""
    if series is None or series.empty:
        return pd.Series(dtype=str)
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return pd.Series(dtype=str)
    parts = s.str.split(SPLIT_PAT, regex=True).explode().str.strip()
    parts = parts.replace("", pd.NA).dropna()
    return parts

def top_counts(series: pd.Series, top_n=10) -> pd.DataFrame:
    """Top value counts for a series."""
    if series is None or series.empty:
        return pd.DataFrame(columns=["label", "count"])
    vc = series.dropna().astype(str).str.strip()
    vc = vc[vc != ""]
    if vc.empty:
        return pd.DataFrame(columns=["label", "count"])
    out = vc.value_counts().head(top_n).reset_index()
    out.columns = ["label", "count"]
    return out

def top_counts_multi(series: pd.Series, top_n=10) -> pd.DataFrame:
    """Top counts after splitting multi-select answers."""
    return top_counts(split_multi(series), top_n=top_n)

def hbar(
    df: pd.DataFrame,
    title: str,
    label_col: str = "label",
    value_col: str = "count",
    height: int = 380
):
    """
    Safe horizontal bar chart.
    Defaults to label/count so older calls still work:
      hbar(df, "Title", height=420)
    And also supports custom columns:
      hbar(df, "Title", label_col="label", value_col="students_unmet")
    """
    if df is None or df.empty:
        st.info(f"No data available for {title}.")
        return
    if label_col not in df.columns or value_col not in df.columns:
        st.info(f"No data available for {title} (missing '{label_col}' or '{value_col}').")
        return

    # Coerce value column to numeric if needed (sometimes comes in as object)
    plot_df = df.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce").fillna(0)

    fig = px.bar(
        plot_df.sort_values(value_col),
        x=value_col,
        y=label_col,
        orientation="h",
        title=title
    )
    fig.update_layout(
        height=height,
        xaxis_title="Count",
        yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)


def dist_bar_numeric(series: pd.Series, title: str, min_val=1, max_val=10):
    """Distribution bar for 1–10 style numeric answers."""
    if series is None or series.empty:
        st.info("Not enough data to display.")
        return
    nums = pd.to_numeric(series.dropna().astype(str).str.extract(r"(\d+)")[0], errors="coerce").dropna()
    nums = nums[(nums >= min_val) & (nums <= max_val)]
    if nums.empty:
        st.info("Not enough numeric responses to display.")
        return
    dist = nums.value_counts().reindex(range(min_val, max_val + 1), fill_value=0).reset_index()
    dist.columns = ["score", "count"]
    fig = px.bar(dist, x="score", y="count", title=title)
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Average: **{nums.mean():.2f}** · Median: **{nums.median():.0f}** · n={len(nums)}")

def sample_quotes(series: pd.Series, n=3, maxlen=240):
    """Show a few representative answers (non-empty)."""
    if series is None or series.empty:
        return
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return
    for q in s.head(n).tolist():
        q = q if len(q) <= maxlen else q[: maxlen - 1] + "…"
        st.caption(f"• “{q}”")

def keyword_bucket(series: pd.Series, buckets: dict[str, list[str]], top_n=10) -> pd.DataFrame:
    """
    Bucket free text (or multi text) into themes by keyword substring match.
    Unmatched items go into Other (collapsed).
    """
    if series is None or series.empty:
        return pd.DataFrame(columns=["label", "count"])
    s = series.dropna().astype(str).str.lower()
    s = s[s.str.strip() != ""]
    if s.empty:
        return pd.DataFrame(columns=["label", "count"])

    counts = Counter()
    other = 0
    for t in s.tolist():
        matched = False
        for label, keys in buckets.items():
            if any(k in t for k in keys):
                counts[label] += 1
                matched = True
        if not matched:
            other += 1

    if other:
        counts["Other"] += other

    out = pd.DataFrame([{"label": k, "count": v} for k, v in counts.items()])
    out = out.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)
    return out

# ───────────────────────────── Optional Sync ─────────────────────────────
with st.sidebar:
    st.header("Controls")
    st.caption("Dashboard reads the merged CSV. Sync is optional.")
    if st.button("🔄 Sync from Google Sheets now"):
        with st.spinner("Running pull + clean scripts..."):
            pull = subprocess.run(["python3", str(REPO / "scripts" / "pull_sheets.py")],
                                  capture_output=True, text=True)
            clean = subprocess.run(["python3", str(REPO / "scripts" / "clean_merge_data.py")],
                                   capture_output=True, text=True)
        st.success("Sync complete.")
        with st.expander("View sync logs"):
            st.code("=== pull_sheets.py ===\n" + pull.stdout + "\n" + pull.stderr)
            st.code("=== clean_merge_data.py ===\n" + clean.stdout + "\n" + clean.stderr)
        st.rerun()

# ───────────────────────────── Load merged data ─────────────────────────────
df = safe_read_csv(COMBINED)
if df.empty:
    st.error("No combined data found. Run pull_sheets.py + clean_merge_data.py first.")
    st.stop()

# Ensure respondent_type exists
if "respondent_type" not in df.columns:
    st.error("combined_responses.csv is missing 'respondent_type'. Re-run clean_merge_data.py.")
    st.stop()

# Split respondent groups
df_student = df[df["respondent_type"] == "student"].copy()
df_medical = df[df["respondent_type"] == "medical"].copy()
df_future  = df[df["respondent_type"] == "future_app_user"].copy()

# ───────────────────────────── Column detection (normalized headers) ─────────────────────────────
# Students (lived experience)
S_challenge = find_col(df_student, "challenge", any_match=True) or find_col(df_student, "barrier", any_match=True) or find_col(df_student, "struggle", any_match=True)
S_helpful   = find_col(df_student, "help", "helped", any_match=True) or find_col(df_student, "actually_helped", any_match=True)
S_wish      = find_col(df_student, "wish", any_match=True)
S_ineff     = find_col(df_student, "didnt", any_match=True) or find_col(df_student, "ineffective", any_match=True)

# Professionals
P_reco      = find_col(df_medical, "recommend", any_match=True) or find_col(df_medical, "resources", any_match=True) or find_col(df_medical, "tools", any_match=True)
P_approach  = find_col(df_medical, "treatment", any_match=True) or find_col(df_medical, "approach", any_match=True) or find_col(df_medical, "modalit", any_match=True)

# Future App Users (product signals)
F_used_apps        = find_col(df_future, "have_you_ever_used", "app", any_match=True) or find_col(df_future, "used", "app", any_match=True)
F_motivations      = find_col(df_future, "motiv", "download", any_match=True) or find_col(df_future, "motiv", any_match=True)
F_when_open        = find_col(df_future, "most_likely", "open", any_match=True) or find_col(df_future, "open_the_app", any_match=True)
F_engagement       = find_col(df_future, "engaged", any_match=True) or find_col(df_future, "engagement", any_match=True)
F_helpfulness      = find_col(df_future, "helpful", "mental", any_match=True) or find_col(df_future, "helpful", any_match=True)
F_stop_reasons     = find_col(df_future, "stop_using", any_match=True) or find_col(df_future, "causes_you_to_stop", any_match=True)
F_retention_feats  = find_col(df_future, "features_make", "keep_using", any_match=True) or find_col(df_future, "keep_using", any_match=True)
F_feelings         = find_col(df_future, "feel", "using", any_match=True) or find_col(df_future, "words_best_describe", any_match=True)
F_wish_better      = find_col(df_future, "wish", "did", "better", any_match=True) or find_col(df_future, "wish", "apps", any_match=True)
F_ideal_first      = find_col(df_future, "ideal", "add", "first", any_match=True) or find_col(df_future, "design", "ideal", any_match=True)

F_ed_open          = find_col(df_future, "eating_disorders", "openly", any_match=True) or find_col(df_future, "openly_discussed", any_match=True)
F_ed_anyone        = find_col(df_future, "affect_anyone", any_match=True)
F_seek_support     = find_col(df_future, "comfortable", "seeking_support", any_match=True) or find_col(df_future, "feel_most_comfortable", any_match=True)
F_app_role_ed      = find_col(df_future, "meaningful_role", any_match=True) or find_col(df_future, "app_can_play", any_match=True)
F_content_support  = find_col(df_future, "type_of_content", any_match=True) or find_col(df_future, "content", "supportive", any_match=True)

# ───────────────────────────── Tabs ─────────────────────────────
tab_overview, tab_students, tab_pros, tab_future, tab_gaps = st.tabs(
    ["Overview", "Students", "Professionals", "Future App Users", "Gaps & Alignment"]
)

# ───────────────────────────── Overview ─────────────────────────────
with tab_overview:
    st.subheader("At-a-glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total responses", f"{len(df):,}")
    c2.metric("Students", f"{len(df_student):,}")
    c3.metric("Professionals", f"{len(df_medical):,}")
    c4.metric("Future App Users", f"{len(df_future):,}")

    st.markdown("---")
    comp = pd.DataFrame({
        "group": ["Students", "Professionals", "Future App Users"],
        "count": [len(df_student), len(df_medical), len(df_future)]
    })
    fig = px.pie(comp, values="count", names="group", hole=0.45)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("What this dashboard answers")
    st.markdown(
        "- **Students:** what helped, what didn’t, and what felt missing.\n"
        "- **Professionals:** what providers recommend and how they describe “ideal support.”\n"
        "- **Future App Users:** why people try apps, why they churn, and what drives retention.\n"
        "- **Gaps:** where expectations and lived experience don’t line up."
    )

# ───────────────────────────── Students ─────────────────────────────
with tab_students:
    st.subheader("Students — Lived experience signals")
    if df_student.empty:
        st.info("No student responses yet.")
    else:
        colA, colB = st.columns([1, 1])

        with colA:
            st.markdown("#### Top challenges / barriers")
            if S_challenge:
                ch_df = top_counts_multi(df_student[S_challenge], top_n=10)
                hbar(ch_df, "Top reported challenges", label_col="label", value_col="count", height=420)

                top3 = top_counts_multi(df_student[S_challenge], top_n=3)["label"].tolist()
                if top3:
                    st.caption("Most common: **" + ", ".join(top3) + "**")
            else:
                st.info("Could not detect the students' challenge/barrier column.")

        with colB:
            st.markdown("#### What actually helped")
            if S_helpful:
                help_df = top_counts_multi(df_student[S_helpful], top_n=10)
                hbar(help_df, "Most helpful supports", label_col="label", value_col="count", height=420)

                top3 = top_counts_multi(df_student[S_helpful], top_n=3)["label"].tolist()
                if top3:
                    st.caption("Most helpful: **" + ", ".join(top3) + "**")
            else:
                st.info("Could not detect the students' helpful-support column.")

        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Unmet needs (what they wish people did)")
            if S_wish:
                # Bucket with simple, readable themes (you can refine over time)
                WISH_BUCKETS = {
                    "Listen / validate": ["listen", "validate", "understand", "judg", "dismiss", "believe"],
                    "Practical meal support": ["meal", "eat", "food", "cook", "grocery", "snack", "restaurant"],
                    "Peer/community support": ["peer", "community", "group", "mentor", "friend", "talk", "chat"],
                    "Professional help access": ["therap", "counsel", "dietitian", "treatment", "appointment", "wait"],
                    "Identity-specific support": ["lgbt", "bipoc", "male", "men", "athlete", "culture"],
                }
                buckets = keyword_bucket(df_student[S_wish], WISH_BUCKETS, top_n=10)
                hbar(buckets, "Wishes grouped into themes", label_col="label", value_col="count", height=380)

                st.markdown("**Example quotes**")
                sample_quotes(df_student[S_wish], n=3)
            else:
                st.info("Could not detect the students' wish/unmet-needs column.")

        with c2:
            st.markdown("#### What didn’t work (if captured)")
            if S_ineff and S_ineff in df_student.columns:
                ineff_df = top_counts_multi(df_student[S_ineff], top_n=10)
                hbar(ineff_df, "Most-cited ineffective supports", label_col="label", value_col="count", height=380)

            else:
                st.caption("No 'ineffective treatment' column detected (this is okay).")

# ───────────────────────────── Professionals ─────────────────────────────
with tab_pros:
    st.subheader("Professionals — Provider view")
    if df_medical.empty:
        st.info("No professional responses yet.")
    else:
        colA, colB = st.columns([1, 1])

        with colA:
            st.markdown("#### What professionals recommend most")
            if P_reco:
                hbar(top_counts_multi(df_medical[P_reco], top_n=10), "Top recommended tools/resources", height=420)
                top3 = top_counts_multi(df_medical[P_reco], top_n=3)["label"].tolist()
                if top3:
                    st.caption("Most recommended: **" + ", ".join(top3) + "**")
            else:
                st.info("Could not detect professionals' recommendations/resources column.")

        with colB:
            st.markdown("#### What “ideal support/treatment” looks like")
            if P_approach:
                hbar(top_counts_multi(df_medical[P_approach], top_n=10), "Most common approaches/modalities", height=420)
                top3 = top_counts_multi(df_medical[P_approach], top_n=3)["label"].tolist()
                if top3:
                    st.caption("Common approaches: **" + ", ".join(top3) + "**")
            else:
                st.info("Could not detect professionals' treatment approach column.")

# ───────────────────────────── Future App Users ─────────────────────────────
with tab_future:
    st.subheader("Future App Users — Product adoption & retention signals")
    if df_future.empty:
        st.info("No future app user responses yet.")
    else:
        # KPI cards
        k1, k2, k3, k4 = st.columns(4)

        # % used apps
        used_pct = None
        if F_used_apps and F_used_apps in df_future.columns:
            v = df_future[F_used_apps].dropna().astype(str).str.strip().str.lower()
            if not v.empty:
                used_pct = (v.str.startswith("y")).mean() * 100

        k1.metric("% have used a mental-health app", f"{used_pct:.0f}%" if used_pct is not None else "—")

        # engagement/helpfulness averages
        if F_engagement:
            nums = pd.to_numeric(df_future[F_engagement].dropna().astype(str).str.extract(r"(\d+)")[0], errors="coerce").dropna()
            k2.metric("Avg engagement (1–10)", f"{nums.mean():.2f}" if not nums.empty else "—")
        else:
            k2.metric("Avg engagement (1–10)", "—")

        if F_helpfulness:
            nums = pd.to_numeric(df_future[F_helpfulness].dropna().astype(str).str.extract(r"(\d+)")[0], errors="coerce").dropna()
            k3.metric("Avg helpfulness (1–10)", f"{nums.mean():.2f}" if not nums.empty else "—")
        else:
            k3.metric("Avg helpfulness (1–10)", "—")

        k4.metric("Responses", f"{len(df_future):,}")

        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Engagement distribution")
            if F_engagement:
                dist_bar_numeric(df_future[F_engagement], "Engagement score distribution (1–10)")
            else:
                st.info("Engagement column not detected.")

        with c2:
            st.markdown("#### Helpfulness distribution")
            if F_helpfulness:
                dist_bar_numeric(df_future[F_helpfulness], "Helpfulness score distribution (1–10)")
            else:
                st.info("Helpfulness column not detected.")

        st.markdown("---")
        c3, c4 = st.columns([1, 1])
        with c3:
            st.markdown("#### Why people try apps (motivation)")
            if F_motivations:
                hbar(top_counts_multi(df_future[F_motivations], top_n=10), "Top motivations to try apps", height=420)
            else:
                st.info("Motivation column not detected.")

        with c4:
            st.markdown("#### Why people stop using apps (churn drivers)")
            if F_stop_reasons:
                hbar(top_counts_multi(df_future[F_stop_reasons], top_n=10), "Top reasons for churn", height=420)
            else:
                st.info("Stop-using column not detected.")

        st.markdown("---")
        c5, c6 = st.columns([1, 1])
        with c5:
            st.markdown("#### Features that keep users coming back")
            if F_retention_feats:
                hbar(top_counts_multi(df_future[F_retention_feats], top_n=10), "Retention-driving features", height=420)
            else:
                st.info("Retention-features column not detected.")

        with c6:
            st.markdown("#### Emotional experience of current apps")
            if F_feelings:
                # Simple sentiment-ish grouping (no ML, just readable)
                POS = ["calm", "supported", "hope", "relieved", "motivated", "connected", "safe", "understood"]
                NEG = ["overwhelm", "guilt", "ashamed", "anxious", "stressed", "disconnected", "indifferent", "judged", "worse"]

                items = split_multi(df_future[F_feelings]).astype(str).str.lower()
                if not items.empty:
                    def group_emotion(x):
                        if any(p in x for p in POS): return "Positive"
                        if any(n in x for n in NEG): return "Negative"
                        return "Neutral/Other"
                    grp = items.map(group_emotion).value_counts().reset_index()
                    grp.columns = ["group", "count"]
                    fig = px.bar(grp, x="group", y="count", title="Emotional tone (grouped)")
                    fig.update_layout(height=320)
                    st.plotly_chart(fig, use_container_width=True)

                    # also show top exact feelings (readable list)
                    exact = items.value_counts().head(10).reset_index()
                    exact.columns = ["label", "count"]
                    hbar(exact, "Most common feelings (exact words)", height=380)
                else:
                    st.info("Not enough emotion words to summarize.")
            else:
                st.info("Feelings column not detected.")

        st.markdown("---")
        st.markdown("#### Open-ended product ideas (sample)")
        c7, c8 = st.columns([1, 1])
        with c7:
            st.markdown("**What they wish apps did better**")
            if F_wish_better:
                sample_quotes(df_future[F_wish_better], n=4)
            else:
                st.caption("Open-text ‘wish apps did better’ column not detected.")
        with c8:
            st.markdown("**First feature they’d add to an ideal app**")
            if F_ideal_first:
                sample_quotes(df_future[F_ideal_first], n=4)
            else:
                st.caption("Open-text ‘ideal app first feature’ column not detected.")

        st.markdown("---")
        st.markdown("#### ED-specific readiness (bridge to Miirror)")
        c9, c10 = st.columns([1, 1])
        with c9:
            if F_seek_support:
                hbar(top_counts_multi(df_future[F_seek_support], top_n=10), "Where users feel comfortable seeking support", height=420)
            else:
                st.caption("Support-seeking column not detected.")
        with c10:
            if F_app_role_ed:
                # This sometimes is 1–10 in your raw form; show distribution
                dist_bar_numeric(df_future[F_app_role_ed], "Belief: apps can support ED recovery (1–10)")
            else:
                st.caption("Belief-in-app-role column not detected.")
            if F_content_support:
                hbar(top_counts_multi(df_future[F_content_support], top_n=8), "Content that feels supportive (top)", height=380)

# ───────────────────────────── Gaps & Alignment ─────────────────────────────
with tab_gaps:
    st.subheader("Gaps & Alignment — What differs across groups")

    if df_student.empty or df_medical.empty:
        st.info("Need both student and professional responses to compute gaps.")
    else:
        st.markdown("### 1) Professionals recommend vs Students say helped")
        if S_helpful and P_reco:
            stud = top_counts_multi(df_student[S_helpful], top_n=12).rename(columns={"count": "students_helped"})
            pro  = top_counts_multi(df_medical[P_reco], top_n=12).rename(columns={"count": "pros_recommend"})
            merged = pd.merge(stud, pro, on="label", how="outer").fillna(0)
            merged["gap_students_minus_pros"] = merged["students_helped"] - merged["pros_recommend"]
            merged = merged.sort_values("students_helped", ascending=False).head(15)

            view = merged.melt(id_vars=["label"], value_vars=["students_helped", "pros_recommend"],
                               var_name="group", value_name="count")
            fig = px.bar(view, x="label", y="count", color="group", barmode="group",
                         title="Alignment check (top items only)")
            fig.update_layout(height=420)
            fig.update_xaxes(tickangle=25)
            st.plotly_chart(fig, use_container_width=True)

            # A clean takeaway
            biggest = merged.sort_values("gap_students_minus_pros", ascending=False).head(1)
            if not biggest.empty:
                lbl = biggest.iloc[0]["label"]
                gap = int(biggest.iloc[0]["gap_students_minus_pros"])
                if gap > 0:
                    st.caption(f"Biggest mismatch: students mention **{lbl}** more than professionals by ~{gap} mentions.")
        else:
            st.info("Missing either students-helped or professionals-recommendation column detection.")

    st.markdown("---")
    st.markdown("### 2) Future users want vs Students need (feature direction)")
    if not df_future.empty and S_wish and F_retention_feats:
        # Students unmet needs (themes)
        WISH_BUCKETS = {
            "Listen / validate": ["listen", "validate", "understand", "judg", "dismiss", "believe"],
            "Practical meal support": ["meal", "eat", "food", "cook", "grocery", "snack", "restaurant"],
            "Peer/community support": ["peer", "community", "group", "mentor", "friend", "talk", "chat"],
            "Professional help access": ["therap", "counsel", "dietitian", "treatment", "appointment", "wait"],
            "Identity-specific support": ["lgbt", "bipoc", "male", "men", "athlete", "culture"],
        }
        wish_theme = keyword_bucket(df_student[S_wish], WISH_BUCKETS, top_n=10).rename(columns={"count": "students_unmet"})
        fut_feats  = top_counts_multi(df_future[F_retention_feats], top_n=10).rename(columns={"count": "future_retention_features"})

        c1, c2 = st.columns([1, 1])
        with c1:
            hbar(
                wish_theme,
                "Students: unmet needs (themes)",
                label_col="label",
                value_col="students_unmet",
                height=380
            )
        with c2:
            hbar(
                fut_feats,
                "Future users: retention features",
                label_col="label",
                value_col="future_retention_features",
                height=380
            )

        st.caption("Use this section to translate *user retention drivers* into *recovery-support needs* without mixing datasets blindly.")

    else:
        st.caption("Add more data (or ensure columns exist) to compute future-vs-student feature direction.")

# ───────────────────────────── Optional preview ─────────────────────────────
with st.expander("Data preview (sanitized)", expanded=False):
    st.caption("Combined dataset (PII-guarded). Showing last 20 rows.")
    safe_cols = [c for c in df.columns if not any(k in c.lower() for k in PII_FRAGMENTS)]
    st.dataframe(df[safe_cols].tail(20), use_container_width=True)
