#!/usr/bin/env python3
# Mirror — Live Insight Dashboard (Public Google Sheets · clear insights)
# Uses your two public Sheet URLs (with known gid) and auto-detects columns.

from pathlib import Path
import io, re, requests
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import html, unicodedata

def clean_text(s: str) -> str:
    """Fix common Google Sheets csv mojibake & smart quotes."""
    if s is None:
        return ""
    s = html.unescape(str(s))
    # common mojibake from UTF-8 → Latin-1 mismatch
    fixes = {
        "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€“": "–", "â€”": "—",
        "â€": '"', "Â": "", "â€¦": "…"
    }
    for bad, good in fixes.items():
        s = s.replace(bad, good)
    return unicodedata.normalize("NFKC", s).strip()


# ───────────────────────────── Page / Style ─────────────────────────────
st.set_page_config(page_title="Mirror — Live Insights", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Pastel

st.title("Mirror — Live Insight Dashboard")
st.caption("Data auto-loads from public Google Sheets (every 60s). Focused on clarity and insight.")

# Auto-refresh every 60s
st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

# ───────────────────────────── Your live Sheets ─────────────────────────────
# You gave these:
DATA_PROS_EDIT     = "https://docs.google.com/spreadsheets/d/13lY6kHhiJCJP6CBP2CQbtVQzuffS2mn-vXCXC9CtlYE/edit?gid=1896040034#gid=1896040034"
DATA_STUDENTS_EDIT = "https://docs.google.com/spreadsheets/d/1qin5S0V2beHcj3A2oV48nF_TX5pW73_M8IdIqx3HIVY/edit?gid=1750397413#gid=1750397413"

# ── Text cleanup helpers ─────────────────────────────────────────
import unicodedata

SMART_QUOTE_MAP = {
    "“":"\"", "”":"\"", "‘":"'", "’":"'", "—":"-", "–":"-",
    "…":"...", "•":"-", "\u00a0":" ", "\u200b":""  # nbsp/zwsp
}

def _fix_mojibake(s: str) -> str:
    # Many forms show artifacts like "didnâ€™t" when UTF-8 was read as CP1252/Latin-1
    try:
        t = s.encode("latin1").decode("utf-8")   # common fix path
    except Exception:
        t = s
    return t

def clean_text(s: str | float) -> str:
    if pd.isna(s): return ""
    t = str(s)
    t = _fix_mojibake(t)
    t = unicodedata.normalize("NFKC", t)
    for k,v in SMART_QUOTE_MAP.items():
        t = t.replace(k, v)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_df_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(clean_text)
    return df


# ───────────────────────────── Helpers ─────────────────────────────
PII_COL_FRAGMENTS = ("email","e-mail","name","first name","last name","contact","phone","address")
SPLIT_PAT = re.compile(r"[;,/|•]")

# Keyword maps for readable rollup buckets
BARRIER_KEYWORDS = {
    "Cost / Insurance": ["cost","expensive","insurance","coverage","copay","financial","money","out-of-network","deductible","authorization"],
    "Waitlists / Access": ["wait","waitlist","access","availability","appointment","capacity","time","scheduling","long"],
    "Stigma / Shame / Privacy": ["stigma","shame","privacy","confidential","embarrass","judg","afraid"],
    "Knowledge / Awareness": ["awareness","know","education","information","misinfo","understand"],
    "Distance / Transport": ["distance","transport","travel","commute","far"],
    "Motivation / Readiness": ["motivation","ready","ambivalence","willingness","denial"],
    "Comorbidity / Severity": ["depress","anxiety","ocd","self harm","suic","medical","severity"],
    "Fit / Quality of Care": ["fit","therapist","specialist","quality","approach","not listen","mismatch","availability"],
}
SUPPORT_KEYWORDS = {
    "Outpatient therapy": ["outpatient therapy","therapy","therapist","counseling","counsell"],
    "Group therapy / Peer": ["group therapy","peer","support group","community"],
    "Family therapy": ["family therapy","family support"],
    "Dietitian / Nutrition": ["dietitian","nutritionist","meal plan","nutrition","rd","food plan"],
    "CBT/DBT/FBT/ACT": ["cbt","dbt","fbt","act","ifs","somatic"],
    "Crisis / Hotline": ["crisis","hotline","988"],
    "Anonymous / Safe space": ["anonymous","safe space","anon"],
    "Education / Psychoeducation": ["education","psychoeducation","learn","information","resources"],
    "Care coordination / Interdisciplinary": ["coordinate","interdisciplinary","team","case manage","refer","triage"],
    "Higher level of care (IOP/PHP/Inpatient)": ["iop","php","inpatient","residential","hospital"],
}

WISH_THEMES = {
    "Listen / Validate (not 'just eat')": ["listen", "validate", "belittle", "just eat", "understood", "honest"],
    "Identity-specific support": ["lgbt", "male", "men", "bipoc", "identity", "boys", "arfid"],
    "Meal support / Practical help": ["meal", "meals", "cook", "grocery", "practical", "snack", "plan"],
    "Peer / Community chat": ["peer", "community", "chat", "support group", "mentor"],
    "Other": [],  # leave as sink bucket
}


# Lightweight column inference (header + sample values)
KEYS = {
    "age": ("how old","age"),
    "gender": ("gender","identify as"),
    "diagnosis": ("diagnosed","which type","type(s) of eating disorder"),
    "challenge": ("challenge","barrier","struggle","difficult","hardest","obstacle"),
    "helpful": ("actually helped","helped in your healing","support actually helped","most helpful"),
    "community": ("supportive community","support system","community support"),
    "wish": ("wish","would have","should have","i needed","i wish"),
    # pros
    "pros_approach": ("treatment approach","what does your treatment look like","modalities","cbt","dbt","fbt","provide"),
    "pros_recommend": ("recommend","resources","tools","refer","suggest"),
}

def extract_sheet_id(url: str) -> str | None:
    m = re.search(r"/d/([A-Za-z0-9-_]+)/", url)
    return m.group(1) if m else None

def to_export_csv_url(edit_url: str, gid: int | str) -> str:
    sid = extract_sheet_id(edit_url)
    return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}" if sid else edit_url

@st.cache_data(show_spinner=False, ttl=60)
def load_live_csv_from_edit(edit_url: str, gid: int | str) -> tuple[pd.DataFrame, dict]:
    """Fetch live CSV via export endpoint; return (df, debug)."""
    url = to_export_csv_url(edit_url, gid)
    debug = {"export_url": url}
    try:
        r = requests.get(url, timeout=20)
        debug["status"] = r.status_code
        text = r.text
        df = pd.read_csv(io.StringIO(text), dtype=str, engine="python", on_bad_lines="skip")
    except Exception as e:
        debug["error"] = str(e)
        df = pd.DataFrame()

    # Hygiene: drop dup headers, PII, and empty cols
    if not df.empty:
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
        drop_cols = [c for c in df.columns if any(k in c.lower() for k in PII_COL_FRAGMENTS)]
        df = df.drop(columns=drop_cols, errors="ignore")
        # normalize all strings (fix mojibake, quotes, whitespace)
        df = clean_df_strings(df)

        def _all_empty(s: pd.Series) -> bool:
            return s.dropna().astype(str).str.strip().eq("").all()
        df = df[[c for c in df.columns if not _all_empty(df[c])]]

    debug["preview"] = (text[:400] if 'text' in locals() else "")
    return df, debug

def score(text: str, inc: tuple[str,...]) -> int:
    t = (text or "").lower()
    return sum(2 for k in inc if k in t)

def infer_col(df: pd.DataFrame, key: str) -> str | None:
    inc = KEYS.get(key, ())
    best, best_score = None, 0
    for c in df.columns:
        h = score(c, inc)
        v = score(" ".join(df[c].dropna().astype(str).head(25).tolist()), inc)
        s = h*3 + v
        if s > best_score:
            best, best_score = c, s
    return best

def explode_multi(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=str)
    s = s.dropna().astype(str)
    parts = s.apply(lambda x: SPLIT_PAT.split(x) if SPLIT_PAT.search(x) else [x])
    parts = pd.Series([p.strip() for lst in parts for p in lst if p and p.strip()])
    return parts

def normalize_counts(series: pd.Series, mapping: dict[str,list[str]], top_n=12) -> pd.DataFrame:
    if series is None or series.empty:
        return pd.DataFrame(columns=["label","count"])
    raw = series.astype(str).str.lower()
    counts = Counter(raw)
    buckets, leftover = Counter(), Counter()
    for text, c in counts.items():
        matched = False
        for bucket, keys in mapping.items():
            if any(k in text for k in keys):
                buckets[bucket] += c
                matched = True
        if not matched:
            leftover[text] += c
    for text, c in leftover.most_common(8):
        label = f"Other: {text[:45]}{'…' if len(text)>45 else ''}"
        buckets[label] += c
    out = pd.DataFrame([{"label": k, "count": v} for k, v in buckets.items()]) \
            .sort_values("count", ascending=False).head(top_n).reset_index(drop=True)
    return out

def series_or_empty(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].dropna().astype(str) if (col and col in df.columns) else pd.Series(dtype=str)

def short_label(s: str, maxlen=48) -> str:
    s = (s or "").strip()
    return s if len(s) <= maxlen else s[:maxlen-1] + "…"

# ── Per-row wish classification + n-gram mining ───────────────────────────────
STOPWORDS = set("""
a an the and or but if when while with without into onto for to of in on at by as
be is are was were been being do does did have has had can could should would may might
me my mine you your yours he she it its we our ours they their them this that these those
just really very not no yes more most less least than then else from about over under out
""".split())

def classify_wish_texts(series: pd.Series, mapping: dict[str, list[str]]) -> pd.DataFrame:
    """
    Return a DataFrame [text, theme] by checking each row against mapping.
    If multiple themes match, keep the first (order of mapping in code is priority).
    """
    rows = []
    for raw in series.dropna().astype(str):
        txt = clean_text(raw).lower()
        assigned = None
        for theme, keys in mapping.items():
            if any(k in txt for k in keys):
                assigned = theme
                break
        rows.append((raw, assigned if assigned else "Other"))
    return pd.DataFrame(rows, columns=["text", "theme"])

def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z']+", (text or "").lower()) if t not in STOPWORDS and len(t) > 2]

def ngram_counts(texts: list[str], n: int = 2) -> Counter:
    """
    Count frequent n-grams (bigrams/trigrams) from texts.
    """
    bag = Counter()
    for t in texts:
        toks = _tokens(t)
        for i in range(len(toks) - n + 1):
            ng = " ".join(toks[i:i+n])
            bag[ng] += 1
    return bag

def auto_promote_from_other(other_texts: list[str], promote_threshold: int = 3, max_promote: int = 4):
    """
    Look for repeated bigrams / trigrams in 'Other' and build new themes.
    Returns list[(label, keywords)] of promotions.
    """
    if not other_texts:
        return []

    bi = ngram_counts(other_texts, 2)
    tri = ngram_counts(other_texts, 3)

    # merge and rank
    all_ = bi + tri
    cands = [(phrase, cnt) for phrase, cnt in all_.most_common(30) if cnt >= promote_threshold]

    promotions = []
    for phrase, _ in cands:
        label = phrase.title()
        keys = phrase.split()  # use words from the phrase as detection seeds
        # small guard against duplicates
        if label not in WISH_THEMES and all(not set(keys).issubset(set(v)) for v in WISH_THEMES.values()):
            promotions.append((label, keys))
        if len(promotions) >= max_promote:
            break
    return promotions


# ───────────────────────────── Load live data ─────────────────────────────
# Use the gid values you supplied
students, dbg_students = load_live_csv_from_edit(DATA_STUDENTS_EDIT, gid=1750397413)
pros,     dbg_pros     = load_live_csv_from_edit(DATA_PROS_EDIT,     gid=1896040034)

with st.expander("Debug · Live fetch (click if something looks off)", expanded=False):
    st.subheader("Students")
    st.json(dbg_students)
    st.write("shape:", students.shape)
    st.subheader("Professionals")
    st.json(dbg_pros)
    st.write("shape:", pros.shape)

if students.empty and pros.empty:
    st.error("Could not read any rows from either Google Sheet.")
    st.stop()

# ───────────────────────────── Auto-infer key columns ─────────────────────────────
# Individuals
S_age        = infer_col(students, "age")
S_gender     = infer_col(students, "gender")
S_diag       = infer_col(students, "diagnosis")
S_challenge  = infer_col(students, "challenge")
S_helpful    = infer_col(students, "helpful")
S_wish       = infer_col(students, "wish")
S_comm       = infer_col(students, "community")
# Professionals
P_role       = infer_col(pros, "pros_approach")
P_reco       = infer_col(pros, "pros_recommend")

# --- Build the Individuals "wishes" series we'll use in D ---
S_wishes_series = series_or_empty(students, S_wish)
# (optional but recommended) clean up mojibake / smart quotes
S_wishes_series = S_wishes_series.map(lambda x: clean_text(x) if pd.notna(x) else x)


with st.expander("Auto-detected columns", expanded=False):
    st.json({
        "students": {
            "age": S_age, "gender": S_gender, "diagnosis": S_diag,
            "challenge": S_challenge, "helpful": S_helpful, "wish": S_wish, "community": S_comm
        },
        "professionals": {
            "treatment_approach": P_role, "recommended": P_reco
        }
    })

# ───────────────────────────── A) Overview ─────────────────────────────
st.markdown("### A) Overview — Who responded")
col1, col2, col3 = st.columns(3)

with col1:
    n_students, n_pros = len(students), len(pros)
    comp = pd.DataFrame({"group":["Individuals","Professionals"], "count":[n_students, n_pros]})
    fig = px.pie(comp, values="count", names="group", hole=0.45)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Total responses: **{n_students+n_pros:,}** (Individuals: {n_students:,}, Professionals: {n_pros:,})")

with col2:
    if S_age and S_age in students.columns:
        ages = students[S_age].dropna().astype(str)
        age_num = pd.to_numeric(ages.str.extract(r"(\d+)")[0], errors="coerce").dropna()
        if not age_num.empty:
            labels = ["≤17", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
            bins = pd.cut(age_num, bins=[0,17,24,34,44,54,64,150],
                          labels=labels, include_lowest=True, right=True)
            dist = (bins.value_counts()
                        .reindex(labels)
                        .reset_index())
            dist.columns = ["Age band", "count"]
            fig = px.bar(dist, x="Age band", y="count")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Individuals — median age: **{int(age_num.median())}** (n={len(age_num)})")
        else:
            st.info("Age responses not numeric enough to summarize.")
    else:
        st.info("No age column detected in Individuals data.")


with col3:
    if S_gender and S_gender in students.columns:
        g = students[S_gender].dropna().astype(str).str.strip()
        g = g[g!=""]
        if not g.empty:
            top = g.value_counts().head(6).reset_index()
            top.columns = ["gender","count"]
            fig = px.bar(top, x="gender", y="count")
            fig.update_layout(xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No gender responses.")
    else:
        st.info("No gender column detected in Individuals data.")

# ───────────────────────────── B) Barriers ─────────────────────────────
st.markdown("---")
st.markdown("### B) Barriers to Recovery — Combined View")

def as_items(s: pd.Series) -> pd.Series:
    if s.empty: return s
    multi_rate = s.str.contains(SPLIT_PAT).mean()
    return explode_multi(s) if multi_rate > 0.2 else s

ind_barriers = as_items(series_or_empty(students, S_challenge))
# If pros don't have a barrier-specific field, their "recommendations" often include access notes; still useful signal.
pro_barriers = as_items(series_or_empty(pros, P_reco))

ind_roll = normalize_counts(ind_barriers, BARRIER_KEYWORDS, top_n=12)
pro_roll = normalize_counts(pro_barriers, BARRIER_KEYWORDS, top_n=12)
if not (ind_roll.empty and pro_roll.empty):
    ind_roll["who"] = "Individuals"
    pro_roll["who"] = "Professionals"
    barriers = pd.concat([ind_roll, pro_roll], ignore_index=True)
    order = (barriers[barriers["who"]=="Individuals"]
             .sort_values("count", ascending=False)["label"].tolist())
    fig = px.bar(barriers, x="label", y="count", color="who", barmode="group",
                 category_orders={"label": order})
    fig.update_layout(xaxis_title="", yaxis_title="Mentions (approx.)")
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)
    take_ind = ind_roll.head(3).assign(label=lambda d: d["label"].apply(short_label))
    take_pro = pro_roll.head(3).assign(label=lambda d: d["label"].apply(short_label))
    st.caption(
        f"Top barriers — **Individuals:** {', '.join(take_ind['label'].tolist())} · "
        f"**Professionals:** {', '.join(take_pro['label'].tolist())}"
    )
else:
    st.info("No barrier-style text detected.")

# ───────────────────────────── C) Helpful Supports & Alignment ─────────────────────────────
st.markdown("---")
st.markdown("### C) Helpful Supports & Alignment (Individuals vs Professionals)")

ind_help_items = as_items(series_or_empty(students, S_helpful))
pro_help_items = as_items(series_or_empty(pros, P_role))  # treatment approach = what pros provide

ind_sup = normalize_counts(ind_help_items, SUPPORT_KEYWORDS, top_n=12)
pro_sup = normalize_counts(pro_help_items, SUPPORT_KEYWORDS, top_n=12)

merged = pd.merge(ind_sup, pro_sup, on="label", how="outer", suffixes=("_ind","_pro")).fillna(0)
if not merged.empty:
    merged["diff"] = merged["count_ind"] - merged["count_pro"]
    merged = merged.sort_values("count_ind", ascending=False)
    fig = px.bar(
        merged.melt(id_vars=["label"], value_vars=["count_ind","count_pro"],
                    var_name="who", value_name="count"),
        x="label", y="count", color="who", barmode="group",
        category_orders={"label": merged["label"].tolist()},
        labels={"who":"Group"}
    )
    fig.update_layout(xaxis_title="", yaxis_title="Mentions (approx.)")
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    top_help_ind = ", ".join(merged.head(3)["label"].tolist())
    biggest_gap = merged.sort_values("diff", ascending=False).head(1)
    if not biggest_gap.empty:
        gap_lbl = biggest_gap.iloc[0]["label"]
        gap_val = int(biggest_gap.iloc[0]["diff"])
        if gap_val > 0:
            st.caption(f"Individuals emphasize **{gap_lbl}** more than professionals by ~{gap_val} mentions.")
        elif gap_val < 0:
            st.caption(f"Professionals emphasize **{gap_lbl}** more than individuals by ~{abs(gap_val)} mentions.")
    st.caption(f"Most helpful for individuals: **{top_help_ind}**")
else:
    st.info("Not enough helpful/support text to compare.")

# ───────────────────────────── D) Unmet Needs / Wishes for Support ─────────────────────────────
st.markdown("### D) Unmet Needs / Wishes for Support (Individuals)")

# 1) Per-row classification first (authoritative source of truth)
wish_rows = classify_wish_texts(S_wishes_series, WISH_THEMES)
wish_buckets = (wish_rows["theme"]
                .value_counts()
                .rename_axis("theme")
                .reset_index(name="count")
                .sort_values("count", ascending=False))

# 2) Main chart (always derived from per-row assignments)
fig = px.bar(wish_buckets, x="theme", y="count")
fig.update_layout(xaxis_title="", yaxis_title="Mentions (approx.)")
fig.update_xaxes(tickangle=30)
st.plotly_chart(fig, use_container_width=True)

# 3) Deep dive into 'Other': mine phrases, auto-promote, and rebuild if needed
st.markdown("#### Deep Dive — What ‘Other’ Really Means")
other_texts = wish_rows.loc[wish_rows["theme"] == "Other", "text"].astype(str).tolist()

if not other_texts:
    st.info("No 'Other' responses found.")
else:
    # show a quick n-gram summary
    bi = ngram_counts(other_texts, 2)
    tri = ngram_counts(other_texts, 3)
    top_phrases = pd.DataFrame(
        [{"phrase": p, "count": c} for p, c in (bi + tri).most_common(12)]
    )
    if not top_phrases.empty:
        fig = px.bar(top_phrases.sort_values("count", ascending=False), x="phrase", y="count")
        fig.update_layout(xaxis_title="", yaxis_title="Mentions (approx.)")
        fig.update_xaxes(tickangle=25)
        st.plotly_chart(fig, use_container_width=True)

    # auto-promote recurring phrases
    PROMOTE_THRESHOLD = 3
    promotions = auto_promote_from_other(other_texts, promote_threshold=PROMOTE_THRESHOLD, max_promote=4)

    if promotions:
        # mutate mapping in-memory, then reclassify + redraw
        for label, keys in promotions:
            WISH_THEMES[label] = keys
        st.success(
            "Promoted new themes from ‘Other’: " +
            "; ".join([f"**{lbl}** ({', '.join(keys)})" for lbl, keys in promotions])
        )

        # Re-run classification with the updated mapping
        wish_rows = classify_wish_texts(S_wishes_series, WISH_THEMES)
        wish_buckets = (wish_rows["theme"]
                        .value_counts()
                        .rename_axis("theme")
                        .reset_index(name="count")
                        .sort_values("count", ascending=False))
        
        fig = px.bar(wish_buckets, x="theme", y="count")
        fig.update_layout(title="D) Unmet Needs / Wishes for Support (Updated with promoted themes)",
                          xaxis_title="", yaxis_title="Mentions (approx.)")
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    # optional: show a few example lines for the biggest buckets (excluding 'Other')
    st.markdown("##### Example quotes (sampled)")
    show_themes = [t for t in wish_buckets["theme"].tolist() if t != "Other"][:5]
    for t in show_themes:
        ex = wish_rows.loc[wish_rows["theme"] == t, "text"].head(2).tolist()
        if not ex:
            continue
        st.markdown(f"**{t}** — {len(wish_rows[wish_rows['theme']==t])} mention(s)")
        for q in ex:
            st.caption(f"• “{clean_text(q)}”")



# ───────────────────────────── E) Treatment Gaps — Wishes vs Pros’ Recommendations ─────────────────────────────
st.markdown("## E) Treatment Gaps — Individuals’ Wishes vs Professionals’ Recommendations")

# Map pros into comparable themes (can overlap SUPPORT_KEYWORDS but keep concise)
PRO_THEMES = {
    "Peer / Community chat": ["peer","community","group"],
    "Education / Resources": ["educat","resource","psychoeducation","handout","guide","workshop","neda"],
    "Care coordination / Referrals": ["coordinate","referral","refer","triage","network","interdisciplinary"],
    "Therapy / Modalities": ["cbt","dbt","fbt","act","ifs","somatic","therapy","psychotherapy","counsel"],
    "Meal support / Plans": ["meal plan","meal support","nutrition","dietitian","rd"],
    "Higher level of care (IOP/PHP/Inpatient)": ["iop","php","inpatient","residential","partial"],
    "Access / Cost / Navigation": ["wait","waitlist","access","insurance","coverage","authorization","navigation"],
    "Identity-specific support": ["lgbt","bipoc","men","boys","athlete","cultural","religion"],
}

def bucketize_generic(series: pd.Series, themes: dict[str, list[str]]):
    if series is None or series.empty:
        return pd.DataFrame(columns=["theme","count"])
    rows = []
    for text in series.dropna().astype(str):
        t = clean_text(text).lower()
        hits = [theme for theme, keys in themes.items() if any(k in t for k in keys)]
        if not hits: hits = ["Other"]
        rows.extend(hits)
    vc = pd.Series(rows).value_counts().reset_index()
    vc.columns = ["theme","count"]
    return vc

wish_counts = bucketize_generic(S_wishes_series, WISH_THEMES)
pro_counts  = bucketize_generic(series_or_empty(pros, P_reco), PRO_THEMES)

gap = pd.merge(wish_counts, pro_counts, on="theme", how="outer", suffixes=("_wish","_pro")).fillna(0)
if gap.empty:
    st.info("Not enough content to compare.")
else:
    gap["delta_wish_minus_pro"] = gap["count_wish"] - gap["count_pro"]
    # clear sort: put biggest patient-asked themes first
    gap = gap.sort_values(["count_wish","count_pro"], ascending=[False, False])

    fig = px.bar(
        gap.melt(id_vars=["theme"], value_vars=["count_wish","count_pro"], var_name="Group", value_name="Mentions"),
        x="theme", y="Mentions", color="Group",
        category_orders={"theme": gap["theme"].tolist()}
    )
    fig.update_layout(xaxis_title="", yaxis_title="Mentions (approx.)")
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    # concise takeaway
    more_wish = gap.sort_values("delta_wish_minus_pro", ascending=False).head(3)
    more_pro  = gap.sort_values("delta_wish_minus_pro").head(3)
    st.caption(
        "Individuals desire more emphasis on: **" +
        ", ".join(more_wish["theme"].tolist()) +
        "**. Professionals emphasize (relative to wishes): **" +
        ", ".join(more_pro["theme"].tolist()) +
        "**."
    )


# ───────────────────────────── Recent rows (sanitized) ─────────────────────────────
st.markdown("---")
st.subheader("Recent Responses (last 25, PII removed)")
def strip_pii(df: pd.DataFrame) -> pd.DataFrame:
    safe = [c for c in df.columns if not any(k in c.lower() for k in PII_COL_FRAGMENTS)]
    return df[safe]
if not students.empty:
    st.write("Individuals")
    st.dataframe(strip_pii(students).tail(25), use_container_width=True)
if not pros.empty:
    st.write("Professionals")
    st.dataframe(strip_pii(pros).tail(25), use_container_width=True)
