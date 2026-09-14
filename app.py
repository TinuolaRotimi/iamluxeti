"""
Archive Search — Streamlit App
===============================
A high-performance search interface for a large historical dataset
hosted on Hugging Face. Loads once into memory, then answers
queries instantly with relevance scoring, filters, sorting,
pagination, and CSV export.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ============================================================
# Configuration
# ============================================================
HF_REPO_ID = "just3nu/RepLadies_Archive"
HF_FILENAME = "repladies_clean.parquet"
HF_REPO_TYPE = "dataset"

RESULTS_PER_PAGE = 25
MAX_BODY_PREVIEW = 700
MIN_QUERY_LENGTH = 2

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Archive Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — dark theme
# ============================================================
st.markdown(
    """
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    h1, h2, h3 { color: #ff4500 !important; letter-spacing: -0.02em; }
    h1 { font-weight: 800; }
    p, li, span, div { color: #c9d1d9; }

    .stTextInput > div > div > input {
        background-color: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 10px;
        font-size: 16px;
        padding: 12px 14px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ff4500;
        box-shadow: 0 0 0 2px rgba(255, 69, 0, 0.2);
    }

    .result-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
        transition: border-color 0.15s ease, transform 0.15s ease;
    }
    .result-card:hover {
        border-color: #ff4500;
        transform: translateY(-1px);
    }
    .result-meta {
        font-size: 13px;
        color: #8b949e;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .result-meta .author { color: #58a6ff; font-weight: 600; }
    .result-meta .upvotes { color: #ff4500; font-weight: 600; }
    .badge-post {
        display: inline-block;
        background: #ff4500;
        color: #ffffff;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 5px;
        letter-spacing: 0.5px;
    }
    .badge-comment {
        display: inline-block;
        background: #30363d;
        color: #c9d1d9;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 5px;
        letter-spacing: 0.5px;
    }
    .result-title {
        font-size: 18px;
        font-weight: 700;
        color: #e6edf3;
        margin: 6px 0 8px 0;
        line-height: 1.35;
    }
    .result-body {
        font-size: 15px;
        line-height: 1.6;
        color: #c9d1d9;
        word-wrap: break-word;
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }
    .result-body a, .result-title a {
        color: #58a6ff;
        text-decoration: none;
        border-bottom: 1px dotted #58a6ff;
    }
    .result-body a:hover, .result-title a:hover { color: #79c0ff; }
    mark {
        background: #ff4500;
        color: #fff;
        padding: 1px 4px;
        border-radius: 3px;
        font-weight: 600;
    }
    .footer {
        color: #6e7681;
        font-size: 12px;
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #30363d;
    }
    .stat-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .stat-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value {
        font-size: 18px;
        color: #ff4500;
        font-weight: 700;
    }
    .chip {
        display: inline-block;
        background: #161b22;
        border: 1px solid #30363d;
        color: #c9d1d9;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 13px;
        margin: 2px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers
# ============================================================
def escape_html(text: str) -> str:
    return html.escape(str(text or ""))


URL_REGEX = re.compile(r"(https?://[^\s<>\"')\]]+)", re.IGNORECASE)


def linkify(escaped_text: str) -> str:
    """Turn bare URLs into clickable links (already-escaped input)."""

    def repl(match: re.Match) -> str:
        url = match.group(1)
        trailing = ""
        while url and url[-1] in ".,;:!?)":
            trailing = url[-1] + trailing
            url = url[:-1]
        return (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
            f"{url}</a>{trailing}"
        )

    return URL_REGEX.sub(repl, escaped_text)


def highlight(escaped_text: str, query: str) -> str:
    """Highlight query terms in already-escaped text (skips inside tags)."""
    terms = [t for t in re.split(r"\s+", query.strip()) if len(t) >= 2]
    if not terms:
        return escaped_text

    parts = re.split(r"(<[^>]+>)", escaped_text)
    for i, part in enumerate(parts):
        if part.startswith("<"):
            continue
        for term in terms:
            part = re.sub(
                re.escape(term),
                lambda m: f"<mark>{m.group(0)}</mark>",
                part,
                flags=re.IGNORECASE,
            )
        parts[i] = part
    return "".join(parts)


def format_date(ts) -> str:
    try:
        n = int(ts)
        if n <= 0:
            return "unknown"
        return datetime.fromtimestamp(n, tz=timezone.utc).strftime("%b %d, %Y")
    except Exception:
        return "unknown"


def truncate(text: str, limit: int = MAX_BODY_PREVIEW) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def render_card(row: pd.Series, query: str) -> str:
    is_post = row.get("type") == "post"
    badge = (
        '<span class="badge-post">POST</span>'
        if is_post
        else '<span class="badge-comment">COMMENT</span>'
    )

    author = escape_html(row.get("author", "unknown"))
    score = int(row.get("score", 0) or 0)
    date_str = format_date(row.get("created_utc", 0))

    title_raw = str(row.get("title", "")).strip()
    title_html = ""
    if title_raw:
        t = highlight(linkify(escape_html(title_raw)), query)
        title_html = f'<div class="result-title">{t}</div>'

    body_raw = truncate(str(row.get("body", "")).strip())
    body_html = ""
    if body_raw:
        b = highlight(linkify(escape_html(body_raw)), query)
        body_html = f'<div class="result-body">{b}</div>'

    return f"""
    <div class="result-card">
        <div class="result-meta">
            {badge}
            <span>by <span class="author">u/{author}</span></span>
            <span>· {date_str}</span>
            <span>· <span class="upvotes">▲ {score}</span></span>
        </div>
        {title_html}
        {body_html}
    </div>
    """


# ============================================================
# Data loading (cached)
# ============================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Download and cache the parquet dataset from Hugging Face."""
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type=HF_REPO_TYPE,
    )
    df = pd.read_parquet(path)

    # Normalize column types
    df["body"] = df["body"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["author"] = df["author"].fillna("unknown").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce").fillna(0).astype(int)

    # Precompute lowercase search blob
    df["_search"] = (df["body"] + " " + df["title"]).str.lower()

    return df


# ============================================================
# Load
# ============================================================
try:
    with st.spinner("Loading 1.98M records from Hugging Face… (this happens once)"):
        df = load_data()
except Exception as exc:
    st.error("❌ Failed to load the dataset from Hugging Face.")
    st.code(str(exc))
    st.stop()

total_records = len(df)
total_posts = int((df["type"] == "post").sum())
total_comments = int((df["type"] == "comment").sum())

# ============================================================
# Header
# ============================================================
st.title("📚 Archive Search")
st.caption(f"A searchable snapshot of {total_records:,} historical records.")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filters")

    include_posts = st.checkbox("Include posts", value=True)
    include_comments = st.checkbox("Include comments", value=True)

    sort_mode = st.radio(
        "Sort by",
        ["Top scored", "Newest", "Oldest", "Relevance"],
        index=0,
    )

    min_score = st.slider("Minimum score", 0, 500, 0, step=5)

    st.markdown("---")
    st.markdown("### 📈 Archive Stats")

    st.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-label">Total records</div>
            <div class="stat-value">{total_records:,}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Posts</div>
            <div class="stat-value">{total_posts:,}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Comments</div>
            <div class="stat-value">{total_comments:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# Search input
# ============================================================
query = st.text_input(
    "🔍 Search the archive",
    placeholder="Try: zimmermann, birkin, taobao, chanel…",
    label_visibility="collapsed",
)

if not query or len(query.strip()) < MIN_QUERY_LENGTH:
    st.info("👆 Type at least **2 characters** in the search bar to begin.")

    st.markdown("**Popular searches to try:**")
    st.markdown(
        """
        <span class="chip">zimmermann</span>
        <span class="chip">birkin</span>
        <span class="chip">taobao</span>
        <span class="chip">chanel 19</span>
        <span class="chip">yupoo</span>
        <span class="chip">weidian</span>
        <span class="chip">hermes</span>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ============================================================
# Search + filter + sort
# ============================================================
q = query.strip().lower()

with st.spinner("Searching…"):
    mask = df["_search"].str.contains(q, regex=False, na=False)
    results = df[mask]

    if not include_posts:
        results = results[results["type"] != "post"]
    if not include_comments:
        results = results[results["type"] != "comment"]
    if min_score > 0:
        results = results[results["score"] >= min_score]

    if sort_mode == "Top scored":
        results = results.sort_values("score", ascending=False)
    elif sort_mode == "Newest":
        results = results.sort_values("created_utc", ascending=False)
    elif sort_mode == "Oldest":
        results = results.sort_values("created_utc", ascending=True)
    else:  # Relevance — approximate: high score first, recent tie-break
        results = results.sort_values(["score", "created_utc"], ascending=[False, False])

total_results = len(results)

if total_results == 0:
    st.warning(f"No results for **{query}**.")
    st.caption("Try a shorter keyword, or widen your filters in the sidebar.")
    st.stop()

# ============================================================
# Header row + CSV download
# ============================================================
col_left, col_right = st.columns([3, 1])
with col_left:
    st.markdown(
        f"### {total_results:,} result"
        f"{'s' if total_results != 1 else ''} for **{escape_html(query)}**"
    )
with col_right:
    csv_bytes = results.head(1000).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download first 1,000 (CSV)",
        data=csv_bytes,
        file_name=f"archive_{re.sub(r'[^a-zA-Z0-9]+', '_', query)[:40]}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# Pagination
# ============================================================
total_pages = max(1, (total_results - 1) // RESULTS_PER_PAGE + 1)

page = st.number_input(
    f"Page (1 – {total_pages})",
    min_value=1,
    max_value=total_pages,
    value=1,
    step=1,
    help="Use the arrows to move through pages.",
)

start = (page - 1) * RESULTS_PER_PAGE
end = start + RESULTS_PER_PAGE
page_results = results.iloc[start:end]

# ============================================================
# Render results
# ============================================================
for _, row in page_results.iterrows():
    st.markdown(render_card(row, query), unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown(
    """
    <div class="footer">
        Archive Search · Historical data preserved for research purposes
    </div>
    """,
    unsafe_allow_html=True,
)
