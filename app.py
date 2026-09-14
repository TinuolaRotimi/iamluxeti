"""
Archive Search Tool
-------------------
A production-grade Streamlit app for exploring a large text archive
containing 1.9M+ posts and comments.

Data source: private research dataset
Deployed on: Streamlit Community Cloud
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO_ID = "just3nu/RepLadies_Archive"
HF_FILENAME = "repladies_clean.parquet"
HF_REPO_TYPE = "dataset"

RESULTS_PER_PAGE = 25
MAX_BODY_PREVIEW_CHARS = 700
MIN_QUERY_LENGTH = 2

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Archive Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark Theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    h1, h2, h3 {
        color: #ff4500 !important;
        letter-spacing: -0.02em;
    }
    h1 { font-weight: 800; }
    p, li, span, div { color: #c9d1d9; }

    .stTextInput > div > div > input {
        background-color: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 8px;
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
    }
    .result-body mark {
        background: #ff4500;
        color: #ffffff;
        padding: 1px 4px;
        border-radius: 3px;
    }

    hr { border-color: #30363d; }

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
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Download and cache the parquet dataset."""
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type=HF_REPO_TYPE,
    )
    df = pd.read_parquet(path)

    df["body"] = df["body"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["author"] = df["author"].fillna("unknown").astype(str)

    df["_search_blob"] = (df["body"] + " " + df["title"]).str.lower()

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def highlight(text: str, query: str) -> str:
    if not text:
        return ""
    escaped = html.escape(text)
    terms = [t for t in re.split(r"\s+", query.strip()) if len(t) >= 2]
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        escaped = pattern.sub(
            lambda m: f"<mark>{m.group(0)}</mark>",
            escaped,
        )
    return escaped


def format_date(ts) -> str:
    try:
        ts = int(ts)
        if ts <= 0:
            return "unknown date"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %d, %Y")
    except Exception:
        return "unknown date"


def truncate(text: str, limit: int = MAX_BODY_PREVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def render_result_card(row: pd.Series, query: str) -> str:
    typ = row.get("type", "comment")
    badge_class = "badge-post" if typ == "post" else "badge-comment"
    badge_label = typ.upper()

    author = html.escape(str(row.get("author", "unknown")))
    score = int(row.get("score", 0) or 0)
    date_str = format_date(row.get("created_utc", 0))

    title_html = ""
    title_raw = str(row.get("title", "")).strip()
    if title_raw:
        title_html = f'<div class="result-title">{highlight(title_raw, query)}</div>'

    body_raw = truncate(str(row.get("body", "")).strip())
    body_html = f'<div class="result-body">{highlight(body_raw, query)}</div>' if body_raw else ""

    return f"""
    <div class="result-card">
        <div class="result-meta">
            <span class="{badge_class}">{badge_label}</span>
            <span>by <span class="author">u/{author}</span></span>
            <span>· {date_str}</span>
            <span>· <span class="upvotes">▲ {score}</span></span>
        </div>
        {title_html}
        {body_html}
    </div>
    """


def filter_and_sort(
    df: pd.DataFrame,
    query: str,
    include_posts: bool,
    include_comments: bool,
    sort_mode: str,
    min_score: int,
) -> pd.DataFrame:
    q = query.strip().lower()
    mask = df["_search_blob"].str.contains(q, regex=False, na=False)
    results = df[mask].copy()

    if not include_posts:
        results = results[results["type"] != "post"]
    if not include_comments:
        results = results[results["type"] != "comment"]
    if min_score > 0:
        results = results[results["score"] >= min_score]

    if sort_mode == "Newest":
        results = results.sort_values("created_utc", ascending=False)
    elif sort_mode == "Top scored":
        results = results.sort_values("score", ascending=False)

    return results


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        df = load_data()
    except Exception as exc:
        st.error(f"❌ Failed to load dataset.\n\n{exc}")
        st.stop()

    st.title("📚 Archive Search")
    st.caption(
        f"A searchable archive of {len(df):,} historical records."
    )

    with st.sidebar:
        st.markdown("### 🎛️ Filters")

        include_posts = st.checkbox("Include posts", value=True)
        include_comments = st.checkbox("Include comments", value=True)

        sort_mode = st.radio(
            "Sort by",
            options=["Relevance", "Newest", "Top scored"],
            index=1,
        )

        min_score = st.slider(
            "Minimum score",
            min_value=0,
            max_value=500,
            value=0,
            step=10,
        )

        st.markdown("---")
        st.markdown("### 📊 Archive Stats")
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-label">Total records</div>
                <div class="stat-value">{len(df):,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Posts</div>
                <div class="stat-value">{(df['type'] == 'post').sum():,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Comments</div>
                <div class="stat-value">{(df['type'] == 'comment').sum():,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    query = st.text_input(
        "🔍 Search the archive",
        placeholder="Type any keyword to search the archive…",
        label_visibility="collapsed",
    )

    if not query or len(query.strip()) < MIN_QUERY_LENGTH:
        st.info(
            "👆 Type at least **2 characters** in the search bar to begin."
        )
        st.stop()

    with st.spinner("Searching…"):
        results = filter_and_sort(
            df=df,
            query=query,
            include_posts=include_posts,
            include_comments=include_comments,
            sort_mode=sort_mode,
            min_score=min_score,
        )

    total_results = len(results)

    if total_results == 0:
        st.warning(f"No results found for **{query}**.")
        st.caption("Try a shorter keyword, or widen your filters in the sidebar.")
        st.stop()

    st.markdown(
        f"### {total_results:,} result{'s' if total_results != 1 else ''} "
        f"for **{html.escape(query)}**"
    )

    csv_bytes = results.head(1000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download first 1,000 results as CSV",
        data=csv_bytes,
        file_name=f"archive_{query[:30].replace(' ', '_')}.csv",
        mime="text/csv",
    )

    total_pages = max(1, (total_results - 1) // RESULTS_PER_PAGE + 1)
    page = st.number_input(
        f"Page (1 – {total_pages})",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )

    start = (page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE
    page_results = results.iloc[start:end]

    for _, row in page_results.iterrows():
        st.markdown(render_result_card(row, query), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer">
            Archive Search · Historical data preserved for research purposes.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
