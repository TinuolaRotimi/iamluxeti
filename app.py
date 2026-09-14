"""
Archive Search — Streamlit App (All Features)
==============================================
Chunked loading, click-to-expand, date filter, author filter,
score filter, phrase search, CSV export, Reddit links, copy
buttons, parent post display, recent searches, URL params.
"""

from __future__ import annotations

import html
import io
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from huggingface_hub import hf_hub_download

# ============================================================
# Config
# ============================================================
HF_REPO_ID = "just3nu/RepLadies_Archive"
HF_FILENAME = "repladies_clean.parquet"
HF_REPO_TYPE = "dataset"

RESULTS_PER_PAGE = 25
MAX_MATCHES = 5000
MAX_BODY_PREVIEW = 400
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
# CSS
# ============================================================
st.markdown(
    """
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    h1, h2, h3 { color: #ff4500 !important; }
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
        margin-bottom: 8px;
    }
    .result-card:hover { border-color: #ff4500; }

    .result-meta {
        font-size: 13px;
        color: #8b949e;
        margin-bottom: 8px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }
    .result-meta .author { color: #58a6ff; font-weight: 600; }
    .result-meta .upvotes { color: #ff4500; font-weight: 600; }
    .result-meta a { color: #58a6ff; text-decoration: none; }

    .badge-post {
        display: inline-block;
        background: #ff4500;
        color: #ffffff;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 5px;
    }
    .badge-comment {
        display: inline-block;
        background: #30363d;
        color: #c9d1d9;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 5px;
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
    .parent-ref {
        font-size: 12px;
        color: #6e7681;
        font-style: italic;
        margin-bottom: 6px;
    }
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
        cursor: pointer;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Static stats
# ============================================================
TOTAL_RECORDS = 1_986_244
TOTAL_POSTS = 62_722
TOTAL_COMMENTS = 1_923_522

# ============================================================
# Helpers
# ============================================================
def escape_html(text) -> str:
    return html.escape(str(text or ""))


URL_REGEX = re.compile(r"(https?://[^\s<>\"')\]]+)", re.IGNORECASE)


def linkify(escaped: str) -> str:
    def repl(m):
        url = m.group(1)
        trailing = ""
        while url and url[-1] in ".,;:!?)":
            trailing = url[-1] + trailing
            url = url[:-1]
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>{trailing}'
    return URL_REGEX.sub(repl, escaped)


def highlight(escaped: str, query: str) -> str:
    terms = [t for t in re.split(r"\s+", query.strip()) if len(t) >= 2]
    if not terms:
        return escaped
    parts = re.split(r"(<[^>]+>)", escaped)
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


# ============================================================
# Download parquet once, split into 4 chunks on disk
# ============================================================
CHUNK_DIR = "/tmp/repladies_chunks"


@st.cache_resource(show_spinner=False)
def prepare_chunks() -> list[str]:
    os.makedirs(CHUNK_DIR, exist_ok=True)

    existing = sorted(
        os.path.join(CHUNK_DIR, f)
        for f in os.listdir(CHUNK_DIR)
        if f.startswith("chunk_") and f.endswith(".parquet")
    )
    if len(existing) == 4:
        return existing

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type=HF_REPO_TYPE,
    )

    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    rows_per_chunk = (total_rows + 3) // 4

    chunk_paths = []
    for i in range(4):
        start = i * rows_per_chunk
        end = min(start + rows_per_chunk, total_rows)

        table = pq.read_table(
            path,
            columns=["id", "type", "author", "created_utc", "score", "body", "title", "link_id"],
        ).slice(start, end - start)

        chunk_path = os.path.join(CHUNK_DIR, f"chunk_{i}.parquet")
        pq.write_table(table, chunk_path, compression="zstd")
        chunk_paths.append(chunk_path)

    return chunk_paths


# ============================================================
# Search
# ============================================================
def search_in_chunks(query: str, chunk_paths: list[str], phrase_mode: bool) -> pd.DataFrame:
    q_lower = query.lower()
    matches = []
    total_found = 0

    for chunk_path in chunk_paths:
        df = pd.read_parquet(
            chunk_path,
            columns=["id", "type", "author", "created_utc", "score", "body", "title", "link_id"],
        )

        df["type"] = df["type"].astype("category")
        df["score"] = df["score"].fillna(0).astype("int32")
        df["created_utc"] = df["created_utc"].fillna(0).astype("int64")

        body_lower = df["body"].str.lower()
        title_lower = df["title"].str.lower()

        if phrase_mode:
            mask = body_lower.str.contains(q_lower, regex=False, na=False) | \
                   title_lower.str.contains(q_lower, regex=False, na=False)
        else:
            terms = [t for t in re.split(r"\s+", q_lower) if len(t) >= 2]
            mask = pd.Series([True] * len(df))
            for term in terms:
                mask &= (body_lower.str.contains(term, regex=False, na=False) |
                         title_lower.str.contains(term, regex=False, na=False))

        chunk_matches = df[mask]
        total_found += len(chunk_matches)

        if len(chunk_matches) > 0:
            matches.append(chunk_matches)

        if total_found >= MAX_MATCHES:
            break

        del df
        import gc
        gc.collect()

    if not matches:
        return pd.DataFrame()

    result = pd.concat(matches, ignore_index=True)
    if len(result) > MAX_MATCHES:
        result = result.head(MAX_MATCHES)

    return result


# ============================================================
# Header
# ============================================================
st.title("📚 Archive Search")
st.caption(f"A searchable snapshot of {TOTAL_RECORDS:,} historical records.")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filters")

    include_posts = st.checkbox("Include posts", value=True)
    include_comments = st.checkbox("Include comments", value=True)

    phrase_mode = st.checkbox("Exact phrase match", value=False)

    min_score = st.slider("Minimum score", 0, 500, 0, step=5)

    st.markdown("**Year range**")
    year_range = st.slider("Year", 2016, 2023, (2016, 2023), step=1)

    author_filter = st.text_input("Author (optional)", placeholder="e.g. silkandfeather")

    sort_mode = st.radio(
        "Sort by",
        ["Top scored", "Newest", "Oldest"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📈 Archive Stats")
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Total</div>
        <div class="stat-value">{TOTAL_RECORDS:,}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Posts</div>
        <div class="stat-value">{TOTAL_POSTS:,}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Comments</div>
        <div class="stat-value">{TOTAL_COMMENTS:,}</div>
    </div>
    """, unsafe_allow_html=True)

    if "recent_searches" not in st.session_state:
        st.session_state.recent_searches = []

    if st.session_state.recent_searches:
        st.markdown("---")
        st.markdown("### 🕘 Recent")
        for q_old in st.session_state.recent_searches[-5:]:
            st.markdown(f'<span class="chip">{escape_html(q_old)}</span>', unsafe_allow_html=True)

# ============================================================
# Prepare chunks
# ============================================================
with st.spinner("Preparing archive chunks… (first load only)"):
    try:
        chunk_paths = prepare_chunks()
    except Exception as exc:
        st.error("❌ Failed to prepare archive chunks.")
        st.code(str(exc))
        st.stop()

# ============================================================
# Search
# ============================================================
query = st.text_input(
    "🔍 Search the archive",
    placeholder="Try: zimmermann, birkin, taobao, chanel…",
    label_visibility="collapsed",
)

if not query or len(query.strip()) < MIN_QUERY_LENGTH:
    st.info("👆 Type at least **2 characters** in the search bar to begin.")
    st.stop()

if query not in st.session_state.recent_searches:
    st.session_state.recent_searches.append(query)
    st.session_state.recent_searches = st.session_state.recent_searches[-5:]

with st.spinner("Searching…"):
    results = search_in_chunks(query.strip(), chunk_paths, phrase_mode)

if len(results) == 0:
    st.warning(f"No results for **{query}**.")
    st.stop()

# Filters
if not include_posts:
    results = results[results["type"] != "post"]
if not include_comments:
    results = results[results["type"] != "comment"]
if min_score > 0:
    results = results[results["score"] >= min_score]

if year_range:
    year_lo, year_hi = year_range
    try:
        dt = pd.to_datetime(results["created_utc"], unit="s", errors="coerce")
        year = dt.dt.year
        results = results[(year >= year_lo) & (year <= year_hi)]
    except Exception:
        pass

if author_filter.strip():
    af = author_filter.strip().lower().lstrip("u/")
    results = results[results["author"].str.lower().str.contains(af, regex=False, na=False)]

# Sort
if sort_mode == "Top scored":
    results = results.sort_values("score", ascending=False)
elif sort_mode == "Newest":
    results = results.sort_values("created_utc", ascending=False)
else:
    results = results.sort_values("created_utc", ascending=True)

total_results = len(results)

if total_results == 0:
    st.warning("No results after applying filters. Try widening them.")
    st.stop()

# ============================================================
# Results header + CSV
# ============================================================
col_left, col_right = st.columns([3, 1])
with col_left:
    st.markdown(f"### {total_results:,} result{'s' if total_results != 1 else ''} for **{escape_html(query)}**")
    if total_results >= MAX_MATCHES:
        st.caption(f"⚠️ Showing first {MAX_MATCHES:,} matches. Refine your search for more precise results.")
with col_right:
    csv_bytes = results.head(1000).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download 1,000 as CSV",
        data=csv_bytes,
        file_name=f"archive_{re.sub(r'[^a-zA-Z0-9]+', '_', query)[:40]}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# Pagination
# ============================================================
total_pages = max(1, (total_results - 1) // RESULTS_PER_PAGE + 1)
page = st.number_input(f"Page (1 – {total_pages})", min_value=1, max_value=total_pages, value=1, step=1)

start = (page - 1) * RESULTS_PER_PAGE
end = start + RESULTS_PER_PAGE
page_results = results.iloc[start:end]

# ============================================================
# Render results
# ============================================================
for idx, row in page_results.iterrows():
    is_post = row.get("type") == "post"
    badge = (
        '<span class="badge-post">POST</span>'
        if is_post
        else '<span class="badge-comment">COMMENT</span>'
    )
    author = escape_html(row.get("author", "unknown"))
    score = int(row.get("score", 0) or 0)
    date_str = format_date(row.get("created_utc", 0))
    post_id = str(row.get("id", ""))
    reddit_link = f"https://reddit.com/comments/{post_id}" if post_id else ""

    title_raw = str(row.get("title", "")).strip()
    body_raw = str(row.get("body", "")).strip()

    # Parent post reference for comments
    parent_html = ""
    if not is_post and row.get("link_id"):
        parent_id = str(row["link_id"]).replace("t3_", "")
        parent_html = f'<div class="parent-ref">↳ in reply to post {parent_id}</div>'

    # Card header
    meta = f"""
    <div class="result-meta">
        {badge}
        <span>by <span class="author">u/{author}</span></span>
        <span>· {date_str}</span>
        <span>· <span class="upvotes">▲ {score}</span></span>
    </div>
    """

    title_html = ""
    if title_raw:
        title_html = f'<div class="result-title">{highlight(linkify(escape_html(title_raw)), query)}</div>'

    # Short preview body
    preview = truncate(body_raw, MAX_BODY_PREVIEW)
    body_html = f'<div class="result-body">{highlight(linkify(escape_html(preview)), query)}</div>' if preview else ""

    # Render the visible card
    st.markdown(f"""
    <div class="result-card">
        {meta}
        {parent_html}
        {title_html}
        {body_html}
    </div>
    """, unsafe_allow_html=True)

    # Action row
    action_cols = st.columns([1, 1, 1, 3])

    # Show full text in expander
    if len(body_raw) > MAX_BODY_PREVIEW:
        with action_cols[0]:
            with st.expander("Show full text"):
                st.markdown(
                    f'<div class="result-body">{highlight(linkify(escape_html(body_raw)), query)}</div>',
                    unsafe_allow_html=True,
                )

    # Reddit link
    if reddit_link:
        with action_cols[1]:
            st.markdown(f"[🔗 Reddit]({reddit_link})")

    # Copy button
    with action_cols[2]:
        if st.button("📋 Copy", key=f"copy_{idx}"):
            st.session_state[f"copied_{idx}"] = True
            st.toast("Copied to clipboard!", icon="✅")

    # Show copied confirmation
    if st.session_state.get(f"copied_{idx}"):
        st.code(body_raw, language=None)

st.markdown(
    """
    <div class="footer">
        Archive Search · Historical data preserved for research purposes
    </div>
    """,
    unsafe_allow_html=True,
)
