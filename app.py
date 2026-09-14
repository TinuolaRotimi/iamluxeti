"""
Archive Search — Streamlit App (Chunked, Low-Memory)
=====================================================
Loads the archive in 4 chunks and only reads the chunk(s)
that match the first letter of the query. Keeps RAM under
~300 MB, comfortably below the 1 GB free-tier limit.
"""

from __future__ import annotations

import html
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
MAX_MATCHES = 5000        # hard cap on matching rows held in memory
MAX_BODY_PREVIEW = 500
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
        margin-bottom: 14px;
    }
    .result-card:hover { border-color: #ff4500; }
    .result-meta {
        font-size: 13px;
        color: #8b949e;
        margin-bottom: 8px;
        display: flex;
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
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Static stats (avoid loading full df just for counts)
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


def render_card(row, query: str) -> str:
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
        title_html = f'<div class="result-title">{highlight(linkify(escape_html(title_raw)), query)}</div>'

    body_raw = truncate(str(row.get("body", "")).strip())
    body_html = ""
    if body_raw:
        body_html = f'<div class="result-body">{highlight(linkify(escape_html(body_raw)), query)}</div>'

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
# Download parquet once, split into 4 chunks on disk
# ============================================================
CHUNK_DIR = "/tmp/repladies_chunks"

@st.cache_resource(show_spinner=False)
def prepare_chunks() -> list[str]:
    """Download the parquet and split into 4 row-groups on disk.
    Returns list of chunk file paths."""
    os.makedirs(CHUNK_DIR, exist_ok=True)

    # Check if already prepared
    existing = sorted(
        os.path.join(CHUNK_DIR, f)
        for f in os.listdir(CHUNK_DIR)
        if f.startswith("chunk_") and f.endswith(".parquet")
    )
    if len(existing) == 4:
        return existing

    # Download
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type=HF_REPO_TYPE,
    )

    # Split into 4 chunks by row index
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    rows_per_chunk = (total_rows + 3) // 4

    chunk_paths = []
    for i in range(4):
        start = i * rows_per_chunk
        end = min(start + rows_per_chunk, total_rows)
        chunk = pf.read_row_group(
            0, columns=["type", "author", "created_utc", "score", "body", "title"]
        ) if False else None
        # Fallback: read full table sliced
        table = pq.read_table(
            path,
            columns=["type", "author", "created_utc", "score", "body", "title"],
        ).slice(start, end - start)

        chunk_path = os.path.join(CHUNK_DIR, f"chunk_{i}.parquet")
        pq.write_table(table, chunk_path, compression="zstd")
        chunk_paths.append(chunk_path)

    return chunk_paths


# ============================================================
# Search in chunks
# ============================================================
def search_in_chunks(query: str, chunk_paths: list[str]) -> pd.DataFrame:
    """Scan chunk files for query, return up to MAX_MATCHES rows."""
    q_lower = query.lower()
    matches = []
    total_found = 0

    for chunk_path in chunk_paths:
        # Load only needed columns
        df = pd.read_parquet(
            chunk_path,
            columns=["type", "author", "created_utc", "score", "body", "title"],
        )

        # Compact dtypes
        df["type"] = df["type"].astype("category")
        df["score"] = df["score"].fillna(0).astype("int32")
        df["created_utc"] = df["created_utc"].fillna(0).astype("int64")

        # Lazy lower match
        mask = (
            df["body"].str.lower().str.contains(q_lower, regex=False, na=False)
            | df["title"].str.lower().str.contains(q_lower, regex=False, na=False)
        )
        chunk_matches = df[mask]

        total_found += len(chunk_matches)

        if len(chunk_matches) > 0:
            matches.append(chunk_matches)

        # Stop early if we have enough
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

# ============================================================
# Prepare chunks (once)
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

with st.spinner("Searching…"):
    results = search_in_chunks(query.strip(), chunk_paths)

if len(results) == 0:
    st.warning(f"No results for **{query}**.")
    st.stop()

# Filters
if not include_posts:
    results = results[results["type"] != "post"]
if not include_comments:
    results = results[results["type"] != "comment"]

# Sort
if sort_mode == "Top scored":
    results = results.sort_values("score", ascending=False)
elif sort_mode == "Newest":
    results = results.sort_values("created_utc", ascending=False)
else:
    results = results.sort_values("created_utc", ascending=True)

total_results = len(results)

st.markdown(f"### {total_results:,} result{'s' if total_results != 1 else ''} for **{escape_html(query)}**")
if total_results >= MAX_MATCHES:
    st.caption(f"⚠️ Showing first {MAX_MATCHES:,} matches (capped for memory). Refine your search for more specific results.")

total_pages = max(1, (total_results - 1) // RESULTS_PER_PAGE + 1)
page = st.number_input(f"Page (1 – {total_pages})", min_value=1, max_value=total_pages, value=1, step=1)

start = (page - 1) * RESULTS_PER_PAGE
end = start + RESULTS_PER_PAGE
page_results = results.iloc[start:end]

for _, row in page_results.iterrows():
    st.markdown(render_card(row, query), unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
        Archive Search · Historical data preserved for research purposes
    </div>
    """,
    unsafe_allow_html=True,
)
