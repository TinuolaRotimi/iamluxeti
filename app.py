<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Archive Search</title>
<meta name="description" content="Search 1.9M+ historical records" />

<script src="https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.28.0/dist/duckdb-browser.mjs" type="module"></script>
<script src="https://cdn.tailwindcss.com"></script>

<script>
  tailwind.config = {
    theme: {
      extend: {
        colors: {
          bg: '#0d1117',
          panel: '#161b22',
          border: '#30363d',
          text: '#e6edf3',
          muted: '#8b949e',
          accent: '#ff4500',
          link: '#58a6ff',
        },
      },
    },
  };
</script>

<style>
  body { background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
  mark { background: #ff4500; color: white; padding: 1px 3px; border-radius: 3px; }
  .result-card { transition: border-color 0.15s ease, transform 0.15s ease; }
  .result-card:hover { border-color: #ff4500; transform: translateY(-1px); }
  .result-body { word-wrap: break-word; overflow-wrap: anywhere; }
  .result-body a, .result-title a { color: #58a6ff; text-decoration: none; border-bottom: 1px dotted #58a6ff; }
  .result-body a:hover { color: #79c0ff; border-bottom-color: #79c0ff; }
  .skeleton { background: linear-gradient(90deg, #161b22 25%, #21262d 50%, #161b22 75%); background-size: 200% 100%; animation: shimmer 1.4s infinite; }
  @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 5px; }
  ::-webkit-scrollbar-thumb:hover { background: #ff4500; }
</style>
</head>

<body class="min-h-screen">

<!-- Header -->
<header class="border-b border-border bg-panel">
  <div class="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-lg bg-accent flex items-center justify-center text-white font-bold text-lg">A</div>
      <div>
        <h1 class="text-xl font-bold text-accent">Archive Search</h1>
        <p class="text-xs text-muted" id="header-stats">Loading…</p>
      </div>
    </div>
    <div class="hidden md:flex items-center gap-2 text-xs text-muted">
      <span class="px-2 py-1 rounded bg-bg border border-border">DuckDB-WASM</span>
      <span class="px-2 py-1 rounded bg-bg border border-border">1.9M records</span>
    </div>
  </div>
</header>

<!-- Main -->
<main class="max-w-6xl mx-auto px-6 py-8">

  <!-- Search -->
  <div class="mb-6">
    <div class="relative">
      <input
        id="search-input"
        type="text"
        placeholder="Search the archive…  (try: zimmermann, birkin, taobao)"
        class="w-full bg-panel border border-border rounded-lg px-12 py-4 text-lg text-text placeholder-muted focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20 transition"
        autocomplete="off"
      />
      <svg class="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
      </svg>
      <button id="clear-btn" class="hidden absolute right-4 top-1/2 -translate-y-1/2 text-muted hover:text-accent transition">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
        </svg>
      </button>
    </div>
  </div>

  <!-- Controls -->
  <div class="flex flex-wrap items-center gap-3 mb-6">
    <label class="flex items-center gap-2 text-sm text-muted cursor-pointer">
      <input type="checkbox" id="filter-posts" checked class="accent-accent" />
      Posts
    </label>
    <label class="flex items-center gap-2 text-sm text-muted cursor-pointer">
      <input type="checkbox" id="filter-comments" checked class="accent-accent" />
      Comments
    </label>
    <select id="sort-select" class="bg-panel border border-border text-text text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-accent">
      <option value="relevance">Sort: Relevance</option>
      <option value="newest" selected>Sort: Newest</option>
      <option value="top">Sort: Top scored</option>
    </select>
    <button id="download-btn" class="hidden bg-panel border border-border text-text text-sm rounded-lg px-3 py-2 hover:border-accent transition">
      ⬇ Download CSV
    </button>
  </div>

  <!-- Status -->
  <div id="status" class="text-sm text-muted mb-4">Initializing search engine…</div>

  <!-- Results -->
  <div id="results"></div>

  <!-- Pagination -->
  <div id="pagination" class="hidden justify-center items-center gap-2 mt-8"></div>

</main>

<footer class="border-t border-border mt-12 py-6 text-center text-xs text-muted">
  Archive Search · Historical data preserved for research purposes
</footer>

<script type="module">
import * as duckdb from 'https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.28.0/dist/duckdb-browser.mjs';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const PARQUET_URL = "https://huggingface.co/datasets/just3nu/RepLadies_Archive/resolve/main/repladies_clean.parquet";
const RESULTS_PER_PAGE = 25;
const DEBOUNCE_MS = 300;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let db = null;
let conn = null;
let currentQuery = "";
let currentPage = 1;
let currentResults = [];
let totalResults = 0;
let isSearching = false;

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const searchInput = $("search-input");
const clearBtn = $("clear-btn");
const statusEl = $("status");
const resultsEl = $("results");
const paginationEl = $("pagination");
const filterPosts = $("filter-posts");
const filterComments = $("filter-comments");
const sortSelect = $("sort-select");
const downloadBtn = $("download-btn");
const headerStats = $("header-stats");

// ---------------------------------------------------------------------------
// Init DuckDB
// ---------------------------------------------------------------------------
async function initDuckDB() {
  try {
    statusEl.textContent = "Loading DuckDB engine…";

    const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
    const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);

    const workerUrl = URL.createObjectURL(
      new Blob([`importScripts("${bundle.mainWorker}");`], { type: "text/javascript" })
    );
    const worker = new Worker(workerUrl);

    const logger = new duckdb.ConsoleLogger();
    db = new duckdb.AsyncDuckDB(logger, worker);
    await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
    URL.revokeObjectURL(workerUrl);

    statusEl.textContent = "Connecting to dataset…";
    conn = await db.connect();

    await conn.query(`
      CREATE OR REPLACE VIEW archive AS
      SELECT * FROM read_parquet('${PARQUET_URL}')
    `);

    // Get stats
    const stats = await conn.query(`
      SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN type = 'post' THEN 1 ELSE 0 END) AS posts,
        SUM(CASE WHEN type = 'comment' THEN 1 ELSE 0 END) AS comments
      FROM archive
    `);
    const row = stats.toArray()[0];
    const total = Number(row.total).toLocaleString();
    headerStats.textContent = `${total} records · searchable`;
    statusEl.innerHTML = `<span class="text-accent">✓</span> Ready — type a keyword to search ${total} records.`;

    // Hide the initial loading state
    resultsEl.innerHTML = emptyState();
  } catch (err) {
    console.error(err);
    statusEl.innerHTML = `<span class="text-red-400">✗ Error:</span> ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
let debounceTimer = null;
searchInput.addEventListener("input", (e) => {
  const val = e.target.value.trim();
  clearBtn.classList.toggle("hidden", val.length === 0);

  clearTimeout(debounceTimer);
  if (val.length < 2) {
    currentResults = [];
    totalResults = 0;
    resultsEl.innerHTML = emptyState();
    paginationEl.classList.add("hidden");
    statusEl.textContent = "Type at least 2 characters.";
    downloadBtn.classList.add("hidden");
    return;
  }
  debounceTimer = setTimeout(() => performSearch(val), DEBOUNCE_MS);
});

clearBtn.addEventListener("click", () => {
  searchInput.value = "";
  clearBtn.classList.add("hidden");
  resultsEl.innerHTML = emptyState();
  paginationEl.classList.add("hidden");
  downloadBtn.classList.add("hidden");
  statusEl.textContent = "Type a keyword to search.";
});

[filterPosts, filterComments, sortSelect].forEach((el) =>
  el.addEventListener("change", () => {
    if (currentQuery) performSearch(currentQuery);
  })
);

async function performSearch(query) {
  if (isSearching) return;
  isSearching = true;
  currentQuery = query;
  currentPage = 1;

  statusEl.innerHTML = `<span class="text-accent">⏳</span> Searching…`;
  resultsEl.innerHTML = skeletons(5);
  paginationEl.classList.add("hidden");

  try {
    const escaped = query.replace(/'/g, "''").toLowerCase();
    const filters = [];

    if (!filterPosts.checked) filters.push("type != 'post'");
    if (!filterComments.checked) filters.push("type != 'comment'");

    const filterClause = filters.length ? `AND ${filters.join(" AND ")}` : "";

    let orderBy = "created_utc DESC";
    if (sortSelect.value === "top") orderBy = "score DESC";
    if (sortSelect.value === "relevance") orderBy = "score DESC";

    // Count
    const countRes = await conn.query(`
      SELECT COUNT(*) AS c FROM archive
      WHERE (LOWER(body) LIKE '%${escaped}%' OR LOWER(title) LIKE '%${escaped}%')
      ${filterClause}
    `);
    totalResults = Number(countRes.toArray()[0].c);

    if (totalResults === 0) {
      statusEl.innerHTML = `<span class="text-yellow-400">⚠</span> No results for "<b>${escapeHtml(query)}</b>".`;
      resultsEl.innerHTML = "";
      downloadBtn.classList.add("hidden");
      isSearching = false;
      return;
    }

    // Fetch first page
    const results = await conn.query(`
      SELECT id, type, author, created_utc, score, body, title
      FROM archive
      WHERE (LOWER(body) LIKE '%${escaped}%' OR LOWER(title) LIKE '%${escaped}%')
      ${filterClause}
      ORDER BY ${orderBy}
      LIMIT ${RESULTS_PER_PAGE}
    `);
    currentResults = results.toArray();

    statusEl.innerHTML = `<span class="text-accent">✓</span> <b>${totalResults.toLocaleString()}</b> result${totalResults === 1 ? "" : "s"} for "<b>${escapeHtml(query)}</b>"`;
    renderResults(currentResults, query);
    renderPagination();
    downloadBtn.classList.remove("hidden");
  } catch (err) {
    console.error(err);
    statusEl.innerHTML = `<span class="text-red-400">✗ Search error:</span> ${err.message}`;
    resultsEl.innerHTML = "";
  } finally {
    isSearching = false;
  }
}

async function goToPage(page) {
  if (page < 1 || page === currentPage) return;
  currentPage = page;
  const offset = (page - 1) * RESULTS_PER_PAGE;
  const escaped = currentQuery.replace(/'/g, "''").toLowerCase();
  const filters = [];
  if (!filterPosts.checked) filters.push("type != 'post'");
  if (!filterComments.checked) filters.push("type != 'comment'");
  const filterClause = filters.length ? `AND ${filters.join(" AND ")}` : "";
  let orderBy = "created_utc DESC";
  if (sortSelect.value === "top") orderBy = "score DESC";

  resultsEl.innerHTML = skeletons(5);
  window.scrollTo({ top: 0, behavior: "smooth" });

  const results = await conn.query(`
    SELECT id, type, author, created_utc, score, body, title
    FROM archive
    WHERE (LOWER(body) LIKE '%${escaped}%' OR LOWER(title) LIKE '%${escaped}%')
    ${filterClause}
    ORDER BY ${orderBy}
    LIMIT ${RESULTS_PER_PAGE} OFFSET ${offset}
  `);
  currentResults = results.toArray();
  renderResults(currentResults, currentQuery);
  renderPagination();
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------
function emptyState() {
  return `
    <div class="text-center py-16">
      <div class="text-5xl mb-4">📚</div>
      <p class="text-muted">Type a keyword to search 1.9M+ historical records.</p>
      <div class="mt-6 flex flex-wrap justify-center gap-2 text-sm">
        ${["zimmermann", "birkin", "taobao", "chanel", "yupoo", "weidian"].map((t) =>
          `<button onclick="document.getElementById('search-input').value='${t}';document.getElementById('search-input').dispatchEvent(new Event('input'))" class="bg-panel border border-border rounded-full px-3 py-1 hover:border-accent transition">${t}</button>`
        ).join("")}
      </div>
    </div>
  `;
}

function skeletons(n) {
  return Array.from({ length: n }).map(() => `
    <div class="result-card bg-panel border border-border rounded-lg p-4 mb-3">
      <div class="skeleton h-3 w-40 rounded mb-3"></div>
      <div class="skeleton h-5 w-3/4 rounded mb-3"></div>
      <div class="skeleton h-3 w-full rounded mb-2"></div>
      <div class="skeleton h-3 w-5/6 rounded"></div>
    </div>
  `).join("");
}

function renderResults(rows, query) {
  if (!rows.length) { resultsEl.innerHTML = ""; return; }
  resultsEl.innerHTML = rows.map((row) => renderCard(row, query)).join("");
}

function renderCard(row, query) {
  const isPost = row.type === "post";
  const badge = isPost
    ? `<span class="bg-accent text-white text-xs font-bold px-2 py-0.5 rounded">POST</span>`
    : `<span class="bg-border text-text text-xs font-bold px-2 py-0.5 rounded">COMMENT</span>`;

  const date = formatDate(row.created_utc);
  const score = Number(row.score || 0);
  const author = escapeHtml(row.author || "unknown");

  const titleHtml = row.title
    ? `<div class="text-lg font-bold text-text mt-2 mb-2">${highlight(linkify(escapeHtml(row.title)), query)}</div>`
    : "";

  const bodyRaw = truncate(row.body || "", 700);
  const bodyHtml = bodyRaw
    ? `<div class="result-body text-[15px] leading-relaxed text-text">${highlight(linkify(escapeHtml(bodyRaw)), query)}</div>`
    : "";

  return `
    <article class="result-card bg-panel border border-border rounded-lg p-4 mb-3">
      <div class="flex flex-wrap items-center gap-2 text-xs text-muted mb-1">
        ${badge}
        <span>by <span class="text-link font-semibold">u/${author}</span></span>
        <span>· ${date}</span>
        <span>· <span class="text-accent font-semibold">▲ ${score}</span></span>
      </div>
      ${titleHtml}
      ${bodyHtml}
    </article>
  `;
}

function renderPagination() {
  const totalPages = Math.max(1, Math.ceil(totalResults / RESULTS_PER_PAGE));
  if (totalPages <= 1) { paginationEl.classList.add("hidden"); return; }

  const prev = currentPage > 1;
  const next = currentPage < totalPages;

  paginationEl.classList.remove("hidden");
  paginationEl.classList.add("flex");
  paginationEl.innerHTML = `
    <button ${prev ? "" : "disabled"} onclick="window.__goto(${currentPage - 1})" class="px-4 py-2 rounded-lg border border-border bg-panel text-text disabled:opacity-40 disabled:cursor-not-allowed hover:border-accent transition">← Prev</button>
    <span class="px-4 py-2 text-sm text-muted">Page <b class="text-text">${currentPage}</b> of <b class="text-text">${totalPages}</b></span>
    <button ${next ? "" : "disabled"} onclick="window.__goto(${currentPage + 1})" class="px-4 py-2 rounded-lg border border-border bg-panel text-text disabled:opacity-40 disabled:cursor-not-allowed hover:border-accent transition">Next →</button>
  `;
}

window.__goto = goToPage;

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------
downloadBtn.addEventListener("click", async () => {
  if (!currentQuery) return;
  const escaped = currentQuery.replace(/'/g, "''").toLowerCase();
  const filters = [];
  if (!filterPosts.checked) filters.push("type != 'post'");
  if (!filterComments.checked) filters.push("type != 'comment'");
  const filterClause = filters.length ? `AND ${filters.join(" AND ")}` : "";

  const res = await conn.query(`
    SELECT id, type, author, created_utc, score, title, body
    FROM archive
    WHERE (LOWER(body) LIKE '%${escaped}%' OR LOWER(title) LIKE '%${escaped}%')
    ${filterClause}
    LIMIT 1000
  `);
  const rows = res.toArray();
  const header = ["id", "type", "author", "created_utc", "score", "title", "body"];
  const csv = [
    header.join(","),
    ...rows.map((r) =>
      header.map((h) => `"${String(r[h] ?? "").replace(/"/g, '""')}"`).join(",")
    ),
  ].join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `archive_${currentQuery.slice(0, 30).replace(/\s+/g, "_")}.csv`;
  a.click();
  URL.revokeObjectURL(url);
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

const URL_REGEX = /(https?:\/\/[^\s<>"')\]]+)/gi;
function linkify(escaped) {
  return escaped.replace(URL_REGEX, (url) => {
    let trailing = "";
    while (url && /[.,;:!?)]$/.test(url)) {
      trailing = url.slice(-1) + trailing;
      url = url.slice(0, -1);
    }
    return `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>${trailing}`;
  });
}

function highlight(html, query) {
  const terms = query.split(/\s+/).filter((t) => t.length >= 2);
  let result = html;
  terms.forEach((term) => {
    const re = new RegExp(`(?<!<[^>]*)(${escapeRegex(term)})`, "gi");
    result = result.replace(re, "<mark>$1</mark>");
  });
  return result;
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function formatDate(ts) {
  try {
    const n = Number(ts);
    if (!n || n <= 0) return "unknown date";
    const d = new Date(n * 1000);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  } catch { return "unknown date"; }
}

function truncate(text, max) {
  if (!text) return "";
  return text.length <= max ? text : text.slice(0, max).replace(/\s+\S*$/, "") + "…";
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
initDuckDB();
</script>
</body>
</html>
