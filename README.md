# Archive Search

A fast, serverless search engine for exploring a large historical text archive.

## Features

- 🔍 Full-text search across 1.9M+ records
- ⚡ DuckDB-WASM — runs entirely in your browser
- 🎨 Dark theme with Reddit-style result cards
- 🔗 Clickable links in results
- 📊 Sort by relevance, newest, or top scored
- ⬇ Export results to CSV
- 🚀 Zero backend — pure static HTML

## Tech Stack

- **DuckDB-WASM** — SQL engine in the browser
- **Tailwind CSS** — styling
- **HTTP Range Requests** — streams only the needed data
- **Hugging Face** — hosts the parquet dataset

## Data Source

Records are served from a public Hugging Face dataset and queried client-side.
