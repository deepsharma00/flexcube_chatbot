# Updates

## 2026-03-25 — Centralize config into .env file
- **Problem**: Model name (`qwen2.5:7b`) and Ollama URL were hardcoded across 6 files. Changing the model required editing every file.
- **Fix**: Created `.env` file and `config.py`. All 6 modules now import `OLLAMA_MODEL` and `OLLAMA_URL` from `config.py`, which reads `.env` on import. Zero external dependencies (no python-dotenv needed).
- **To change model/URL**: Edit `.env` only. CLI `--model` / `--host` flags still override.
- **New files**: `.env`, `config.py`
- **Files changed**: `indexer.py`, `querier.py`, `chat.py`, `batch_indexer.py`, `multi_querier.py`, `multi_chat.py`

## 2026-03-25 — Hierarchical Multi-Layer Indexing & Cascade Search
- **Problem**: Flat master_index.json treated all docs equally. With a product like FlexCube having FCUBS/ELCM/OBPM, each with many modules and sub-docs, the query had no way to navigate the hierarchy. Context was wasted on irrelevant products/modules.
- **Solution**: 3-layer hierarchical indexing with cascade search:
  - **Layer 1 — `product_index.json`**: Lists all products (FCUBS, ELCM, OBPM) with aggregated keywords/summaries from all their modules.
  - **Layer 2 — `module_index.json` (per product)**: Lists modules within a product (common_core, universal_banking) with aggregated keywords from their docs.
  - **Layer 3 — `doc_index.json` (per module)**: Lists documents with top sections, summaries, and keywords extracted from the document tree.
  - **Layer 4 — `document.json`**: Existing document tree with sections and page text.
- **Query cascade**: Product → Module → Document → Section. Each layer uses pure keyword matching (no LLM). The LLM is called only twice: once for query expansion, once for the final answer.
- **Context budget**: 18,000 chars max. Sections processed in score order. Pages added one at a time until budget is reached. Score filter drops sections below 25% of top score.
- **Files rewritten**:
  - `batch_indexer.py` — Detects Product/Module/Document hierarchy from folder structure. Builds all 3 layer indexes from document JSONs (no extra LLM calls). Resume support (skips existing JSONs).
  - `multi_querier.py` — `QueryLayeredStream()` cascades through layers. Generic `ScoreEntry()` and `SelectBestEntries()` work at any layer. Budget-aware `BuildContext()`.
  - `multi_chat.py` — Shows routing path at each layer. Commands: `tree`, `products`, `modules`, `docs`, `quit`.
- **Usage**:
  - Folder: `FlexCube/FCUBS/common_core/*.pdf` (Root/Product/Module/Docs)
  - Index: `python batch_indexer.py --input FlexCube/ --output index/`
  - Chat: `python multi_chat.py --index index/`
  - CLI: `python multi_querier.py --index index/ --query "your question"`
- **No changes to existing single-doc files** — `indexer.py`, `querier.py`, `chat.py` untouched

## 2026-03-24 — Fix Ollama Timeout on Answer Generation
- **Problem**: `answer_question` used `stream: False` with a 300s timeout. For large contexts on a remote Ollama server, the model couldn't finish generating before the timeout expired → `[Timeout — model took too long]`.
- **Fix**: Added `call_ollama_stream()` in `querier.py` — uses `stream: True` so tokens arrive incrementally and the connection never times out waiting for the full response. Added `answer_question_stream()` generator for the interactive chat path. Updated `chat.py` to stream tokens to the console in real-time.
- **Files changed**: `querier.py`, `chat.py`

## 2026-03-24 — Increase Streaming Output Token Limit
- **Problem**: `num_predict: 1024` capped the answer at ~1024 tokens, causing responses to get cut off mid-sentence on detailed questions.
- **Fix**: Increased `call_ollama_stream` default `num_predict` from 1024 → **4096**. Non-streaming `call_ollama` stays at 1024 (only used for short query expansion calls).
- **Files changed**: `querier.py`
