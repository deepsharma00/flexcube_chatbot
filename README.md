# Simple JSON RAG (FlexCube Chatbot)

A **Hierarchical, Multi-Document Retrieval-Augmented Generation (RAG) System** built entirely for processing large, complex product documentation sets (such as Oracle FlexCube, ELCM, and OBPM) exported as PDFs and converted into structured JSON trees.

The system smartly mitigates the expensive token context limit of local LLMs by introducing a **4-Layer Cascade Search** mechanism, calling the LLM strictly only when absolutely necessary.

## 🌟 Key Features

### 1. 🏢 Hierarchical Multi-Layer Indexing
Handles massive multi-module codebases efficiently by segmenting documents into a 4-layer taxonomy:
*   **Layer 1 (Product):** FCUBS, ELCM, OBPM
*   **Layer 2 (Module):** `common_core`, `universal_banking`
*   **Layer 3 (Document):** e.g., `CoreServices.pdf`, `EMS.pdf`
*   **Layer 4 (Section & Content):** Text segments and cross-references.

### 2. ⚡ Cascade Search (Zero-LLM Routing)
Instead of stuffing thousands of documents into a vector database, the search mechanism cascades progressively from **Product → Module → Document → Section** using **pure keyword matching**. 
*   **Result:** Lightning-fast semantic routing with zero LLM API costs during the filtering stages.

### 3. 🧠 Smart Context Budgeting & LLM Optimization
*   **Minimal LLM Calls:** The LLM is invoked exactly **twice** per chat:
    1.  **Query Expansion:** Identifies broad synonyms for the user's raw input.
    2.  **Generation:** Produces the final response based on the top 5 targeted sections.
*   **Streaming Responses:** Fully supports token-by-token streaming, avoiding 300s Ollama timeouts on slow local models.
*   **Budget-Aware Context:** Hard limit of ~18,000 characters ensures the Mistral/Qwen 8K context window is never exceeded, preventing trailing cut-offs.

---

## 🏗️ Architecture & Files

| Component | File | Description |
| :--- | :--- | :--- |
| **Hierarchical Processing** | `batch_indexer.py` | Scans nested folders (Product/Module/*.pdf) and generates 3-layer index JSONs and localized document tree JSONs. |
| | `multi_querier.py` | CLI execution for cascade searching. Traces Product → Module → Document layers. |
| | `multi_chat.py` | Interactive terminal Chat GUI tracing the full layered RAG cascade output. |
| **Single-Doc Processing** | `indexer.py` | Core indexing logic to parse single PDF sections into JSON. |
| | `querier.py` | LLM invocation framework (Ollama APIs) and single-doc search. |
| | `chat.py` | Flat interactive chat UI for single documents. |
| **Configuration** | `config.py` & `.env` | Environment definition to centralize `OLLAMA_MODEL` and `OLLAMA_URL` globally. |

---

## 🛠️ Setup & Configuration

### Prerequisites
*   Python 3.9+
*   Running **Ollama** server with your desired model (e.g. `qwen2.5:7b` or `mistral:7b`)
*   Install requirements: `pip install -r requirements.txt` (Installs `pymupdf` and `requests`).

### 1. Configuration
Modify the `.env` file located in the root directory to point to your LLM instances. You do not need third-party packages to inject these vars; `config.py` handles parsing automatically.

```env
# ── Indexing Configuration (Heavy Workload) ──
INDEXER_HOST=groq
INDEXER_MODEL=llama-3.1-8b-instant

# ── Querying Configuration (Interactive) ──
QUERIER_HOST=http://localhost:11434
QUERIER_MODEL=qwen2.5:7b

# ── API Keys ──
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Prepare Documents
Structure your documentation PDFs using the strict organizational hierarchy:
```text
FlexCube/
├── FCUBS/
│   ├── common_core/
│   │   └── CoreServices.pdf
│   └── universal_banking/
│       └── Retail.pdf
└── ELCM/
    └── general/
        └── LettersOfCredit.pdf
```

---

## 🚀 Usage

### Step 1: Batch Indexing
Process the folder structure and generate the multi-layer metadata index. This process can be interrupted and safely resumed (skips already processed documents).

You can rely on the `.env` defaults, or override them dynamically using CLI flags:

```bash
# Default (reads INDEXER parameters from .env)
python batch_indexer.py --input FlexCube/ --output index/

# Dynamic override (use Groq specifically for this run)
python batch_indexer.py --input FlexCube/ --output index/ --host groq --model llama-3.1-8b-instant
```

### Step 2: Interactive Cascading Chat
Start the multi-layered chat application. 

```bash
# Default (reads QUERIER parameters from .env)
python multi_chat.py --index index/

# Dynamic override (use Groq specifically for chatting)
python multi_chat.py --index index/ --host groq --model llama-3.3-70b-versatile
```

**Commands inside Multi-Chat:**
*   `products` → List all indexed products.
*   `modules` → List all modules for the given products.
*   `docs` → Show all indexed documents and page counts.
*   `tree` → Display full node/tree hierarchy.
*   `quit` → Exit the session.

### Alternative Step 2: Single CLI Query
If you prefer hitting the engine programmatically for a single response:

```bash
# Default query
python multi_querier.py --index index/ --query "how to reset rates based on holiday calendar"

# Dynamic Groq query
python multi_querier.py --index index/ --query "how to create a branch" --host groq --model llama-3.3-70b-versatile
```

## 🔄 Decoupled Model Pipeline
The system fully supports **decoupled AI pipelines**. You can use a massively fast, hosted API (like Groq) for the heavy lifting of indexing 100,000+ pages, while keeping all interactive querying completely private on your local `qwen2.5:7b` server. 

To switch models, just open `.env`, update `QUERIER_MODEL=llama3:8b`, and run your query again. **No re-indexing required!**


# Single PDF
python groq_pool_indexer.py --pdf CoreServices.pdf --output index/CoreServices.json

# Batch folder (same structure as batch_indexer.py)
python groq_pool_indexer.py --input FlexCube/ --output index/

# Pass keys inline (no .env needed)
python groq_pool_indexer.py --input FlexCube/ --keys gsk_aaa,gsk_bbb,gsk_ccc

# Re-index everything (ignore existing JSONs)
python groq_pool_indexer.py --input FlexCube/ --reindex

# run ui 
streamlit run ui.py