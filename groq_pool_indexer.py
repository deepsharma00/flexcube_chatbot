"""
Groq Multi-Key Pool Indexer
============================
Indexes PDFs using a pool of Groq API keys.

Key features:
  - Maintains a pool of up to N Groq API keys
  - Tracks RPM (requests per minute) per key using a 60s sliding window
  - On 429 rate limit → immediately rotates to next available key
  - On all keys exhausted → sleeps the minimum time until one key is free
  - Skips already-indexed PDFs (resume support)
  - Supports both single PDF and batch folder modes

Usage:
  # Single PDF
  python groq_pool_indexer.py --pdf path/to/doc.pdf --output doc.json

  # Batch folder (Product/Module/Doc.pdf structure)
  python groq_pool_indexer.py --input FlexCube/ --output index/

Keys in .env:
  GROQ_API_KEY_1=gsk_...
  GROQ_API_KEY_2=gsk_...
  GROQ_API_KEY_3=gsk_...
  GROQ_API_KEY_4=gsk_...
  GROQ_API_KEY_5=gsk_...
  GROQ_MODEL=llama-3.1-8b-instant

Or pass directly via CLI:
  python groq_pool_indexer.py --keys gsk_aaa,gsk_bbb,gsk_ccc --input FlexCube/
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import requests
import fitz  # PyMuPDF

from config import INDEXER_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Groq Key Pool
# ─────────────────────────────────────────────────────────────────────────────

class GroqKeyPool:
    """
    Manages a pool of Groq API keys with:
      - Per-key RPM tracking (60-second sliding window)
      - Smart rotation: pick the key with most remaining capacity
      - 429 handling: cool a key for 65s, rotate immediately
      - All-keys-exhausted: sleep minimum time until a slot opens
    """

    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_keys: list[str], rpm_limit: int = 28):
        """
        Args:
            api_keys:  List of Groq API key strings.
            rpm_limit: Max requests per minute per key (Groq free = 30, we use 28 for safety).
        """
        if not api_keys:
            raise ValueError("GroqKeyPool requires at least one API key.")

        self.keys      = api_keys
        self.rpm_limit = rpm_limit

        # Per-key tracking state
        # requests: list of timestamps of successful requests in last 60s
        # cool_until: epoch time when key is available again after 429
        self._stats = {
            key: {"requests": [], "cool_until": 0.0}
            for key in api_keys
        }

        self._current_idx = 0   # start with first key
        self._total_calls  = 0  # lifetime counter
        self._total_429s   = 0  # lifetime 429 counter

        print(f"  [POOL] Initialized with {len(api_keys)} key(s), "
              f"{rpm_limit} RPM limit per key "
              f"({rpm_limit * len(api_keys)} RPM effective max)\n")

    # ── Internals ──────────────────────────────────────────────────────────

    def _trim_window(self, key: str):
        """Remove timestamps older than 60s from this key's request log."""
        cutoff = time.time() - 60.0
        self._stats[key]["requests"] = [
            t for t in self._stats[key]["requests"] if t > cutoff
        ]

    def _rpm_used(self, key: str) -> int:
        """Return how many requests this key has made in the last 60s."""
        self._trim_window(key)
        return len(self._stats[key]["requests"])

    def _is_cooling(self, key: str) -> bool:
        """True if this key is still in the 429 cooldown period."""
        return time.time() < self._stats[key]["cool_until"]

    def _rpm_remaining(self, key: str) -> int:
        """Slots left before this key hits the RPM cap."""
        if self._is_cooling(key):
            return 0
        return max(0, self.rpm_limit - self._rpm_used(key))

    def _seconds_until_slot(self, key: str) -> float:
        """
        How many seconds until this key has at least one free RPM slot.
        Returns 0 if a slot is available now.
        """
        if self._is_cooling(key):
            return self._stats[key]["cool_until"] - time.time()

        self._trim_window(key)
        reqs = self._stats[key]["requests"]
        if len(reqs) < self.rpm_limit:
            return 0.0  # slot available now

        # Oldest request in window will expire after 60s
        oldest = min(reqs)
        wait   = 60.0 - (time.time() - oldest) + 0.5  # +0.5s buffer
        return max(0.0, wait)

    # ── Public API ─────────────────────────────────────────────────────────

    def pick_key(self) -> str:
        """
        Return the best available API key.
        - Prefers key with most remaining RPM capacity
        - If all keys are at limit, sleeps the minimum wait then returns
        Never returns None — always blocks until a key is free.
        """
        while True:
            best_key  = None
            best_remaining = -1

            for key in self.keys:
                rem = self._rpm_remaining(key)
                if rem > best_remaining:
                    best_remaining = rem
                    best_key = key

            if best_remaining > 0:
                self._current_idx = self.keys.index(best_key)
                return best_key

            # All keys are either cooling or at RPM limit.
            # Find minimum wait across all keys.
            min_wait = min(self._seconds_until_slot(k) for k in self.keys)
            min_wait = max(min_wait, 1.0)  # at least 1s

            # Show which keys are cooling vs at RPM cap
            status_parts = []
            for k in self.keys:
                label = f"...{k[-6:]}"
                if self._is_cooling(k):
                    remaining = self._stats[k]["cool_until"] - time.time()
                    status_parts.append(f"{label}=COOL({remaining:.0f}s)")
                else:
                    status_parts.append(f"{label}={self._rpm_used(k)}/{self.rpm_limit}rpm")

            print(f"\n  [POOL] All keys busy — "
                  f"{', '.join(status_parts)}. Sleeping {min_wait:.1f}s...\n")
            time.sleep(min_wait)

    def record_request(self, key: str):
        """Record a successful request for this key."""
        self._stats[key]["requests"].append(time.time())
        self._total_calls += 1

    def handle_429(self, key: str):
        """
        Called when this key returns HTTP 429.
        Cools the key for 65s and rotates current index away from it.
        """
        self._stats[key]["cool_until"] = time.time() + 65.0
        self._total_429s += 1
        print(f"  [429]  Key ...{key[-6:]} rate-limited → "
              f"cooling 65s, rotating to next key.")

    def status_line(self) -> str:
        """One-line pool status for progress output."""
        parts = []
        for k in self.keys:
            rem = self._rpm_remaining(k)
            if self._is_cooling(k):
                parts.append(f"...{k[-6:]}=COOL")
            else:
                parts.append(f"...{k[-6:]}={rem}left")
        return f"[{' | '.join(parts)}] calls={self._total_calls} 429s={self._total_429s}"

    def print_status(self):
        """Full status table."""
        now = time.time()
        print(f"\n  ┌─ Pool Status ({len(self.keys)} keys, {self.rpm_limit} RPM/key) ─────────")
        for i, key in enumerate(self.keys):
            self._trim_window(key)
            used = len(self._stats[key]["requests"])
            if self._is_cooling(key):
                cooling_for = self._stats[key]["cool_until"] - now
                state = f"🔴 COOLING ({cooling_for:.0f}s left)"
            else:
                state = f"🟢 OK  {used}/{self.rpm_limit} RPM used"
            print(f"  │  Key {i+1}  ...{key[-12:]:<12}  {state}")
        print(f"  │  Total calls: {self._total_calls}  |  Total 429s: {self._total_429s}")
        print(f"  └─────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Groq Call (with pool rotation)
# ─────────────────────────────────────────────────────────────────────────────

def call_groq(pool: GroqKeyPool, prompt: str, model: str,
              timeout: int = 120, max_attempts: int = 8) -> str:
    """
    Call Groq using the pool, rotating keys automatically on 429.

    Args:
        pool:         GroqKeyPool instance.
        prompt:       Text prompt.
        model:        Groq model name.
        timeout:      HTTP request timeout in seconds.
        max_attempts: Hard limit on total attempts before giving up.

    Returns:
        Response text, or "" on repeated failure.
    """
    url = GroqKeyPool.GROQ_URL

    for attempt in range(1, max_attempts + 1):
        key = pool.pick_key()

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":    model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)

            if resp.status_code == 429:
                pool.handle_429(key)
                # Don't count as a successful request — loop and pick fresh key
                continue

            resp.raise_for_status()

            # Record usage only on success
            pool.record_request(key)
            return resp.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            print(f"  [TIMEOUT] Key ...{key[-6:]}, attempt {attempt}/{max_attempts}")
            pool.record_request(key)  # still count — Groq processed it
            if attempt == max_attempts:
                return ""
            time.sleep(2)
            continue

        except Exception as e:
            print(f"  [ERROR] attempt {attempt}/{max_attempts}: {e}")
            if attempt == max_attempts:
                return ""
            time.sleep(3)
            continue

    print(f"  [FAIL] Exceeded {max_attempts} attempts. Skipping this request.")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# JSON Utility
# ─────────────────────────────────────────────────────────────────────────────

def extract_json_safe(text: str):
    """Extract JSON from model response, handling markdown fences."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for pattern in [r'\[.*?\]', r'\{.*?\}']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PDF Extraction  (identical to indexer.py — kept standalone for no circular import)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_path: str) -> list:
    """Extract raw text from every page."""
    doc   = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages


def extract_builtin_toc(pdf_path: str) -> list:
    """Use PDF metadata TOC if available."""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    doc.close()
    if not toc:
        return []

    result   = []
    counters = {}
    for level, title, page in toc:
        counters[level] = counters.get(level, 0) + 1
        for deeper in list(counters.keys()):
            if deeper > level:
                del counters[deeper]
        structure = ".".join(str(counters[l]) for l in sorted(counters.keys()))
        result.append({"structure": structure, "title": title.strip(), "page_number": page})

    print(f"  Found {len(result)} entries in built-in PDF TOC ✅")
    return result


def extract_toc_from_text(pages: list) -> list:
    """Fallback: parse TOC lines from first ≤12 pages."""
    toc_items   = []
    seen_titles = set()

    for page in pages[:12]:
        text  = page["text"]
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(
                r'^(\d+(?:\.\d+)*)\s{1,4}([A-Z][^\n]{3,70}?)[\s\.]{2,}(\d+[-\d]*)\s*$',
                line
            )
            if match:
                structure = match.group(1)
                title     = re.sub(r'\s+', ' ', match.group(2)).strip()
                if title not in seen_titles and len(title) > 3:
                    seen_titles.add(title)
                    toc_items.append({
                        "structure": structure,
                        "title":     title,
                        "toc_ref":   match.group(3)
                    })

    if toc_items:
        print(f"  Extracted {len(toc_items)} TOC entries from text ✅")
    return toc_items


def map_toc_to_pages(toc_items: list, pages: list) -> list:
    """Map TOC entries to physical page numbers by title-text search."""
    result = []
    for item in toc_items:
        title       = item["title"]
        title_lower = title.lower().strip()
        best_page   = item.get("page_number", 1)

        for page in pages:
            clean_title = re.sub(r'\s+', ' ', title_lower)
            if clean_title in page["text"].lower():
                best_page = page["page_number"]
                break

        result.append({
            "structure":        item["structure"],
            "title":            title,
            "start_page":       best_page,
            "end_page":         None,
            "summary":          "",
            "cross_references": [],
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Reference Detection
# ─────────────────────────────────────────────────────────────────────────────

CROSS_REF_PATTERNS = [
    r'[Ss]ection\s+(\d+(?:\.\d+)+)',
    r'[Cc]hapter\s+(\d+)',
    r'[Aa]nnex(?:ure)?\s+([A-Z\d]+)',
    r'(?:refer to|see|as per|detailed in|described in|mentioned in)\s+(\d+(?:\.\d+)*)',
    r'(?:in|under|at)\s+[Ss]ection\s+(\d+(?:\.\d+)+)',
]


def find_cross_references_in_text(text: str, tree: list) -> list:
    """Find cross-reference patterns in text and map them to tree nodes."""
    found = set()
    for pattern in CROSS_REF_PATTERNS:
        for m in re.findall(pattern, text):
            ref = m.strip()
            if ref:
                found.add(ref)

    referenced = []
    for structure in found:
        for node in tree:
            ns = node.get("structure", "")
            if ns == structure or ns.startswith(structure + "."):
                info = {
                    "structure":  ns,
                    "title":      node.get("title", ""),
                    "start_page": node.get("start_page", 1),
                }
                if info not in referenced:
                    referenced.append(info)
    return referenced


def add_cross_references_to_tree(tree: list, page_summaries: list) -> list:
    """Populate cross_references on every tree node from its raw page text."""
    page_map   = {p["page_number"]: p for p in page_summaries}
    total_refs = 0

    print("\n  Detecting cross-references...")
    for i, node in enumerate(tree):
        start  = node.get("start_page", 1)
        end    = tree[i + 1].get("start_page", start + 1) if i + 1 < len(tree) else start + 2
        node["end_page"] = end

        section_text = "".join(
            page_map[pn].get("raw_text", "")
            for pn in range(start, end)
            if pn in page_map
        )
        refs = find_cross_references_in_text(section_text, tree)
        refs = [r for r in refs if r["structure"] != node["structure"]]
        node["cross_references"] = refs

        if refs:
            total_refs += len(refs)
            ref_names = [f"{r['structure']} ({r['title'][:30]})" for r in refs]
            print(f"    [{node['structure']}] {node['title'][:40]}")
            print(f"      → refs: {', '.join(ref_names)}")

    print(f"  Total cross-references found: {total_refs}")
    return tree


# ─────────────────────────────────────────────────────────────────────────────
# LLM Summarisation (pool-aware)
# ─────────────────────────────────────────────────────────────────────────────

def summarize_page(page: dict, pool: GroqKeyPool, model: str) -> dict:
    """Summarize one page with LLM, tracking which key is used."""
    page_num = page["page_number"]
    text     = page["text"]

    if not text or len(text) < 20:
        return {
            "page_number": page_num,
            "sections":    [],
            "summary":     "Empty or unreadable page",
            "raw_text":    text,
        }

    text_chunk = text[:1500]

    prompt = f"""Read this text from page {page_num} of a document and do two things:
1. List all section headings or titles you find (numbered or named sections only)
2. Write a 1-2 sentence summary of what this page is about

Text:
{text_chunk}

Respond ONLY in this JSON format, nothing else:
{{
  "sections": ["Section title 1", "Section title 2"],
  "summary": "One or two sentence summary."
}}

If no headings found, return empty sections array.
Return ONLY the JSON object, no other text."""

    print(f"    Page {page_num:3d}…", end=" ", flush=True)
    t0       = time.time()
    response = call_groq(pool, prompt, model)
    elapsed  = time.time() - t0
    key_used = pool.keys[pool._current_idx]
    print(f"({elapsed:.1f}s) [key …{key_used[-6:]}]  {pool.status_line()}")

    result = extract_json_safe(response)
    if result and isinstance(result, dict):
        return {
            "page_number": page_num,
            "sections":    result.get("sections", []),
            "summary":     result.get("summary", ""),
            "raw_text":    text,
        }

    return {
        "page_number": page_num,
        "sections":    [],
        "summary":     response[:200] if response else "Could not summarize",
        "raw_text":    text,
    }


def build_tree_from_summaries(page_summaries: list, pool: GroqKeyPool, model: str) -> list:
    """Last-resort: ask LLM to build a hierarchical TOC from page summaries."""
    overview_lines = []
    for p in page_summaries:
        sections = ", ".join(p["sections"][:3]) if p["sections"] else "no headings"
        overview_lines.append(f"Page {p['page_number']}: {sections} | {p['summary'][:100]}")
    overview = "\n".join(overview_lines)

    prompt = f"""Build a hierarchical table of contents from these page summaries.

{overview}

Return ONLY a JSON array:
[
  {{"structure": "1", "title": "...", "start_page": 1, "summary": "..."}},
  {{"structure": "1.1", "title": "...", "start_page": 2, "summary": "..."}}
]

Return ONLY the JSON array, no other text."""

    print("  Building tree from summaries (LLM call)…", end=" ", flush=True)
    t0       = time.time()
    response = call_groq(pool, prompt, model, timeout=300)
    print(f"({time.time() - t0:.1f}s)")

    tree = extract_json_safe(response)
    if tree and isinstance(tree, list):
        for node in tree:
            node.setdefault("end_page", node.get("start_page", 1) + 1)
            node.setdefault("cross_references", [])
        return tree

    # Final flat fallback
    print("  [WARN] Using flat structure fallback")
    flat    = []
    counter = 1
    for p in page_summaries:
        for section in p.get("sections", []):
            flat.append({
                "structure":        str(counter),
                "title":            section,
                "start_page":       p["page_number"],
                "end_page":         p["page_number"] + 1,
                "summary":          p.get("summary", ""),
                "cross_references": [],
            })
            counter += 1
    return flat


def add_summaries_to_tree(tree: list, page_summaries: list) -> list:
    """Merge LLM page summaries into the tree nodes."""
    page_map = {p["page_number"]: p for p in page_summaries}
    for node in tree:
        pn = node.get("start_page", 1)
        if pn in page_map and not node.get("summary"):
            node["summary"] = page_map[pn].get("summary", "")
    return tree


# ─────────────────────────────────────────────────────────────────────────────
# Main Indexer
# ─────────────────────────────────────────────────────────────────────────────

def index_pdf_pooled(pdf_path: str, pool: GroqKeyPool, model: str,
                     output_path: str = None) -> dict:
    """
    Index a single PDF using the multi-key Groq pool.
    Output structure is identical to indexer.py — fully compatible.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"  Error: PDF not found: {pdf_path}")
        return {}

    if output_path is None:
        output_path = Path(__file__).parent / (pdf_path.stem + ".json")
    output_path = Path(output_path)

    print(f"\n  {'─'*55}")
    print(f"  📄 Indexing : {pdf_path.name}")
    print(f"  🤖 Model   : {model}")
    print(f"  {'─'*55}")

    # Step 1: Extract text
    print("\n  Step 1: Extracting PDF text…")
    pages = extract_pdf_pages(str(pdf_path))
    print(f"  Found {len(pages)} pages")

    # Step 2: TOC
    print("\n  Step 2: Extracting Table of Contents…")
    toc_items = extract_builtin_toc(str(pdf_path))
    if not toc_items:
        print("  No built-in TOC, trying text extraction…")
        toc_items = extract_toc_from_text(pages)

    use_llm_for_tree = len(toc_items) == 0
    if use_llm_for_tree:
        print("  No TOC found — will use LLM to build tree")

    # Step 3: Summarise pages
    print("\n  Step 3: Summarising pages with Groq pool…")
    page_summaries = []
    for page in pages:
        summary = summarize_page(page, pool, model)
        page_summaries.append(summary)

    # Step 4: Build tree
    print("\n  Step 4: Building document tree…")
    if toc_items:
        tree       = map_toc_to_pages(toc_items, pages)
        tree       = add_summaries_to_tree(tree, page_summaries)
        toc_method = "builtin" if any("page_number" in t for t in toc_items) else "text"
        print(f"  ✅ Tree built from TOC ({len(tree)} nodes)")
    else:
        tree       = build_tree_from_summaries(page_summaries, pool, model)
        toc_method = "llm"
        print(f"  ✅ Tree built from LLM ({len(tree)} nodes)")

    # Step 5: Cross-references
    tree = add_cross_references_to_tree(tree, page_summaries)

    # Step 6: Save
    index_data = {
        "source":      str(pdf_path),
        "model_used":  model,
        "total_pages": len(pages),
        "toc_method":  toc_method,
        "tree":        tree,
        "pages":       page_summaries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    total_refs = sum(len(n.get("cross_references", [])) for n in tree)
    print(f"\n  ✅ Saved → {output_path}")
    print(f"     Nodes: {len(tree)} | Pages: {len(pages)} | "
          f"TOC: {toc_method} | Cross-refs: {total_refs}")

    return index_data


# ─────────────────────────────────────────────────────────────────────────────
# Batch Indexer (Product / Module / Doc hierarchy)
# ─────────────────────────────────────────────────────────────────────────────

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'and', 'or', 'but', 'of', 'in', 'on', 'at', 'for', 'with',
    'to', 'by', 'this', 'that', 'it', 'from', 'as', 'not', 'no',
    'has', 'have', 'had', 'will', 'can', 'may', 'shall', 'should',
}


def scan_hierarchy(root_path: str) -> list:
    """
    Scan for Product/Module/*.pdf structure.
    Returns a list of products each containing modules each containing docs.
    PDFs directly under a product get module='general'.
    """
    root = Path(root_path)
    if not root.exists():
        print(f"Error: Folder not found: {root}")
        sys.exit(1)

    products = []
    for product_dir in sorted(root.iterdir()):
        if not product_dir.is_dir():
            continue
        product = {"name": product_dir.name, "modules": []}

        for module_dir in sorted(product_dir.iterdir()):
            if not module_dir.is_dir():
                continue
            pdfs = sorted(module_dir.glob("*.pdf"))
            if pdfs:
                product["modules"].append({
                    "name": module_dir.name,
                    "documents": [
                        {"stem": p.stem, "filename": p.name, "pdf_path": str(p)}
                        for p in pdfs
                    ],
                })

        direct_pdfs = sorted(product_dir.glob("*.pdf"))
        if direct_pdfs:
            product["modules"].append({
                "name": "general",
                "documents": [
                    {"stem": p.stem, "filename": p.name, "pdf_path": str(p)}
                    for p in direct_pdfs
                ],
            })

        if product["modules"]:
            products.append(product)

    return products


def extract_keywords_from_tree(tree: list) -> list:
    """Extract meaningful keywords from all TOC titles (no LLM)."""
    all_words = set()
    for node in tree:
        words = re.sub(r'[^\w\s]', ' ', node.get("title", "").lower()).split()
        all_words.update(w for w in words if len(w) > 2 and w not in STOP_WORDS)
    return sorted(all_words)


def get_top_sections(tree: list, max_depth: int = 1) -> list:
    """Return top-level section titles for the layer index."""
    return [
        node.get("title", "")
        for node in tree
        if node.get("structure", "").count(".") <= max_depth
    ][:15]


def get_document_summary(index_data: dict) -> str:
    """Build a short doc summary from first 5 page summaries (no extra LLM)."""
    summaries = [
        p.get("summary", "").strip()
        for p in index_data.get("pages", [])[:5]
        if p.get("summary", "").strip() not in ("", "Empty or unreadable page", "Could not summarize")
    ]
    return " ".join(summaries)[:300]


def build_doc_index(product_name: str, module_name: str, output_folder: str) -> dict:
    """Build doc_index.json for a module folder."""
    module_dir = Path(output_folder) / product_name / module_name
    doc_index  = {
        "product":        product_name,
        "module":         module_name,
        "document_count": 0,
        "documents":      [],
    }

    for json_file in sorted(module_dir.glob("*.json")):
        if json_file.name == "doc_index.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        tree = data.get("tree", [])
        doc_index["documents"].append({
            "name":           json_file.stem,
            "filename":       json_file.stem + ".pdf",
            "index_file":     json_file.name,
            "total_pages":    data.get("total_pages", 0),
            "total_sections": len(tree),
            "toc_method":     data.get("toc_method", "unknown"),
            "top_sections":   get_top_sections(tree),
            "summary":        get_document_summary(data),
            "keywords":       extract_keywords_from_tree(tree),
        })

    doc_index["document_count"] = len(doc_index["documents"])
    index_path = module_dir / "doc_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, indent=2, ensure_ascii=False)

    print(f"        doc_index.json → {doc_index['document_count']} documents")
    return doc_index


def build_module_index(product_name: str, output_folder: str) -> dict:
    """Build module_index.json for a product folder."""
    product_dir  = Path(output_folder) / product_name
    module_index = {
        "product":      product_name,
        "module_count": 0,
        "modules":      [],
    }

    for module_dir in sorted(product_dir.iterdir()):
        if not module_dir.is_dir():
            continue
        doc_index_path = module_dir / "doc_index.json"
        if not doc_index_path.exists():
            continue

        with open(doc_index_path, "r", encoding="utf-8") as f:
            doc_index = json.load(f)

        all_keywords  = set()
        all_summaries = []
        all_sections  = []
        doc_names     = []

        for doc in doc_index.get("documents", []):
            all_keywords.update(doc.get("keywords", []))
            s = doc.get("summary", "")
            if s:
                all_summaries.append(s[:150])
            all_sections.extend(doc.get("top_sections", [])[:5])
            doc_names.append(doc["name"])

        module_index["modules"].append({
            "name":           module_dir.name,
            "document_count": doc_index.get("document_count", 0),
            "documents":      doc_names,
            "top_sections":   all_sections[:20],
            "summary":        " ".join(all_summaries)[:400],
            "keywords":       sorted(all_keywords),
        })

    module_index["module_count"] = len(module_index["modules"])
    index_path = product_dir / "module_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(module_index, f, indent=2, ensure_ascii=False)

    print(f"      module_index.json → {module_index['module_count']} modules")
    return module_index


def build_product_index(output_folder: str) -> dict:
    """Build product_index.json at the root of the index folder."""
    output_folder = Path(output_folder)
    product_index = {
        "created_at":     datetime.now().isoformat(),
        "index_folder":   str(output_folder.resolve()),
        "total_products": 0,
        "products":       [],
    }

    for product_dir in sorted(output_folder.iterdir()):
        if not product_dir.is_dir():
            continue
        module_index_path = product_dir / "module_index.json"
        if not module_index_path.exists():
            continue

        with open(module_index_path, "r", encoding="utf-8") as f:
            module_index = json.load(f)

        all_keywords  = set()
        module_names  = []
        all_summaries = []
        total_docs    = 0

        for module in module_index.get("modules", []):
            all_keywords.update(module.get("keywords", []))
            module_names.append(module["name"])
            s = module.get("summary", "")
            if s:
                all_summaries.append(s[:150])
            total_docs += module.get("document_count", 0)

        product_index["products"].append({
            "name":            product_dir.name,
            "module_count":    len(module_names),
            "modules":         module_names,
            "total_documents": total_docs,
            "summary":         " ".join(all_summaries)[:500],
            "keywords":        sorted(all_keywords),
        })

    product_index["total_products"] = len(product_index["products"])
    index_path = output_folder / "product_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(product_index, f, indent=2, ensure_ascii=False)

    print(f"    product_index.json → {product_index['total_products']} products")
    return product_index


def batch_index_pooled(root_path: str, output_folder: str,
                       pool: GroqKeyPool, model: str,
                       skip_existing: bool = True) -> dict:
    """
    Batch-index all PDFs under root_path using the Groq key pool.

    root_path/
      Product/
        Module/
          Doc.pdf

    Builds the full layered index (doc_index, module_index, product_index).
    """
    root_path     = Path(root_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Groq Pool Batch Indexer")
    print(f"  Root   : {root_path}")
    print(f"  Output : {output_folder}")
    print(f"  Model  : {model}")
    print(f"  Skip existing: {skip_existing}")
    print(f"{'═'*60}\n")

    pool.print_status()

    # Scan
    print("Step 1: Scanning folder hierarchy…")
    products = scan_hierarchy(str(root_path))

    if not products:
        print("  No products found! Expected: Root/Product/Module/*.pdf")
        sys.exit(1)

    total_docs = 0
    for product in products:
        doc_count = sum(len(m["documents"]) for m in product["modules"])
        total_docs += doc_count
        print(f"  {product['name']}: "
              f"{len(product['modules'])} modules, {doc_count} docs")

    # Index each PDF
    print(f"\nStep 2: Indexing {total_docs} PDFs…\n")
    indexed  = 0
    skipped  = 0
    failed   = 0
    doc_num  = 0
    start_ts = time.time()

    for product in products:
        for module in product["modules"]:
            module_out = output_folder / product["name"] / module["name"]
            module_out.mkdir(parents=True, exist_ok=True)

            for doc in module["documents"]:
                doc_num     += 1
                pdf_path     = doc["pdf_path"]
                stem         = doc["stem"]
                json_out     = module_out / f"{stem}.json"
                route_label  = f"{product['name']}/{module['name']}/{stem}"

                if skip_existing and json_out.exists():
                    print(f"  [{doc_num:3d}/{total_docs}] SKIP  {route_label}")
                    skipped += 1
                    continue

                print(f"\n  [{doc_num:3d}/{total_docs}] INDEXING  {route_label}")

                try:
                    index_pdf_pooled(pdf_path, pool, model, str(json_out))
                    indexed += 1
                except SystemExit:
                    print(f"  [FAILED] {pdf_path} not found")
                    failed += 1
                except Exception as e:
                    print(f"  [FAILED] {e}")
                    failed += 1

                # Print pool status every 10 docs
                if indexed % 10 == 0 and indexed > 0:
                    pool.print_status()

    elapsed = (time.time() - start_ts) / 60
    print(f"\n  Indexing done in {elapsed:.1f} min — "
          f"new={indexed}  skipped={skipped}  failed={failed}\n")
    pool.print_status()

    # Build layered indexes (no LLM needed)
    print("Step 3: Building layered indexes…\n")

    print("  Layer 3 — doc_index.json (per module):")
    for product in products:
        for module in product["modules"]:
            print(f"    {product['name']}/{module['name']}:")
            build_doc_index(product["name"], module["name"], str(output_folder))

    print("\n  Layer 2 — module_index.json (per product):")
    for product in products:
        print(f"    {product['name']}:")
        build_module_index(product["name"], str(output_folder))

    print("\n  Layer 1 — product_index.json (root):")
    product_index = build_product_index(str(output_folder))

    print(f"\n{'═'*60}")
    print(f"  ✅ Batch indexing complete!")
    print(f"  Products : {product_index['total_products']}")
    for p in product_index["products"]:
        print(f"    {p['name']}: {p['module_count']} modules, "
              f"{p['total_documents']} docs")
    print(f"  Index at : {output_folder / 'product_index.json'}")
    print(f"{'═'*60}\n")
    pool.print_status()

    return product_index


# ─────────────────────────────────────────────────────────────────────────────
# Key Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_keys_from_env() -> list[str]:
    """
    Load API keys from .env file.

    Looks for:
      GROQ_API_KEY_1=gsk_...
      GROQ_API_KEY_2=gsk_...
      ...
      GROQ_API_KEY_N=gsk_...

    Falls back to GROQ_API_KEY if no numbered keys found.
    """
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return []

    keys = {}
    single_key = None

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()

            # Numbered keys: GROQ_API_KEY_1 … GROQ_API_KEY_N
            m = re.match(r'^GROQ_API_KEY_(\d+)$', k)
            if m and v.startswith("gsk_"):
                keys[int(m.group(1))] = v

            # Single key fallback
            if k == "GROQ_API_KEY" and v.startswith("gsk_"):
                single_key = v

    if keys:
        return [keys[i] for i in sorted(keys.keys())]
    if single_key:
        return [single_key]
    return []


def load_model_from_env() -> str:
    """Read GROQ_MODEL from .env, default to llama-3.1-8b-instant."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return INDEXER_MODEL

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GROQ_MODEL="):
                return line.split("=", 1)[1].strip()
            if line.startswith("INDEXER_MODEL="):
                return line.split("=", 1)[1].strip()
    return INDEXER_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Groq Multi-Key Pool Indexer\n"
            "Indexes PDFs with automatic API key rotation and RPM management.\n\n"
            "Modes:\n"
            "  Single PDF:   --pdf doc.pdf [--output doc.json]\n"
            "  Batch folder: --input FlexCube/ [--output index/]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Mode ──
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--pdf",   metavar="FILE",   help="Single PDF to index")
    mode_group.add_argument("--input", metavar="FOLDER", help="Root folder (Product/Module/Doc.pdf)")

    # ── Output ──
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output JSON file (single) or index folder (batch). "
             "Defaults to <stem>.json or ./index/",
    )

    # ── Keys ──
    parser.add_argument(
        "--keys", metavar="KEY1,KEY2,...",
        help=(
            "Comma-separated Groq API keys. "
            "If omitted, reads GROQ_API_KEY_1…N from .env"
        ),
    )
    parser.add_argument(
        "--rpm", type=int, default=28, metavar="N",
        help="RPM limit per key (default: 28, Groq free = 30)",
    )

    # ── Model ──
    parser.add_argument(
        "--model", default=None, metavar="MODEL",
        help="Groq model (default: reads GROQ_MODEL or INDEXER_MODEL from .env)",
    )

    # ── Batch options ──
    parser.add_argument(
        "--reindex", action="store_true",
        help="Re-index PDFs even if JSON already exists (batch mode only)",
    )

    args = parser.parse_args()

    # ── Resolve API keys ──────────────────────────────────────────────────
    if args.keys:
        api_keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    else:
        api_keys = load_keys_from_env()

    if not api_keys:
        print("\n[ERROR] No Groq API keys found.")
        print("  Option 1 — Add to .env:")
        print("    GROQ_API_KEY_1=gsk_...")
        print("    GROQ_API_KEY_2=gsk_...")
        print("  Option 2 — Pass via CLI:")
        print("    --keys gsk_aaa,gsk_bbb,gsk_ccc")
        sys.exit(1)

    # ── Resolve model ─────────────────────────────────────────────────────
    model = args.model or load_model_from_env()

    # ── Create pool ───────────────────────────────────────────────────────
    pool = GroqKeyPool(api_keys=api_keys, rpm_limit=args.rpm)

    print(f"\n{'═'*60}")
    print(f"  Groq Pool Indexer")
    print(f"  Model  : {model}")
    print(f"  Keys   : {len(api_keys)}")
    print(f"  RPM    : {args.rpm}/key → {args.rpm * len(api_keys)} effective max")
    print(f"{'═'*60}")

    # ── Run ───────────────────────────────────────────────────────────────
    if args.pdf:
        # ── Single PDF mode ──
        output = args.output or (Path(args.pdf).stem + ".json")
        index_pdf_pooled(args.pdf, pool, model, output)

    else:
        # ── Batch folder mode ──
        output = args.output or "index"
        batch_index_pooled(
            root_path=args.input,
            output_folder=output,
            pool=pool,
            model=model,
            skip_existing=not args.reindex,
        )

    pool.print_status()


if __name__ == "__main__":
    main()
