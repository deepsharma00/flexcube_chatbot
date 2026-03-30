"""
PDF Indexer v3 - Builds a JSON tree from a PDF using Ollama
Improvements over v2:
- Cross-reference detection during indexing (NEW)
  Finds "See Section X", "Refer to Chapter Y" patterns
  Stores links in tree nodes for use at query time
- Built-in PDF TOC extraction (no LLM for structure)
- Page-by-page LLM summarization (no timeouts)
- Section boundary aware context
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

import requests
import fitz  # PyMuPDF

from config import INDEXER_HOST, INDEXER_MODEL, GROQ_API_KEY


# ─────────────────────────────────────────
# LLM API
# ─────────────────────────────────────────

def call_ollama(prompt: str, model: str, host: str = INDEXER_HOST, timeout: int = 300) -> str:
    """Call Ollama API or Groq API and return response text."""
    if host.lower() == "groq":
        if not GROQ_API_KEY:
            print("  [ERROR] GROQ_API_KEY not found in .env files.")
            return ""
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        try:
            # Groq free tier limit is often 30 RPM (2s per request). Sleep to avoid 429.
            time.sleep(2.5)
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                print("  [RATE LIMIT] Sleeping for 10 seconds due to strict RPM...")
                time.sleep(10)
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            print(f"  [TIMEOUT] Groq took too long.")
            return ""
        except Exception as e:
            print(f"  [GROQ ERROR] {e}")
            return ""

    # Default to standard local Ollama
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 2048,
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        print(f"  [TIMEOUT] Model took too long.")
        return ""
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""


def warmup_model(model: str, host: str):
    """Pre-load model into RAM."""
    if host.lower() == "groq":
        print(f"  Groq {model} selected (warmup ignored).")
        return
        
    print(f"  Warming up {model}...", end=" ", flush=True)
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": "hi",
        "stream": False,
        "options": {"num_predict": 1}
    }
    try:
        requests.post(url, json=payload, timeout=300)
        print("✓")
    except Exception as e:
        print(f"Warning: {e}")


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


# ─────────────────────────────────────────
# PDF Extraction
# ─────────────────────────────────────────

def extract_pdf_pages(pdf_path: str) -> list:
    """Extract text from each page."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages


def extract_builtin_toc(pdf_path: str) -> list:
    """
    Get built-in TOC from PDF metadata.
    Returns list of {structure, title, page_number}.
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    doc.close()

    if not toc:
        return []

    result = []
    counters = {}

    for level, title, page in toc:
        counters[level] = counters.get(level, 0) + 1
        for deeper in list(counters.keys()):
            if deeper > level:
                del counters[deeper]
        structure = ".".join(str(counters[l]) for l in sorted(counters.keys()))
        result.append({
            "structure": structure,
            "title": title.strip(),
            "page_number": page
        })

    print(f"  Found {len(result)} entries in built-in PDF TOC ✅")
    return result


def extract_toc_from_text(pages: list) -> list:
    """Fallback: parse TOC from text of first few pages."""
    toc_items = []
    seen_titles = set()

    for page in pages[:8]:
        text = page["text"]
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
                title = re.sub(r'\s+', ' ', match.group(2)).strip()
                if title not in seen_titles and len(title) > 3:
                    seen_titles.add(title)
                    toc_items.append({
                        "structure": structure,
                        "title": title,
                        "toc_ref": match.group(3)
                    })

    if toc_items:
        print(f"  Extracted {len(toc_items)} TOC entries from text ✅")
    return toc_items


def map_toc_to_pages(toc_items: list, pages: list) -> list:
    """Map TOC entries to physical page numbers."""
    result = []
    for item in toc_items:
        title = item["title"]
        title_lower = title.lower().strip()
        best_page = item.get("page_number", 1)

        for page in pages:
            page_text_lower = page["text"].lower()
            clean_title = re.sub(r'\s+', ' ', title_lower)
            if clean_title in page_text_lower:
                best_page = page["page_number"]
                break

        result.append({
            "structure": item["structure"],
            "title": item["title"],
            "start_page": best_page,
            "end_page": None,       # filled later
            "summary": "",          # filled later
            "cross_references": []  # filled during cross-ref detection
        })

    return result


# ─────────────────────────────────────────
# Cross Reference Detection (NEW in v3)
# ─────────────────────────────────────────

# Patterns that indicate a cross reference in text
CROSS_REF_PATTERNS = [
    # "Section 2.4.2" or "section 2.4"
    r'[Ss]ection\s+(\d+(?:\.\d+)+)',
    # "Chapter 3" or "chapter 3"
    r'[Cc]hapter\s+(\d+)',
    # "Annexure A" or "Annex 1"
    r'[Aa]nnex(?:ure)?\s+([A-Z\d]+)',
    # "refer to 2.3.1" or "see 4.1"
    r'(?:refer to|see|as per|detailed in|described in|mentioned in)\s+(\d+(?:\.\d+)*)',
    # "in section 2.3" or "under section 4"
    r'(?:in|under|at)\s+[Ss]ection\s+(\d+(?:\.\d+)+)',
]


def find_cross_references_in_text(text: str, tree: list) -> list:
    """
    Scan text for cross-reference patterns.
    Returns list of tree nodes that are referenced.

    Example:
      Text: "...see Section 2.4.2 for reversal check..."
      Returns: [tree node for structure "2.4.2"]
    """
    found_structures = set()

    for pattern in CROSS_REF_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            ref = match.strip()
            if ref:
                found_structures.add(ref)

    # Match found structure numbers to tree nodes
    referenced_nodes = []
    for structure in found_structures:
        for node in tree:
            node_structure = node.get("structure", "")
            # Exact match or starts with (catches "2" matching "2.1", "2.2" etc)
            if node_structure == structure or node_structure.startswith(structure + "."):
                ref_info = {
                    "structure": node_structure,
                    "title": node.get("title", ""),
                    "start_page": node.get("start_page", 1)
                }
                if ref_info not in referenced_nodes:
                    referenced_nodes.append(ref_info)

    return referenced_nodes


def add_cross_references_to_tree(tree: list, page_summaries: list) -> list:
    """
    For each tree node, scan its pages for cross-references.
    Stores found references directly in the tree node.

    This runs DURING INDEXING so querier can use them instantly.
    """
    page_map = {p["page_number"]: p for p in page_summaries}

    print("\nStep 4: Detecting cross-references...")
    total_refs = 0

    for i, node in enumerate(tree):
        start_page = node.get("start_page", 1)

        # Determine end page from next node
        if i + 1 < len(tree):
            end_page = tree[i + 1].get("start_page", start_page + 1)
        else:
            end_page = start_page + 2

        node["end_page"] = end_page

        # Collect all text for this section
        section_text = ""
        for pn in range(start_page, end_page):
            if pn in page_map:
                section_text += page_map[pn].get("raw_text", "")

        # Find cross references in this section's text
        refs = find_cross_references_in_text(section_text, tree)

        # Remove self-references
        refs = [r for r in refs if r["structure"] != node["structure"]]

        node["cross_references"] = refs

        if refs:
            ref_names = [f"{r['structure']} ({r['title'][:30]})" for r in refs]
            print(f"  [{node['structure']}] {node['title'][:40]}")
            print(f"    → references: {', '.join(ref_names)}")
            total_refs += len(refs)

    print(f"  Total cross-references found: {total_refs}")
    return tree


# ─────────────────────────────────────────
# LLM Summarization
# ─────────────────────────────────────────

def summarize_page(page: dict, model: str, host: str) -> dict:
    """Summarize a single page and extract section headings."""
    page_num = page["page_number"]
    text = page["text"]

    if not text or len(text) < 20:
        return {
            "page_number": page_num,
            "sections": [],
            "summary": "Empty or unreadable page",
            "raw_text": text
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

    print(f"  Page {page_num}...", end=" ", flush=True)
    start = time.time()
    response = call_ollama(prompt, model, host)
    elapsed = time.time() - start
    print(f"({elapsed:.1f}s)")

    result = extract_json_safe(response)
    if result and isinstance(result, dict):
        return {
            "page_number": page_num,
            "sections": result.get("sections", []),
            "summary": result.get("summary", ""),
            "raw_text": text
        }

    return {
        "page_number": page_num,
        "sections": [],
        "summary": response[:200] if response else "Could not summarize",
        "raw_text": text
    }


def add_summaries_to_tree(tree: list, page_summaries: list) -> list:
    """Add LLM summaries to tree nodes."""
    page_map = {p["page_number"]: p for p in page_summaries}
    for node in tree:
        page_num = node.get("start_page", 1)
        if page_num in page_map and not node.get("summary"):
            node["summary"] = page_map[page_num].get("summary", "")
    return tree


def build_tree_from_summaries(page_summaries: list, model: str, host: str) -> list:
    """Last resort: build tree from LLM when no TOC found."""
    overview_lines = []
    for p in page_summaries:
        sections = ", ".join(p["sections"][:3]) if p["sections"] else "no headings"
        overview_lines.append(
            f"Page {p['page_number']}: {sections} | {p['summary'][:100]}"
        )
    overview = "\n".join(overview_lines)

    prompt = f"""Build a hierarchical table of contents from these page summaries.

{overview}

Return ONLY a JSON array:
[
  {{"structure": "1", "title": "...", "start_page": 1, "summary": "..."}},
  {{"structure": "1.1", "title": "...", "start_page": 2, "summary": "..."}}
]

Return ONLY the JSON array, no other text."""

    print("\n  Building tree from summaries...", end=" ", flush=True)
    start = time.time()
    response = call_ollama(prompt, model, host, timeout=300)
    print(f"({time.time()-start:.1f}s)")

    tree = extract_json_safe(response)
    if tree and isinstance(tree, list):
        # Add default fields
        for node in tree:
            node.setdefault("end_page", node.get("start_page", 1) + 1)
            node.setdefault("cross_references", [])
        return tree

    # Final fallback
    print("  [WARN] Using flat structure fallback")
    flat = []
    counter = 1
    for p in page_summaries:
        for section in p.get("sections", []):
            flat.append({
                "structure": str(counter),
                "title": section,
                "start_page": p["page_number"],
                "end_page": p["page_number"] + 1,
                "summary": p.get("summary", ""),
                "cross_references": []
            })
            counter += 1
    return flat


# ─────────────────────────────────────────
# Main Indexer
# ─────────────────────────────────────────

def index_pdf(pdf_path: str, model: str, host: str, output_path: str = None):
    """Main indexing function."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    if output_path is None:
        output_path = Path(__file__).parent / (pdf_path.stem + ".json")
    output_path = Path(output_path)

    print(f"\n{'='*55}")
    print(f"📄 Indexing: {pdf_path.name}")
    print(f"🤖 Model:    {model}")
    print(f"🌐 Ollama:   {host}")
    print(f"{'='*55}\n")

    # Warmup
    warmup_model(model, host)

    # ── Step 1: Extract PDF text ──────────────────
    print("\nStep 1: Extracting PDF text...")
    pages = extract_pdf_pages(str(pdf_path))
    print(f"  Found {len(pages)} pages")

    # ── Step 2: Get TOC ───────────────────────────
    print("\nStep 2: Extracting Table of Contents...")
    toc_items = extract_builtin_toc(str(pdf_path))

    if not toc_items:
        print("  No built-in TOC, trying text extraction...")
        toc_items = extract_toc_from_text(pages)

    use_llm_for_tree = len(toc_items) == 0
    if use_llm_for_tree:
        print("  No TOC found — will use LLM for tree building")

    # ── Step 3: Summarize pages (LLM) ────────────
    print("\nStep 3: Summarizing pages...")
    page_summaries = []
    for page in pages:
        summary = summarize_page(page, model, host)
        page_summaries.append(summary)

    # ── Step 4: Build tree ────────────────────────
    print("\nStep 4: Building document tree...")
    if toc_items:
        tree = map_toc_to_pages(toc_items, pages)
        tree = add_summaries_to_tree(tree, page_summaries)
        toc_method = "builtin" if any(
            "page_number" in t for t in toc_items
        ) else "text"
        print(f"  ✅ Tree built from TOC ({len(tree)} nodes)")
    else:
        tree = build_tree_from_summaries(page_summaries, model, host)
        toc_method = "llm"
        print(f"  ✅ Tree built from LLM ({len(tree)} nodes)")

    # ── Step 5: Cross-reference detection (NEW) ───
    tree = add_cross_references_to_tree(tree, page_summaries)

    # ── Step 6: Save ──────────────────────────────
    index_data = {
        "source": str(pdf_path),
        "model_used": model,
        "total_pages": len(pages),
        "toc_method": toc_method,
        "tree": tree,
        "pages": page_summaries
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    # Count total cross-references
    total_refs = sum(len(n.get("cross_references", [])) for n in tree)

    print(f"\n{'='*55}")
    print(f"✅ Index saved to: {output_path}")
    print(f"   Tree nodes:        {len(tree)}")
    print(f"   Pages indexed:     {len(pages)}")
    print(f"   TOC method:        {toc_method}")
    print(f"   Cross-references:  {total_refs} found")
    print(f"{'='*55}\n")

    return index_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a PDF into a JSON tree")
    parser.add_argument("--pdf",    required=True,        help="Path to PDF file")
    parser.add_argument("--model",  default=INDEXER_MODEL,  help="Model name")
    parser.add_argument("--host",   default=INDEXER_HOST,   help="Server URL or groq")
    parser.add_argument("--output",                       help="Output JSON path")
    args = parser.parse_args()

    index_pdf(args.pdf, args.model, args.host, args.output)