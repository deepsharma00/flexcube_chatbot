"""
PDF Querier v4 - Query your indexed PDF using Ollama
Improvements over v3:
- LLM Query Expansion: Generates synonyms before searching to fix vocabulary mismatches
- Follows cross-references stored in index
- Overlap pages: ±1 page buffer around sections
- 3-layer search: tree → page summaries → raw text
- Section boundary aware context (no arbitrary cuts)
"""

import re
import json
import sys
import time
import argparse
from pathlib import Path

import requests

from config import OLLAMA_URL, OLLAMA_MODEL


# ─────────────────────────────────────────
# Ollama API
# ─────────────────────────────────────────

def call_ollama(prompt: str, model: str, host: str = OLLAMA_URL, timeout: int = 300, num_predict: int = 1024) -> str:
    """Call Ollama API (non-streaming) and return full response text. Best for short outputs like query expansion."""
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "num_predict": num_predict,
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        return "[Timeout — model took too long]"
    except Exception as e:
        return f"[Error: {e}]"


def call_ollama_stream(prompt: str, model: str, host: str = OLLAMA_URL, num_predict: int = 8192):
    """
    Call Ollama API in streaming mode — yields tokens as they arrive.
    Avoids timeout issues because the connection stays alive while tokens flow.
    Each chunk only needs to arrive within connect_timeout, not the entire response.
    """
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "num_predict": num_predict,
        }
    }
    try:
        # connect timeout = 30s, read timeout = 600s (per chunk, not total)
        resp = requests.post(url, json=payload, stream=True, timeout=(30, 600))
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break
    except requests.exceptions.Timeout:
        yield "\n[Timeout — model took too long]"
    except Exception as e:
        yield f"\n[Error: {e}]"


# ─────────────────────────────────────────
# Index Loading
# ─────────────────────────────────────────

def load_index(index_path: str) -> dict:
    """Load the JSON index file."""
    path = Path(index_path)
    if not path.exists():
        print(f"Error: Index not found: {index_path}")
        print("Run indexer.py first to create the index.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────
# Query Expansion (NEW)
# ─────────────────────────────────────────

def expand_query_keywords(query: str, model: str, host: str) -> list:
    """Uses the LLM to generate synonyms and related terms for the query."""
    prompt = f"""You are a search assistant. Look at this question and generate 4-6 highly relevant synonyms, alternative phrasings, or related technical terms.
    
    Question: {query}
    
    Return ONLY a single comma-separated list of words. No explanations, no markdown."""

    # Using a small num_predict since we only need a few words
    # You can hardcode a smaller model here (like 'llama3.2:1b') if needed for speed/memory
    response = call_ollama(prompt, model, host, timeout=300, num_predict=50)
    
    if response.startswith("[Error") or response.startswith("[Timeout"):
        return []
        
    # Split comma-separated phrases into individual words
    words = []
    for phrase in response.split(","):
        words.extend(phrase.strip().lower().split())
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


# ─────────────────────────────────────────
# Keyword Search
# ─────────────────────────────────────────

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'what', 'how', 'when', 'where', 'why', 'who', 'which',
    'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by',
    'this', 'that', 'these', 'those', 'do', 'does', 'did',
    'can', 'could', 'will', 'would', 'should', 'tell', 'me',
    'about', 'give', 'show', 'explain', 'describe', 'list',
    'and', 'or', 'but', 'if', 'then', 'so', 'also', 'it'
}


def get_query_keywords(query: str) -> list:
    """Extract meaningful keywords from query."""
    words = re.sub(r'[^\w\s]', ' ', query.lower()).split()
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def score_text(text: str, keywords: list) -> float:
    """Score relevance of text to keywords."""
    text_lower = text.lower()
    score = 0
    for kw in keywords:
        if kw in text_lower:
            score += 2 if f' {kw} ' in f' {text_lower} ' else 1
    return score


def find_relevant_sections(query: str, expanded_keywords: list, index_data: dict) -> list:
    """
    3-layer keyword search (Now with Query Expansion):
    Layer 1: Tree nodes (titles + summaries)
    Layer 2: Page summaries (catches missed tree nodes)
    Layer 3: Raw page text (last resort)
    Returns scored list of relevant sections.
    """
    tree  = index_data.get("tree", [])
    pages = index_data.get("pages", [])
    
    # Combine user's base keywords with LLM expanded keywords
    base_keywords = get_query_keywords(query)
    keywords = list(set(base_keywords + expanded_keywords))

    if not keywords:
        return tree[:3]

    candidates = []

    # ── Layer 1: Tree nodes ───────────────────────
    for node in tree:
        text  = f"{node.get('title','')} {node.get('summary','')}"
        score = score_text(text, keywords)
        if score > 0:
            candidates.append({
                "score":      score,
                "source":     "tree",
                "structure":  node.get("structure", ""),
                "title":      node.get("title", ""),
                "start_page": node.get("start_page", 1),
                "end_page":   node.get("end_page", node.get("start_page", 1) + 1),
                "summary":    node.get("summary", ""),
                "cross_references": node.get("cross_references", [])
            })

    # ── Layer 2: Page summaries ───────────────────
    tree_pages = {c["start_page"] for c in candidates}
    for page in pages:
        pn = page["page_number"]
        if pn in tree_pages:
            continue
        text = " ".join(page.get("sections", [])) + " " + page.get("summary", "")
        score = score_text(text, keywords)
        if score > 0:
            title = (page["sections"][0] if page.get("sections")
                     else f"Page {pn}")
            candidates.append({
                "score":      score,
                "source":     "page_summary",
                "structure":  str(pn),
                "title":      title,
                "start_page": pn,
                "end_page":   pn + 1,
                "summary":    page.get("summary", ""),
                "cross_references": []
            })

    # ── Layer 3: Raw text fallback ────────────────
    if not candidates:
        for page in pages:
            pn   = page["page_number"]
            text = page.get("raw_text", "")[:2000]
            score = score_text(text, keywords)
            if score > 0:
                title = (page["sections"][0] if page.get("sections")
                         else f"Page {pn}")
                candidates.append({
                    "score":      score,
                    "source":     "raw_text",
                    "structure":  str(pn),
                    "title":      title,
                    "start_page": pn,
                    "end_page":   pn + 1,
                    "summary":    page.get("summary", ""),
                    "cross_references": []
                })

    # Sort by score, deduplicate by page
    candidates.sort(key=lambda x: x["score"], reverse=True)
    seen_pages = set()
    result = []
    for c in candidates:
        pg = c["start_page"]
        if pg not in seen_pages:
            seen_pages.add(pg)
            result.append(c)
        if len(result) >= 4:
            break

    return result


# ─────────────────────────────────────────
# Cross Reference Following 
# ─────────────────────────────────────────

def follow_cross_references(relevant: list, index_data: dict) -> list:
    """
    For each found section, check its cross_references.
    Automatically load referenced sections too.
    """
    tree = index_data.get("tree", [])
    tree_map = {n["structure"]: n for n in tree}

    already_loaded = {s["start_page"] for s in relevant}
    extra_sections = []

    for section in relevant:
        refs = section.get("cross_references", [])
        for ref in refs:
            ref_structure = ref.get("structure", "")
            ref_page = ref.get("start_page", 0)

            # Skip if already in context
            if ref_page in already_loaded:
                continue

            # Find full node in tree
            if ref_structure in tree_map:
                node = tree_map[ref_structure]
                extra_sections.append({
                    "score":      0,  # lower priority than direct matches
                    "source":     "cross_reference",
                    "structure":  node.get("structure", ""),
                    "title":      node.get("title", ""),
                    "start_page": node.get("start_page", 1),
                    "end_page":   node.get("end_page",
                                          node.get("start_page", 1) + 1),
                    "summary":    node.get("summary", ""),
                    "cross_references": [],
                    "referenced_from": section.get("title", "")
                })
                already_loaded.add(ref_page)

    return relevant + extra_sections


# ─────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────

def get_page_context(relevant_sections: list, index_data: dict) -> str:
    """
    Get complete section text using tree boundaries.
    Includes ±1 page overlap to catch boundary content.
    No arbitrary character cuts.
    """
    pages    = index_data.get("pages", [])
    page_map = {p["page_number"]: p for p in pages}
    total_pages = index_data.get("total_pages", 999)

    context_parts = []
    seen_pages    = set()

    for section in relevant_sections:
        start_page = section.get("start_page", 1)
        end_page   = section.get("end_page", start_page + 1)
        source     = section.get("source", "")
        title      = section.get("title", f"Page {start_page}")

        # ±1 overlap: one page before and after section
        fetch_start = max(1, start_page - 1)
        fetch_end   = min(total_pages + 1, end_page + 1)

        section_text = ""
        for pn in range(fetch_start, fetch_end):
            if pn in page_map and pn not in seen_pages:
                seen_pages.add(pn)
                raw = page_map[pn].get("raw_text", "")
                if raw:
                    section_text += f"\n[Page {pn}]\n{raw}"

        if section_text:
            # Label cross-referenced sections clearly
            if source == "cross_reference":
                ref_from = section.get("referenced_from", "")
                header = f"=== {title} (referenced from: {ref_from}) ==="
            else:
                header = f"=== {title} ==="

            context_parts.append(f"{header}\n{section_text}")

    return "\n\n".join(context_parts)


# ─────────────────────────────────────────
# Answer Generation
# ─────────────────────────────────────────

def _build_answer_prompt(query: str, context: str, relevant_sections: list) -> str:
    """Build the prompt for answer generation — shared by streaming and non-streaming paths."""
    sections_info = ", ".join([f"'{s['title']}'" for s in relevant_sections])
    return f"""You are a document assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "This information was not found in the retrieved sections."

Sections retrieved: {sections_info}

Document Context:
{context}

Question: {query}

Instructions:
- If partial information is found, answer ONLY using that
- Do NOT say "not found" if some relevant steps exist
- Answer directly and clearly
- Use only information from the context above
- If steps or lists are present, format them clearly
- Do not make up any information not in the context"""


def answer_question(query: str, context: str,
                    relevant_sections: list,
                    model: str, host: str) -> str:
    """Generate answer using retrieved context (non-streaming, used by CLI/query_pdf)."""
    if not context:
        return "Could not find relevant content in the document."
    prompt = _build_answer_prompt(query, context, relevant_sections)
    return call_ollama(prompt, model, host, timeout=300)


def answer_question_stream(query: str, context: str,
                           relevant_sections: list,
                           model: str, host: str):
    """
    Generate answer in streaming mode — yields tokens as they arrive.
    Use this in the interactive chat to avoid timeouts on large contexts.
    """
    if not context:
        yield "Could not find relevant content in the document."
        return
    prompt = _build_answer_prompt(query, context, relevant_sections)
    yield from call_ollama_stream(prompt, model, host)


# ─────────────────────────────────────────
# Public Query Function
# ─────────────────────────────────────────

def query_pdf(index_path: str, query: str, model: str, host: str):
    """Full query pipeline with cross-reference following and query expansion."""
    index_data = load_index(index_path)

    # Step 1: Expand query vocabulary
    expanded_keywords = expand_query_keywords(query, model, host)

    # Step 2: keyword search (using expanded words)
    relevant = find_relevant_sections(query, expanded_keywords, index_data)

    # Step 3: follow cross references
    relevant = follow_cross_references(relevant, index_data)

    # Step 4: build context with overlap
    context = get_page_context(relevant, index_data)

    # Step 5: generate answer
    answer = answer_question(query, context, relevant, model, host)

    return answer, relevant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query an indexed PDF")
    parser.add_argument("--index",  required=True,        help="Path to JSON index")
    parser.add_argument("--query",  required=True,        help="Your question")
    parser.add_argument("--model",  default=OLLAMA_MODEL,  help="Ollama model name")
    parser.add_argument("--host",   default=OLLAMA_URL,   help="Ollama server URL")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"🔍 Query: {args.query}")
    print(f"{'='*55}\n")

    index_data = load_index(args.index)

    print("🧠 Expanding query vocabulary...", end=" ", flush=True)
    expanded_keywords = expand_query_keywords(args.query, args.model, args.host)
    if expanded_keywords:
        print(f"✓ (Added: {', '.join(expanded_keywords[:4])})")
    else:
        print("✓ (No extra terms generated)")

    print("Finding relevant sections...", end=" ", flush=True)
    relevant = find_relevant_sections(args.query, expanded_keywords, index_data)
    print(f"✓ {len(relevant)} found")

    print("Following cross-references...", end=" ", flush=True)
    relevant = follow_cross_references(relevant, index_data)
    print(f"✓ {len(relevant)} total after following refs")

    if relevant:
        print("\n📍 Sections loaded:")
        for s in relevant:
            src_icon = {"tree": "🌲", "page_summary": "📄",
                        "raw_text": "📝", "cross_reference": "🔗"
                        }.get(s.get("source", ""), "•")
            ref_info = (f" ← from '{s['referenced_from']}'"
                        if s.get("source") == "cross_reference" else "")
            print(f"   {src_icon} [{s['structure']}] {s['title']}"
                  f" — p.{s['start_page']}  (score:{s['score']}){ref_info}")

    context = get_page_context(relevant, index_data)

    print("\nGenerating answer...", end=" ", flush=True)
    start = time.time()
    answer = answer_question(args.query, context, relevant, args.model, args.host)
    print(f"✓ ({time.time()-start:.1f}s)")

    print(f"\n{'='*55}")
    print("📝 ANSWER:")
    print(f"{'='*55}")
    print(answer)
    print(f"{'='*55}\n")