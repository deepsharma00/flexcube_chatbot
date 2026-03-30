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

from config import QUERIER_HOST, QUERIER_MODEL, GROQ_API_KEY


# ─────────────────────────────────────────
# LLM API
# ─────────────────────────────────────────

def call_ollama(prompt: str, model: str, host: str = QUERIER_HOST, timeout: int = 300, num_predict: int = 1024) -> str:
    """Call Ollama or Groq API (non-streaming) and return full response text."""
    if host.lower() == "groq":
        if not GROQ_API_KEY: return "[Error: GROQ_API_KEY not found]"
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": num_predict}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(5)
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[Groq Error: {e}]"
            
    # Default: Ollama
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 16384,
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


def call_ollama_stream(prompt: str, model: str, host: str = QUERIER_HOST, num_predict: int = 8192):
    """Call Ollama or Groq API in streaming mode."""
    if host.lower() == "groq":
        if not GROQ_API_KEY: 
            yield "[Error: GROQ_API_KEY not found]"
            return
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": num_predict, "stream": True}
        try:
            resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=(30, 600))
            if resp.status_code == 429:
                yield "\n[Rate Limit] Sleeping for 5 seconds...\n"
                time.sleep(5)
                resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=(30, 600))
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            token = chunk["choices"][0].get("delta", {}).get("content", "")
                            if token:
                                yield token
        except Exception as e:
            yield f"\n[Groq Error: {e}]"
        return

    # Default: Ollama
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_ctx": 16384,
            "num_predict": num_predict,
        }
    }
    try:
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
    """
    Uses the LLM to generate targeted search keywords from the query.
    Also injects prerequisite-related terms so the section search
    automatically surfaces setup / requirement sections alongside main steps.
    """
    prompt = f"""You are an enterprise search assistant for Oracle FlexCube banking documentation.

Analyze the question below and return a concise list of search keywords.

Rules:
1. Fix any typos in the question.
2. Extract 3-5 critical technical keywords: exact screen names, module names, feature names, or field labels.
3. For process/how-to questions also add prerequisite-related terms that will help find setup
   sections in the documentation. Use words like: prerequisite, setup, configuration, required,
   maintenance, overview, initial along with the main topic noun.
   Example: if the question is "how to create a customer", also include:
   "customer prerequisite", "customer setup", "customer maintenance"
4. Do NOT use broad synonyms (e.g. do not change 'maintenance' to 'management').
5. Keep exact FlexCube terminology.

Question: {query}

Return ONLY a single comma-separated list. No bullets, no numbering, no explanations."""

    response = call_ollama(prompt, model, host, timeout=300, num_predict=80)

    if response.startswith("[Error") or response.startswith("[Timeout"):
        return []

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


def _section_text(item) -> str:
    """
    Safely extract a string from a section item.
    Some indexed JSONs store sections as plain strings ("Section Title"),
    others as dicts ({"title": "Section Title", ...}) depending on which
    LLM response format was produced at index time. This handles both.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # Common keys the LLM might use
        return item.get("title") or item.get("name") or item.get("text") or str(item)
    return str(item)


# Words in a section title/summary that signal it covers prerequisites,
# setup requirements, or overview information. Sections matching these AND
# at least one topic keyword get a score bonus so they surface automatically.
PREREQ_SIGNALS = {
    'prerequisite', 'prerequisites', 'pre-requisite', 'pre-requisites',
    'before you begin', 'requirement', 'requirements', 'mandatory',
    'initial setup', 'configuration', 'overview', 'introduction',
    'maintain', 'maintenance', 'setup', 'setting up',
    'required', 'needed', 'must',
}


def get_query_keywords(query: str) -> list:
    """Extract meaningful keywords from query."""
    words = re.sub(r'[^\w\s]', ' ', query.lower()).split()
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def score_text(text: str, keywords: list) -> float:
    """Score relevance of text to keywords."""
    text_lower = text.lower()
    score = 0.0
    for kw in keywords:
        if kw in text_lower:
            score += 2 if f' {kw} ' in f' {text_lower} ' else 1
    return score


def _prereq_bonus(text: str, topic_keywords: list) -> float:
    """
    Return a small bonus score when a section looks like it covers prerequisites
    AND is topically related to the query.

    Logic:
      - The section text must contain at least one PREREQ_SIGNAL word  →  it's a setup/overview section
      - AND at least one topic keyword                                  →  it's relevant to this query
    Bonus is capped at +1.5 so it can't override a strong direct match.
    """
    text_lower = text.lower()
    has_prereq_signal = any(sig in text_lower for sig in PREREQ_SIGNALS)
    has_topic_match   = any(kw in text_lower for kw in topic_keywords)
    if has_prereq_signal and has_topic_match:
        return 1.5
    return 0.0


def find_relevant_sections(query: str, expanded_keywords: list, index_data: dict) -> list:
    """
    4-layer keyword search with prerequisite boosting:

    Layer 0 (bonus)  : prerequisite/setup sections that are topically related
                       get a +1.5 score boost so they surface alongside main steps.
    Layer 1          : Tree nodes (titles + summaries)
    Layer 2          : Page summaries (catches tree nodes with poor summaries)
    Layer 3          : Raw page text (last resort when Layers 1-2 find nothing)

    Returns up to 6 de-duplicated, scored sections (highest score first).
    """
    tree  = index_data.get("tree", [])
    pages = index_data.get("pages", [])

    # Combine base query keywords with LLM-expanded keywords
    base_keywords = get_query_keywords(query)
    keywords      = list(set(base_keywords + expanded_keywords))

    if not keywords:
        return tree[:3]

    candidates = []

    # ── Layer 1: Tree nodes (title + summary) ────────────────────────────
    for node in tree:
        text  = f"{node.get('title', '')} {node.get('summary', '')}"
        score = score_text(text, keywords)

        # Layer 0 bonus: prerequisite / setup / overview sections
        # that are topically relevant should always be included
        score += _prereq_bonus(text, base_keywords)

        if score > 0:
            candidates.append({
                "score":            score,
                "source":           "tree",
                "structure":        node.get("structure", ""),
                "title":            node.get("title", ""),
                "start_page":       node.get("start_page", 1),
                "end_page":         node.get("end_page", node.get("start_page", 1) + 1),
                "summary":          node.get("summary", ""),
                "cross_references": node.get("cross_references", []),
            })

    # ── Layer 2: Page summaries ───────────────────────────────────────────
    tree_pages = {c["start_page"] for c in candidates}
    for page in pages:
        pn = page["page_number"]
        if pn in tree_pages:
            continue
        sections   = page.get("sections", [])
        text       = " ".join(_section_text(s) for s in sections) + " " + page.get("summary", "")
        score      = score_text(text, keywords) + _prereq_bonus(text, base_keywords)
        if score > 0:
            title = _section_text(sections[0]) if sections else f"Page {pn}"
            candidates.append({
                "score":            score,
                "source":           "page_summary",
                "structure":        str(pn),
                "title":            title,
                "start_page":       pn,
                "end_page":         pn + 1,
                "summary":          page.get("summary", ""),
                "cross_references": [],
            })

    # ── Layer 3: Raw text fallback ────────────────────────────────────────
    if not candidates:
        for page in pages:
            pn    = page["page_number"]
            text  = page.get("raw_text", "")[:2000]
            score = score_text(text, keywords)
            if score > 0:
                sections = page.get("sections", [])
                title    = _section_text(sections[0]) if sections else f"Page {pn}"
                candidates.append({
                    "score":            score,
                    "source":           "raw_text",
                    "structure":        str(pn),
                    "title":            title,
                    "start_page":       pn,
                    "end_page":         pn + 1,
                    "summary":          page.get("summary", ""),
                    "cross_references": [],
                })

    # Sort by score descending, deduplicate by start_page, cap at 6
    candidates.sort(key=lambda x: x["score"], reverse=True)
    seen_pages = set()
    result     = []
    for c in candidates:
        pg = c["start_page"]
        if pg not in seen_pages:
            seen_pages.add(pg)
            result.append(c)
        if len(result) >= 6:
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
    """
    Build the answer prompt.
    Always instructs the LLM to look for prerequisites even when the user
    did not explicitly ask — because banking workflows always have them.
    """
    sections_info = ", ".join(f"'{s['title']}'" for s in relevant_sections)

    return f"""You are an expert Oracle FlexCube banking documentation assistant.
Your goal is to give a COMPLETE, ACTIONABLE answer using ONLY the retrieved documentation below.

Sections retrieved: {sections_info}

--- DOCUMENTATION CONTEXT START ---
{context}
--- DOCUMENTATION CONTEXT END ---

User question: {query}

=== HOW TO STRUCTURE YOUR ANSWER ===

Always follow this structure. Skip any section where the context contains no relevant information —
do NOT write "not found" or "not mentioned"; simply omit that heading.

**Prerequisites / Required Setup**
- Identify every configuration, maintenance, or access right that must exist BEFORE this task.
- For EACH prerequisite, provide its FULL ANSWER using the context — do not just name it:
    1. [Prerequisite name] (Screen: [screen name or code if mentioned])
       → What it is and how to complete it, based on the documentation.
- Look for keywords in context: prerequisite, required, must, initial setup, maintained, configured, overview.

**How to Access**
- Provide the exact screen name and the fastest way to open it:
  menu path (e.g. Retail Teller > Operations > Teller Transaction Input)
  OR the screen code to type in the FlexCube toolbar (e.g. STDCIFCR, TELTXNIN).

**Step-by-Step Process**
- Number every step.
- Quote field names and values exactly as they appear in the documentation.
- Include any button clicks, checkboxes, or tab selections.

**Important Fields**
- List key fields and what they control (only if the context describes them).

**Notes & Warnings**
- Validations, authorization steps, system-generated events, or special cases found in the context.

=== STRICT RULES ===
- Use ONLY the information in the documentation context above.
- Never invent screen names, field names, or steps.
- SCREEN NAMES AND CODES ARE MANDATORY: every screen name (e.g. "Islamic Customer Accounts
  Maintenance") and every screen code (e.g. STDCIFCR, CSDCUSTM, TELTXNIN) that appears
  anywhere in the context MUST be included in your response. Never omit them.
- If cross-referenced sections are included in the context, use their information too.
- If the context has PARTIAL information, share what is available; do not say the whole answer is unavailable.
- Be specific — users are bank operations staff who need exact screen codes and field names."""


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
    parser.add_argument("--model",  default=QUERIER_MODEL,  help="Ollama model name")
    parser.add_argument("--host",   default=QUERIER_HOST,   help="Ollama server URL")
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