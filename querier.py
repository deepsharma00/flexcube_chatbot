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

# ─────────────────────────────────────────
# Query Intent Detection
# ─────────────────────────────────────────

# Patterns that signal a conceptual / overview question (not a how-to)
_CONCEPTUAL_PATTERNS = [
    r'\bwhat\s+is\b',
    r'\bwhat\s+are\b',
    r'\bdefine\b',
    r'\bexplain\b',
    r'\bdescribe\b',
    r'\btell\s+me\s+about\b',
    r'\bwhat\s+does\b',
    r'\bwhat\s+do\b',
    r'\boverview\b',
    r'\bintroduction\b',
]

# Greetings / small-talk phrases — matched before any RAG logic
# NOTE: detect_greeting() normalises repeated letters first (hii → hi)
_GREETING_MAP = [
    # (compiled regex, reply_key)
    (re.compile(r'^(hi|hello|hey|hiya|howdy|greetings|good\s*(morning|afternoon|evening|day))$'), "greeting"),
    (re.compile(r'^how\s+are\s+(you|u)$'),                                                         "how_are_you"),
    (re.compile(r"^what'?s\s+up$"),                                                                 "greeting"),
    (re.compile(r'^(thanks|thank\s+you|thx|ty)$'),                                                  "thanks"),
    (re.compile(r'^(bye|goodbye|see\s+you|cya|take\s+care)$'),                                      "bye"),
    (re.compile(r'^(ok|okay|got\s+it|sure|alright|cool|nice|great|awesome)[!.]*$'),                 "ok"),
    (re.compile(r'^who\s+are\s+you$'),                                                              "who"),
    (re.compile(r'^what\s+(can|do)\s+you\s+do$'),                                                  "what_do"),
]

_GREETING_REPLIES = {
    "greeting": (
        "Hello! 👋 I'm the FlexCube documentation assistant. "
        "Ask me anything about Oracle FlexCube — how screens work, "
        "step-by-step processes, prerequisites, and more."
    ),
    "how_are_you": (
        "I'm doing great, thanks for asking! 😊 "
        "Ready to help you with FlexCube documentation. What would you like to know?"
    ),
    "thanks": (
        "You're welcome! 😊 Let me know if you have any other FlexCube questions."
    ),
    "bye": (
        "Goodbye! Feel free to come back anytime you need FlexCube help. 👋"
    ),
    "ok": (
        "Sure! Feel free to ask me anything about FlexCube documentation."
    ),
    "who": (
        "I'm a RAG-powered assistant built to answer questions about "
        "Oracle FlexCube banking documentation. "
        "Ask me about screens, processes, prerequisites, or any feature — I'll search the docs and answer!"
    ),
    "what_do": (
        "I can answer questions about Oracle FlexCube documentation — "
        "how-to procedures, screen navigation, field explanations, prerequisites, and more. "
        "Just type your question!"
    ),
}


def detect_greeting(query: str) -> str | None:
    """
    Returns a friendly reply string if the query is small-talk / a greeting.
    Returns None if this is a real question (run the full RAG pipeline).

    Handles stretched input like 'Hii', 'Heyyy', 'Thankss' by collapsing
    runs of 3+ identical letters down to 2 before matching.
    Example: 'hiiiii' -> 'hii' -> matches 'hi' pattern after further collapse.
    """
    # Lowercase and strip punctuation/spaces
    q = query.strip().lower()
    q = re.sub(r'[!?,\.]+$', '', q).strip()   # drop trailing punctuation

    # Collapse runs of 3+ same letter to 2: 'hiiiii' -> 'hii', 'heyyy' -> 'heyy'
    q_norm = re.sub(r'(.)\1{2,}', r'\1\1', q)
    # Then collapse runs of 2 same letter to 1: 'hii' -> 'hi', 'heyy' -> 'hey'
    q_norm = re.sub(r'(.)\1+', r'\1', q_norm)

    for pattern, reply_key in _GREETING_MAP:
        if pattern.match(q_norm):
            return _GREETING_REPLIES[reply_key]
    return None


def classify_query(query: str) -> str:
    """
    Returns 'conceptual' for definition/overview questions,
    'procedural' for how-to/process questions.
    """
    q = query.lower()
    for pattern in _CONCEPTUAL_PATTERNS:
        if re.search(pattern, q):
            return 'conceptual'
    return 'procedural'


def expand_query_keywords(query: str, model: str, host: str) -> list:
    """
    Uses the LLM to generate targeted search keywords from the query.
    For procedural questions also injects prerequisite-related terms so
    setup/requirement sections surface alongside main steps.
    For conceptual questions keeps it focused — no setup injection.
    """
    intent = classify_query(query)

    if intent == 'conceptual':
        prereq_rule = (
            "3. Do NOT add prerequisite or setup terms — the user wants a definition/overview."
        )
    else:
        prereq_rule = (
            "3. For process/how-to questions also add prerequisite-related terms to help find "
            "setup sections. Use words like: prerequisite, setup, configuration, required, "
            "maintenance, initial along with the main topic noun. "
            "Example: 'how to create a customer' → also include "
            "'customer prerequisite', 'customer setup', 'customer maintenance'."
        )

    prompt = f"""You are an enterprise search assistant for Oracle FlexCube banking documentation.

Analyze the question below and return a concise list of search keywords.

Rules:
1. Fix any typos in the question.
2. Extract 3-5 critical technical keywords: exact screen names, module names, feature names, or field labels.
{prereq_rule}
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

def _build_conceptual_prompt(query: str, context: str, relevant_sections: list) -> str:
    """Prompt for definition / overview / 'what is' questions."""
    sections_info = ", ".join(f"'{s['title']}'" for s in relevant_sections)
    return f"""You are an expert Oracle FlexCube banking documentation assistant.
Answer the user's question using ONLY the retrieved documentation below.

Sections retrieved: {sections_info}

--- DOCUMENTATION CONTEXT START ---
{context}
--- DOCUMENTATION CONTEXT END ---

User question: {query}

=== INSTRUCTIONS ===
The user is asking a conceptual question — they want to UNDERSTAND what something is,
not follow a procedure.

Write a clear, plain-English answer:
- Start with a 1-2 sentence direct definition or overview.
- Then expand with the key capabilities, modules, or components described in the context.
- Use bullet points for lists of features/modules if helpful.
- Do NOT output step-by-step procedures, screen codes, or navigation paths unless the
  context specifically explains them as part of the concept.
- Do NOT expose internal document section numbers, chapter headings, or file names.
- Use ONLY information from the documentation context above.
- If the context does not contain enough information to fully answer, say so briefly."""


def _build_procedural_prompt(query: str, context: str, relevant_sections: list) -> str:
    """Prompt for how-to / process / step-by-step questions."""
    sections_info = ", ".join(f"'{s['title']}'" for s in relevant_sections)
    return f"""You are an expert Oracle FlexCube banking documentation assistant.
Your goal is to give a COMPLETE, ACTIONABLE answer using ONLY the retrieved documentation below.

Sections retrieved: {sections_info}

--- DOCUMENTATION CONTEXT START ---
{context}
--- DOCUMENTATION CONTEXT END ---

User question: {query}

=== HOW TO STRUCTURE YOUR ANSWER ===

Follow this structure. Skip any section where the context has nothing relevant —
do NOT write "not found"; simply omit that heading.

**Prerequisites / Required Setup**
- Every configuration, maintenance, or access right that must exist BEFORE this task.
- For each prerequisite, provide its full answer from the context:
    1. [Prerequisite name] (Screen: [screen name/code if mentioned])
       What it is and how to complete it.

**How to Access**
- Exact screen name and fastest way to open it:
  Menu path OR screen code to type in the FlexCube toolbar.

**Step-by-Step Process**
- Number every step.
- Quote field names and values exactly as they appear in the documentation.
- Include button clicks, checkboxes, or tab selections.

**Important Fields**
- Key fields and what they control (only if the context describes them).

**Notes & Warnings**
- Validations, authorization steps, system-generated events, or special cases.

=== STRICT RULES ===
- Use ONLY the information in the documentation context above.
- Never invent screen names, field names, or steps.
- Every screen name and screen code found in the context MUST be included.
- If cross-referenced sections are included, use their information too.
- Be specific — users are bank operations staff who need exact screen codes and field names."""


def _build_answer_prompt(query: str, context: str, relevant_sections: list) -> str:
    """Route to the right prompt based on detected query intent."""
    intent = classify_query(query)
    if intent == 'conceptual':
        return _build_conceptual_prompt(query, context, relevant_sections)
    return _build_procedural_prompt(query, context, relevant_sections)


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
    # Greetings bypass — already handled upstream by query_pdf_stream,
    # but guard here too in case called directly.
    greeting = detect_greeting(query)
    if greeting:
        yield greeting
        return
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

    # ── Greeting / small-talk short-circuit ──────────────────────────────
    greeting = detect_greeting(query)
    if greeting:
        return greeting, []   # no RAG, no document context

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