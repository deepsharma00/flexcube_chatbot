"""
PDF RAG Chat v4 - Interactive chat with your indexed PDF
Shows cross-referenced sections clearly in output.
Now includes LLM Query Expansion.
"""

import argparse
import requests
from pathlib import Path

from config import QUERIER_HOST, QUERIER_MODEL
from querier import (
    load_index,
    expand_query_keywords,
    find_relevant_sections,
    follow_cross_references,
    get_page_context,
    answer_question_stream,
    _section_text,
)


# ─────────────────────────────────────────
# Tree Display
# ─────────────────────────────────────────

def print_tree(index_data: dict):
    """Print document tree with cross-reference info."""
    tree  = index_data.get("tree", [])
    pages = index_data.get("total_pages", 0)

    print(f"\n📋 Document Structure")
    print(f"   File:   {Path(index_data.get('source', '')).name}")
    print(f"   Pages:  {pages}")
    print(f"   Nodes:  {len(tree)}")
    print(f"   Method: {index_data.get('toc_method', 'unknown')}")
    print()

    for node in tree:
        structure = node.get("structure", "")
        depth     = structure.count(".")
        indent    = "    " * depth
        title     = node.get("title", "")
        page      = node.get("start_page", "?")
        end_page  = node.get("end_page", "?")
        summary   = node.get("summary", "")[:60]
        refs      = node.get("cross_references", [])

        print(f"  {indent}[{structure}] {title}  (p.{page}-{end_page})")
        if summary:
            print(f"  {indent}     → {summary}...")
        if refs:
            ref_list = ", ".join(
                f"{r['structure']} ({r['title'][:25]})" for r in refs
            )
            print(f"  {indent}     🔗 refs: {ref_list}")
    print()


# ─────────────────────────────────────────
# Warmup
# ─────────────────────────────────────────

def warmup_model(model: str, host: str):
    """Pre-load model into RAM before chat starts."""
    print("Warming up model...", end=" ", flush=True)
    try:
        requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 1}
            },
            timeout=300
        )
        print("✓")
    except Exception as e:
        print(f"Warning: {e}")


# ─────────────────────────────────────────
# Main Chat Loop
# ─────────────────────────────────────────

def chat(index_path: str, model: str, host: str):
    """Interactive chat loop."""

    print("""
╔══════════════════════════════════════════╗
║       PDF RAG Chat v4                    ║
║       with LLM Query Expansion           ║
╚══════════════════════════════════════════╝
""")

    # ── Load index ────────────────────────────
    print("Loading index...", end=" ", flush=True)
    index_data  = load_index(index_path)
    source      = Path(index_data.get("source", "Unknown")).name
    total_pages = index_data.get("total_pages", 0)
    sections    = len(index_data.get("tree", []))
    toc_method  = index_data.get("toc_method", "unknown")
    total_refs  = sum(
        len(n.get("cross_references", []))
        for n in index_data.get("tree", [])
    )
    print("✓")

    # ── Warmup model ──────────────────────────
    warmup_model(model, host)

    # ── Print document info ───────────────────
    print(f"""
📄 Document        : {source}
📃 Pages           : {total_pages}
🌲 Sections        : {sections}
🔗 Cross-references: {total_refs} stored
📑 TOC method      : {toc_method}
🤖 Model           : {model}

Commands:
  tree    → show document structure
  pages   → show page summaries
  quit    → exit
""")

    # ── Main loop ─────────────────────────────
    while True:
        try:
            query = input("❓ Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        cmd = query.lower()

        # ── Commands ──────────────────────────
        if cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if cmd == "tree":
            print_tree(index_data)
            continue

        if cmd == "pages":
            print("\n📄 All Page Summaries:")
            for p in index_data.get("pages", []):
                secs    = p.get("sections", [])
                summary = p.get("summary", "")[:80]
                print(f"  Page {p['page_number']:2d}: "
                      f"{', '.join(_section_text(s) for s in secs[:2]) or 'no headings'}")
                print(f"         {summary}")
            print()
            continue

        # ── Answer question ───────────────────
        print()

        # Step 1: Expand query vocabulary
        print("🧠 Expanding query...", end=" ", flush=True)
        expanded_keywords = expand_query_keywords(query, model, host)
        if expanded_keywords:
            print(f"✓ (Added: {', '.join(expanded_keywords[:4])})")
        else:
            print("✓ (No extra terms)")

        # Step 2: Find relevant sections (keyword search)
        relevant = find_relevant_sections(query, expanded_keywords, index_data)

        # Step 3: Follow cross references
        relevant = follow_cross_references(relevant, index_data)

        # Step 4: Show what was found
        if relevant:
            print("📍 Sections loaded:")
            for s in relevant:
                src_icon = {
                    "tree":            "🌲",
                    "page_summary":    "📄",
                    "raw_text":        "📝",
                    "cross_reference": "🔗"
                }.get(s.get("source", ""), "•")

                ref_info = ""
                if s.get("source") == "cross_reference":
                    ref_info = f"  ← from '{s.get('referenced_from', '')}'"

                print(f"   {src_icon} [{s['structure']}] "
                      f"{s['title']} — p.{s['start_page']}{ref_info}")
        else:
            print("⚠️  No sections found, searching full document...")

        # Step 5: Build context and stream answer token-by-token
        context = get_page_context(relevant, index_data)
        print("💭 Thinking...\n")

        print("📝 Answer:")
        print("─" * 50)
        for token in answer_question_stream(query, context, relevant, model, host):
            print(token, end="", flush=True)
        print("\n" + "─" * 50)
        print()


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with your indexed PDF")
    parser.add_argument("--index", required=True,        help="Path to JSON index")
    parser.add_argument("--model", default=QUERIER_MODEL,  help="Ollama model")
    parser.add_argument("--host",  default=QUERIER_HOST,   help="Ollama server URL")
    args = parser.parse_args()

    chat(args.index, args.model, args.host)