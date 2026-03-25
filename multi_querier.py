"""
Layered Multi-Document Querier — Cascading search through Product → Module → Document → Section.

The key insight: instead of loading all document JSONs, we cascade through
lightweight layer indexes. Each layer narrows the search with zero LLM calls.

    Layer 1: product_index.json  → pick 1 product (FCUBS vs ELCM vs OBPM)
    Layer 2: module_index.json   → pick 1-2 modules (common_core vs universal_banking)
    Layer 3: doc_index.json      → pick 1-2 documents (CoreServices vs EMS)
    Layer 4: document.json       → search sections → budget-aware context → LLM answer

Each layer index is tiny (1-5KB). Only the final selected document's JSON is fully loaded.
The LLM is called exactly twice: once for query expansion, once for the answer.

Designed for Mistral:7b (8K context). Context budget caps at ~18,000 chars (~4.5K tokens).
"""

import json
import sys
import argparse
import time
from pathlib import Path

from config import OLLAMA_URL, OLLAMA_MODEL
from querier import (
    get_query_keywords,
    expand_query_keywords,
    find_relevant_sections,
    follow_cross_references,
    answer_question,
    answer_question_stream,
)


# ─────────────────────────────────────────
# JSON Loading
# ─────────────────────────────────────────

def LoadJsonFile(FilePath) -> dict:
    """Load a JSON file. Returns empty dict if not found."""
    FilePath = Path(FilePath)
    if not FilePath.exists():
        print(f"  [WARN] File not found: {FilePath}")
        return {}
    with open(FilePath, "r", encoding="utf-8") as F:
        return json.load(F)


# ─────────────────────────────────────────
# Generic Scoring (works at any layer)
# ─────────────────────────────────────────

def ScoreEntry(Entry: dict, Keywords: list) -> float:
    """
    Score any hierarchy entry (product, module, or document) against query keywords.
    Each entry is expected to have: name, keywords, summary, and optionally top_sections.

    Weights:
      name match (5) > section title (3) > stored keyword (2) > summary (1)
    """
    Score = 0.0
    Name = Entry.get("name", "").lower()
    TopSections = " ".join(Entry.get("top_sections", [])).lower()
    StoredKeywords = " ".join(Entry.get("keywords", [])).lower()
    Summary = Entry.get("summary", "").lower()
    # Also check module/document list names
    SubItems = " ".join(
        Entry.get("modules", []) + Entry.get("documents", [])
    ).lower()

    for Kw in Keywords:
        KwLower = Kw.lower()
        if KwLower in Name:
            Score += 5.0
        if KwLower in SubItems:
            Score += 4.0
        if KwLower in TopSections:
            Score += 3.0
        if KwLower in StoredKeywords:
            Score += 2.0
        if KwLower in Summary:
            Score += 1.0

    return Score


def SelectBestEntries(Entries: list, Keywords: list, TopK: int = 2) -> list:
    """
    Score all entries at a given layer and return the top K.
    Filters out entries scoring below 25% of the top score.
    Falls back to returning first K entries if no keywords match.
    """
    if not Entries:
        return []

    if not Keywords:
        return Entries[:TopK]

    Scored = []
    for Entry in Entries:
        Score = ScoreEntry(Entry, Keywords)
        if Score > 0:
            Scored.append({"score": Score, "entry": Entry})

    if not Scored:
        return Entries[:TopK]

    Scored.sort(key=lambda X: X["score"], reverse=True)

    # Drop entries scoring less than 25% of the best match
    TopScore = Scored[0]["score"]
    MinScore = TopScore * 0.25
    Scored = [S for S in Scored if S["score"] >= MinScore]

    return [S["entry"] for S in Scored[:TopK]]


# ─────────────────────────────────────────
# Context Budget
# ─────────────────────────────────────────

# Mistral:7b num_ctx=8192 tokens. Budget:
#   ~500 tokens for system prompt + question
#   ~2000 tokens reserved for response
#   ~5600 tokens for context → ~22,000 chars
# Using 18,000 as safe limit.
MAX_CONTEXT_CHARS = 18000


def BuildContext(Sections: list, DocTrees: dict, MaxChars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Budget-aware context builder. Processes sections in score order.
    Adds page text one page at a time, stops when character budget is reached.
    Deduplicates pages across sections within the same document.

    DocTrees: dict mapping doc_name → loaded JSON index data
    """
    if not Sections:
        return ""

    SeenPages = {}
    ContextParts = []
    TotalChars = 0

    for Section in Sections:
        if TotalChars >= MaxChars:
            break

        DocName = Section.get("doc_name", "")
        DocTree = DocTrees.get(DocName)
        if not DocTree:
            continue

        Pages = DocTree.get("pages", [])
        PageMap = {P["page_number"]: P for P in Pages}
        TotalDocPages = DocTree.get("total_pages", 999)

        StartPage = Section.get("start_page", 1)
        EndPage = Section.get("end_page", StartPage + 1)

        # +-1 page overlap for boundary content
        FetchStart = max(1, StartPage - 1)
        FetchEnd = min(TotalDocPages + 1, EndPage + 1)

        if DocName not in SeenPages:
            SeenPages[DocName] = set()

        SectionText = ""
        for Pn in range(FetchStart, FetchEnd):
            if Pn in PageMap and Pn not in SeenPages[DocName]:
                Raw = PageMap[Pn].get("raw_text", "")
                if Raw:
                    PageChunk = f"\n[Page {Pn}]\n{Raw}"
                    if TotalChars + len(SectionText) + len(PageChunk) > MaxChars:
                        break
                    SectionText += PageChunk
                    SeenPages[DocName].add(Pn)

        if not SectionText.strip():
            continue

        Title = Section.get("title", "")
        Source = Section.get("source", "")
        DocFilename = Section.get("doc_filename", DocName)

        if Source == "cross_reference":
            RefFrom = Section.get("referenced_from", "")
            Header = f"== {DocFilename} / {Title} (ref from: {RefFrom}) =="
        else:
            Header = f"== {DocFilename} / {Title} =="

        Part = f"{Header}{SectionText}"
        ContextParts.append(Part)
        TotalChars += len(Part)

    return "\n\n".join(ContextParts)


# ─────────────────────────────────────────
# Layered Cascade (Streaming)
# ─────────────────────────────────────────

def QueryLayeredStream(IndexFolder, Query: str, Model: str, Host: str):
    """
    Full layered cascade search — yields (stage, data) tuples for real-time UI.

    Stages:
      ("layer1", selected_products)       — product selection done
      ("layer2", (product, modules))      — module selection done
      ("layer3", (product, module, docs)) — document selection done
      ("expanded", keywords)              — query expansion done (LLM call)
      ("sections", section_list)          — section search done
      ("context_size", char_count)        — context size info
      ("token", text)                     — answer token (stream)

    The first 3 layers use pure keyword matching — instant, no LLM.
    LLM is called only for query expansion (small) and final answer (streaming).
    """
    IndexFolder = Path(IndexFolder)
    Keywords = get_query_keywords(Query)

    # ── Layer 1: Select product ────────────────────
    ProductIndex = LoadJsonFile(IndexFolder / "product_index.json")
    AllProducts = ProductIndex.get("products", [])

    if not AllProducts:
        yield ("token", "No products found in index. Run batch_indexer.py first.")
        return

    # If only 1 product, auto-select it (no need to score)
    if len(AllProducts) == 1:
        SelectedProducts = AllProducts
    else:
        SelectedProducts = SelectBestEntries(AllProducts, Keywords, TopK=1)

    yield ("layer1", SelectedProducts)

    if not SelectedProducts:
        yield ("token", "No matching product found for this query.")
        return

    ProductName = SelectedProducts[0]["name"]

    # ── Layer 2: Select module ─────────────────────
    ModuleIndex = LoadJsonFile(IndexFolder / ProductName / "module_index.json")
    AllModules = ModuleIndex.get("modules", [])

    if not AllModules:
        yield ("token", f"No modules found in {ProductName}.")
        return

    # If only 1 module, auto-select
    if len(AllModules) == 1:
        SelectedModules = AllModules
    else:
        SelectedModules = SelectBestEntries(AllModules, Keywords, TopK=2)

    yield ("layer2", (ProductName, SelectedModules))

    if not SelectedModules:
        yield ("token", f"No matching module found in {ProductName}.")
        return

    # ── Layer 3: Select documents ──────────────────
    # Search across selected modules for best documents
    AllDocEntries = []
    SelectedModuleNames = []

    for Module in SelectedModules:
        ModuleName = Module["name"]
        SelectedModuleNames.append(ModuleName)
        DocIndex = LoadJsonFile(
            IndexFolder / ProductName / ModuleName / "doc_index.json"
        )
        for Doc in DocIndex.get("documents", []):
            Doc["_module"] = ModuleName
            AllDocEntries.append(Doc)

    if not AllDocEntries:
        yield ("token", f"No documents found in {ProductName}/{', '.join(SelectedModuleNames)}.")
        return

    # If only 1 document total, auto-select
    if len(AllDocEntries) == 1:
        SelectedDocs = AllDocEntries
    else:
        SelectedDocs = SelectBestEntries(AllDocEntries, Keywords, TopK=2)

    yield ("layer3", (ProductName, SelectedModuleNames, SelectedDocs))

    # ── Query expansion (only LLM call before answer) ──
    ExpandedKeywords = expand_query_keywords(Query, Model, Host)
    yield ("expanded", ExpandedKeywords)

    # ── Layer 4: Search sections within selected docs ──
    AllSections = []
    DocTrees = {}

    for DocEntry in SelectedDocs:
        ModuleName = DocEntry.get("_module", SelectedModuleNames[0])
        IndexFile = DocEntry.get("index_file", DocEntry["name"] + ".json")
        DocPath = IndexFolder / ProductName / ModuleName / IndexFile

        DocTree = LoadJsonFile(str(DocPath))
        if not DocTree:
            continue

        DocTrees[DocEntry["name"]] = DocTree

        # Reuse existing 3-layer section search
        Sections = find_relevant_sections(Query, ExpandedKeywords, DocTree)
        Sections = follow_cross_references(Sections, DocTree)

        for S in Sections:
            S["doc_name"] = DocEntry["name"]
            S["doc_filename"] = DocEntry.get("filename", DocEntry["name"])
            S["doc_module"] = ModuleName

        AllSections.extend(Sections)

    # Sort by score, filter noise, cap
    AllSections.sort(key=lambda X: X.get("score", 0), reverse=True)

    if AllSections and AllSections[0].get("score", 0) > 0:
        TopScore = AllSections[0]["score"]
        MinScore = TopScore * 0.25
        AllSections = [S for S in AllSections if S.get("score", 0) >= MinScore]

    AllSections = AllSections[:5]
    yield ("sections", AllSections)

    # ── Build context (budget-aware) ───────────────
    Context = BuildContext(AllSections, DocTrees)
    yield ("context_size", len(Context))

    # ── Stream answer ──────────────────────────────
    for Token in answer_question_stream(Query, Context, AllSections, Model, Host):
        yield ("token", Token)


def QueryLayered(IndexFolder, Query: str, Model: str, Host: str):
    """
    Non-streaming version of layered cascade search.
    Returns: (answer_text, sections_list, route_info)
    route_info = {"product": ..., "modules": [...], "documents": [...]}
    """
    IndexFolder = Path(IndexFolder)
    Keywords = get_query_keywords(Query)
    RouteInfo = {}

    # Layer 1: Product
    ProductIndex = LoadJsonFile(IndexFolder / "product_index.json")
    AllProducts = ProductIndex.get("products", [])
    SelectedProducts = (AllProducts if len(AllProducts) == 1
                        else SelectBestEntries(AllProducts, Keywords, TopK=1))
    if not SelectedProducts:
        return "No matching product found.", [], {}

    ProductName = SelectedProducts[0]["name"]
    RouteInfo["product"] = ProductName

    # Layer 2: Module
    ModuleIndex = LoadJsonFile(IndexFolder / ProductName / "module_index.json")
    AllModules = ModuleIndex.get("modules", [])
    SelectedModules = (AllModules if len(AllModules) == 1
                       else SelectBestEntries(AllModules, Keywords, TopK=2))
    if not SelectedModules:
        return f"No matching module in {ProductName}.", [], RouteInfo

    RouteInfo["modules"] = [M["name"] for M in SelectedModules]

    # Layer 3: Document
    AllDocEntries = []
    for Module in SelectedModules:
        DocIndex = LoadJsonFile(
            IndexFolder / ProductName / Module["name"] / "doc_index.json"
        )
        for Doc in DocIndex.get("documents", []):
            Doc["_module"] = Module["name"]
            AllDocEntries.append(Doc)

    SelectedDocs = (AllDocEntries if len(AllDocEntries) == 1
                    else SelectBestEntries(AllDocEntries, Keywords, TopK=2))
    if not SelectedDocs:
        return f"No matching document found.", [], RouteInfo

    RouteInfo["documents"] = [D["name"] for D in SelectedDocs]

    # Query expansion
    ExpandedKeywords = expand_query_keywords(Query, Model, Host)

    # Layer 4: Section search
    AllSections = []
    DocTrees = {}

    for DocEntry in SelectedDocs:
        ModuleName = DocEntry.get("_module", RouteInfo["modules"][0])
        IndexFile = DocEntry.get("index_file", DocEntry["name"] + ".json")
        DocPath = IndexFolder / ProductName / ModuleName / IndexFile

        DocTree = LoadJsonFile(str(DocPath))
        if not DocTree:
            continue

        DocTrees[DocEntry["name"]] = DocTree
        Sections = find_relevant_sections(Query, ExpandedKeywords, DocTree)
        Sections = follow_cross_references(Sections, DocTree)

        for S in Sections:
            S["doc_name"] = DocEntry["name"]
            S["doc_filename"] = DocEntry.get("filename", DocEntry["name"])
            S["doc_module"] = ModuleName

        AllSections.extend(Sections)

    AllSections.sort(key=lambda X: X.get("score", 0), reverse=True)

    if AllSections and AllSections[0].get("score", 0) > 0:
        TopScore = AllSections[0]["score"]
        AllSections = [S for S in AllSections if S["score"] >= TopScore * 0.25]

    AllSections = AllSections[:5]

    Context = BuildContext(AllSections, DocTrees)
    Answer = answer_question(Query, Context, AllSections, Model, Host)

    return Answer, AllSections, RouteInfo


# ─────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────

if __name__ == "__main__":
    Parser = argparse.ArgumentParser(
        description="Layered cascade search: Product → Module → Document → Section"
    )
    Parser.add_argument(
        "--index", required=True,
        help="Path to index folder (contains product_index.json)"
    )
    Parser.add_argument(
        "--query", required=True,
        help="Your question"
    )
    Parser.add_argument(
        "--model", default=OLLAMA_MODEL,
        help="Ollama model name"
    )
    Parser.add_argument(
        "--host", default=OLLAMA_URL,
        help="Ollama server URL"
    )
    Args = Parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Query: {Args.query}")
    print(f"{'='*55}\n")

    Start = time.time()

    for Stage, Data in QueryLayeredStream(Args.index, Args.query, Args.model, Args.host):
        if Stage == "layer1":
            Names = [P["name"] for P in Data]
            print(f"  Layer 1 - Product:  {', '.join(Names)}")

        elif Stage == "layer2":
            ProductName, Modules = Data
            Names = [M["name"] for M in Modules]
            print(f"  Layer 2 - Module:   {ProductName} -> {', '.join(Names)}")

        elif Stage == "layer3":
            ProductName, ModuleNames, Docs = Data
            Names = [D["name"] for D in Docs]
            print(f"  Layer 3 - Document: {', '.join(Names)}")

        elif Stage == "expanded":
            if Data:
                print(f"  Query expanded:     +{', '.join(Data[:4])}")
            else:
                print(f"  Query expanded:     (no extra terms)")

        elif Stage == "sections":
            print(f"  Sections found:     {len(Data)}")
            for S in Data:
                SrcIcon = {
                    "tree": "+", "page_summary": "*",
                    "raw_text": "#", "cross_reference": "@"
                }.get(S.get("source", ""), "-")
                print(f"   {SrcIcon} [{S.get('doc_module','')}/{S['structure']}] "
                      f"{S['title']} (score:{S['score']})")

        elif Stage == "context_size":
            print(f"  Context size:       {Data:,} chars "
                  f"({Data*100//MAX_CONTEXT_CHARS}% of budget)")
            print(f"\n{'='*55}")
            print("  Answer:")
            print(f"{'='*55}")

        elif Stage == "token":
            print(Data, end="", flush=True)

    Elapsed = time.time() - Start
    print(f"\n{'='*55}")
    print(f"  ({Elapsed:.1f}s)")
    print(f"{'='*55}\n")
