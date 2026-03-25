"""
Multi-Document Layered Chat — Interactive chat with hierarchical document search.

Automatically cascades through Product → Module → Document → Section
to find the right answer. Shows which path was taken at each layer.

Usage:
    python multi_chat.py --index index/
    python multi_chat.py --index index/

Commands inside chat:
    products  → list all products
    modules   → list all modules per product
    docs      → list all documents per module
    tree      → show full hierarchy
    quit      → exit
"""

import argparse
import requests
from pathlib import Path

from config import OLLAMA_URL, OLLAMA_MODEL
from multi_querier import (
    LoadJsonFile,
    QueryLayeredStream,
    MAX_CONTEXT_CHARS,
)


# ─────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────

def PrintFullTree(IndexFolder: str):
    """Show the complete Product → Module → Document hierarchy."""
    IndexFolder = Path(IndexFolder)
    ProductIndex = LoadJsonFile(IndexFolder / "product_index.json")
    Products = ProductIndex.get("products", [])

    if not Products:
        print("  No products indexed yet.")
        return

    TotalProducts = len(Products)
    TotalDocs = sum(P.get("total_documents", 0) for P in Products)
    print(f"\n  Document Hierarchy ({TotalProducts} products, {TotalDocs} documents)")
    print(f"  {'─'*50}")

    for Product in Products:
        ProductName = Product["name"]
        ModuleCount = Product.get("module_count", 0)
        DocCount = Product.get("total_documents", 0)
        print(f"\n  [{ProductName}] ({ModuleCount} modules, {DocCount} docs)")

        # Load module index for this product
        ModuleIndex = LoadJsonFile(IndexFolder / ProductName / "module_index.json")
        for Module in ModuleIndex.get("modules", []):
            ModuleName = Module["name"]
            ModuleDocCount = Module.get("document_count", 0)
            print(f"    [{ModuleName}] ({ModuleDocCount} docs)")

            # Load doc index for this module
            DocIndex = LoadJsonFile(
                IndexFolder / ProductName / ModuleName / "doc_index.json"
            )
            for Doc in DocIndex.get("documents", []):
                Pages = Doc.get("total_pages", 0)
                Sections = Doc.get("total_sections", 0)
                print(f"      - {Doc['name']} ({Pages}p, {Sections} sections)")

    print()


def PrintProducts(IndexFolder: str):
    """List all products with their module and document counts."""
    ProductIndex = LoadJsonFile(Path(IndexFolder) / "product_index.json")
    Products = ProductIndex.get("products", [])

    print(f"\n  Products ({len(Products)}):")
    print(f"  {'─'*50}")
    for P in Products:
        Summary = P.get("summary", "")[:80]
        print(f"  [{P['name']}] {P.get('module_count',0)} modules, "
              f"{P.get('total_documents',0)} docs")
        if Summary:
            print(f"    {Summary}...")
    print()


def PrintModules(IndexFolder: str):
    """List all modules per product."""
    IndexFolder = Path(IndexFolder)
    ProductIndex = LoadJsonFile(IndexFolder / "product_index.json")

    for Product in ProductIndex.get("products", []):
        ProductName = Product["name"]
        ModuleIndex = LoadJsonFile(IndexFolder / ProductName / "module_index.json")
        Modules = ModuleIndex.get("modules", [])

        print(f"\n  [{ProductName}] ({len(Modules)} modules):")
        for M in Modules:
            DocCount = M.get("document_count", 0)
            TopDocs = ", ".join(M.get("documents", [])[:3])
            print(f"    [{M['name']}] {DocCount} docs: {TopDocs}")
    print()


def PrintDocs(IndexFolder: str):
    """List all documents with their top sections."""
    IndexFolder = Path(IndexFolder)
    ProductIndex = LoadJsonFile(IndexFolder / "product_index.json")

    for Product in ProductIndex.get("products", []):
        ProductName = Product["name"]
        ModuleIndex = LoadJsonFile(IndexFolder / ProductName / "module_index.json")

        for Module in ModuleIndex.get("modules", []):
            ModuleName = Module["name"]
            DocIndex = LoadJsonFile(
                IndexFolder / ProductName / ModuleName / "doc_index.json"
            )
            Docs = DocIndex.get("documents", [])
            if not Docs:
                continue

            print(f"\n  [{ProductName}/{ModuleName}]")
            for Doc in Docs:
                Pages = Doc.get("total_pages", 0)
                TopSections = Doc.get("top_sections", [])[:3]
                print(f"    {Doc['name']} ({Pages} pages)")
                if TopSections:
                    print(f"      Sections: {', '.join(TopSections)}")
    print()


# ─────────────────────────────────────────
# Model Warmup
# ─────────────────────────────────────────

def WarmupModel(Model: str, Host: str):
    """Pre-load the Ollama model into RAM."""
    print("Warming up model...", end=" ", flush=True)
    try:
        requests.post(
            f"{Host}/api/generate",
            json={
                "model": Model,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 1}
            },
            timeout=300
        )
        print("done")
    except Exception as E:
        print(f"Warning: {E}")


# ─────────────────────────────────────────
# Main Chat Loop
# ─────────────────────────────────────────

def MultiChat(IndexFolder: str, Model: str, Host: str):
    """
    Interactive layered chat loop.
    Each question automatically cascades through Product → Module → Document → Section.
    Shows the routing path so you can see which document was selected at each layer.
    """
    IndexFolder = Path(IndexFolder)

    print("""
+-----------------------------------------------+
|       Multi-Document Layered RAG Chat         |
|       Product -> Module -> Doc -> Section     |
+-----------------------------------------------+
""")

    # Load product index for summary info
    print("Loading index...", end=" ", flush=True)
    ProductIndex = LoadJsonFile(IndexFolder / "product_index.json")
    Products = ProductIndex.get("products", [])

    if not Products:
        print("FAILED")
        print("No product_index.json found. Run batch_indexer.py first.")
        return

    TotalProducts = len(Products)
    TotalDocs = sum(P.get("total_documents", 0) for P in Products)
    print("done")

    WarmupModel(Model, Host)

    # Show summary
    ProductNames = [P["name"] for P in Products]
    print(f"""
   Products  : {TotalProducts} ({', '.join(ProductNames)})
   Documents : {TotalDocs} total
   Model     : {Model}

Commands:
  tree      -> full hierarchy
  products  -> list products
  modules   -> list modules
  docs      -> list documents
  quit      -> exit
""")

    # ── Chat loop ──────────────────────────────────
    while True:
        try:
            Query = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not Query:
            continue

        Cmd = Query.lower()

        if Cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if Cmd == "tree":
            PrintFullTree(str(IndexFolder))
            continue

        if Cmd == "products":
            PrintProducts(str(IndexFolder))
            continue

        if Cmd == "modules":
            PrintModules(str(IndexFolder))
            continue

        if Cmd == "docs":
            PrintDocs(str(IndexFolder))
            continue

        # ── Layered cascade query ──────────────────
        print()

        for Stage, Data in QueryLayeredStream(str(IndexFolder), Query, Model, Host):
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
                if Data:
                    print(f"  Sections found:     {len(Data)}")
                    for S in Data:
                        SrcIcon = {
                            "tree": "+", "page_summary": "*",
                            "raw_text": "#", "cross_reference": "@"
                        }.get(S.get("source", ""), "-")
                        print(f"   {SrcIcon} [{S.get('doc_module','')}/{S['structure']}] "
                              f"{S['title']} (score:{S['score']})")
                else:
                    print("  No relevant sections found.")

            elif Stage == "context_size":
                Pct = Data * 100 // MAX_CONTEXT_CHARS if MAX_CONTEXT_CHARS > 0 else 0
                print(f"  Context:            {Data:,} chars ({Pct}% of budget)")
                print()
                print("Answer:")
                print("-" * 50)

            elif Stage == "token":
                print(Data, end="", flush=True)

        print("\n" + "-" * 50)
        print()


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────

if __name__ == "__main__":
    Parser = argparse.ArgumentParser(
        description="Interactive layered chat: Product -> Module -> Document -> Section"
    )
    Parser.add_argument(
        "--index", required=True,
        help="Path to index folder (contains product_index.json)"
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

    MultiChat(str(Args.index), Args.model, Args.host)
