"""
Hierarchical Batch PDF Indexer — Builds layered indexes for multi-product document sets.

Designed for products like FlexCube where the structure is:
    Product → Module → Documents

Expected folder structure:
    Root/
    ├── FCUBS/                         ← Product (Layer 1)
    │   ├── common_core/               ← Module (Layer 2)
    │   │   ├── CoreServices.pdf       ← Document (Layer 3)
    │   │   └── EMS.pdf
    │   └── universal_banking/
    │       └── Retail.pdf
    ├── ELCM/
    │   └── lc_module/
    │       └── LCProcessing.pdf
    └── OBPM/
        └── ...

Output (3 layers of indexes + document trees):
    index/
    ├── product_index.json             ← Layer 1: which product?
    ├── FCUBS/
    │   ├── module_index.json          ← Layer 2: which module?
    │   ├── common_core/
    │   │   ├── doc_index.json         ← Layer 3: which document?
    │   │   ├── CoreServices.json      ← Document tree (existing format)
    │   │   └── EMS.json
    │   └── universal_banking/
    │       ├── doc_index.json
    │       └── Retail.json
    ├── ELCM/
    │   ├── module_index.json
    │   └── ...
    └── OBPM/
        └── ...

At query time, each layer is searched with pure keyword matching (no LLM).
The LLM is only called once — at the final section-level answer step.
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

from config import INDEXER_HOST, INDEXER_MODEL
from indexer import index_pdf


# Words to skip when extracting keywords from TOC titles
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'and', 'or', 'but', 'of', 'in', 'on', 'at', 'for', 'with',
    'to', 'by', 'this', 'that', 'it', 'from', 'as', 'not', 'no',
    'has', 'have', 'had', 'will', 'can', 'may', 'shall', 'should'
}


# ─────────────────────────────────────────
# Hierarchy Scanning
# ─────────────────────────────────────────

def ScanHierarchy(RootPath: str) -> list:
    """
    Scan folder structure to detect Product/Module/Document hierarchy.

    Returns list of products, each containing modules, each containing documents:
    [
      {
        "name": "FCUBS",
        "modules": [
          {
            "name": "common_core",
            "documents": [
              {"stem": "CoreServices", "filename": "CoreServices.pdf", "pdf_path": "..."}
            ]
          }
        ]
      }
    ]

    PDFs directly under a product (no module subfolder) get module="general".
    """
    Root = Path(RootPath)
    if not Root.exists():
        print(f"Error: Folder not found: {Root}")
        sys.exit(1)

    Products = []

    for ProductDir in sorted(Root.iterdir()):
        if not ProductDir.is_dir():
            continue

        Product = {"name": ProductDir.name, "modules": []}

        # Scan for module subdirectories containing PDFs
        for ModuleDir in sorted(ProductDir.iterdir()):
            if not ModuleDir.is_dir():
                continue

            Pdfs = sorted(ModuleDir.glob("*.pdf"))
            if Pdfs:
                Product["modules"].append({
                    "name": ModuleDir.name,
                    "documents": [
                        {"stem": P.stem, "filename": P.name, "pdf_path": str(P)}
                        for P in Pdfs
                    ]
                })

        # PDFs directly under product (no module folder) → module="general"
        DirectPdfs = sorted(ProductDir.glob("*.pdf"))
        if DirectPdfs:
            Product["modules"].append({
                "name": "general",
                "documents": [
                    {"stem": P.stem, "filename": P.name, "pdf_path": str(P)}
                    for P in DirectPdfs
                ]
            })

        if Product["modules"]:
            Products.append(Product)

    return Products


# ─────────────────────────────────────────
# Metadata Extraction (from existing doc JSONs)
# ─────────────────────────────────────────

def ExtractKeywordsFromTree(Tree: list) -> list:
    """Extract meaningful keywords from all TOC titles. No LLM needed."""
    AllWords = set()
    for Node in Tree:
        Title = Node.get("title", "")
        Words = re.sub(r'[^\w\s]', ' ', Title.lower()).split()
        for Word in Words:
            if len(Word) > 2 and Word not in STOP_WORDS:
                AllWords.add(Word)
    return sorted(AllWords)


def GetTopSections(Tree: list, MaxDepth: int = 1) -> list:
    """Get top-level section titles (depth 0 or 1) for quick overview."""
    TopSections = []
    for Node in Tree:
        Structure = Node.get("structure", "")
        Depth = Structure.count(".")
        if Depth <= MaxDepth:
            TopSections.append(Node.get("title", ""))
    return TopSections[:15]


def GetDocumentSummary(IndexData: dict) -> str:
    """Build a short doc summary from first few page summaries. No extra LLM call."""
    Pages = IndexData.get("pages", [])
    Summaries = []
    for Page in Pages[:5]:
        Summary = Page.get("summary", "").strip()
        if Summary and Summary not in ("Empty or unreadable page", "Could not summarize"):
            Summaries.append(Summary)
    return " ".join(Summaries)[:300]


# ─────────────────────────────────────────
# Layer 3: Document Index (per module)
# ─────────────────────────────────────────

def BuildDocIndex(ProductName: str, ModuleName: str, OutputFolder: str) -> dict:
    """
    Build doc_index.json for a single module.
    Reads each document JSON in the module folder and extracts lightweight metadata:
    top sections, summary, keywords, page count.
    """
    ModuleDir = Path(OutputFolder) / ProductName / ModuleName
    DocIndex = {
        "product": ProductName,
        "module": ModuleName,
        "document_count": 0,
        "documents": []
    }

    for JsonFile in sorted(ModuleDir.glob("*.json")):
        if JsonFile.name == "doc_index.json":
            continue

        try:
            with open(JsonFile, "r", encoding="utf-8") as F:
                Data = json.load(F)
        except Exception:
            continue

        Tree = Data.get("tree", [])
        DocEntry = {
            "name": JsonFile.stem,
            "filename": JsonFile.stem + ".pdf",
            "index_file": JsonFile.name,
            "total_pages": Data.get("total_pages", 0),
            "total_sections": len(Tree),
            "toc_method": Data.get("toc_method", "unknown"),
            "top_sections": GetTopSections(Tree),
            "summary": GetDocumentSummary(Data),
            "keywords": ExtractKeywordsFromTree(Tree)
        }
        DocIndex["documents"].append(DocEntry)

    DocIndex["document_count"] = len(DocIndex["documents"])

    IndexPath = ModuleDir / "doc_index.json"
    with open(IndexPath, "w", encoding="utf-8") as F:
        json.dump(DocIndex, F, indent=2, ensure_ascii=False)

    print(f"      doc_index.json: {DocIndex['document_count']} documents")
    return DocIndex


# ─────────────────────────────────────────
# Layer 2: Module Index (per product)
# ─────────────────────────────────────────

def BuildModuleIndex(ProductName: str, OutputFolder: str) -> dict:
    """
    Build module_index.json for a single product.
    Reads each module's doc_index.json and aggregates keywords/summaries upward.
    """
    ProductDir = Path(OutputFolder) / ProductName
    ModuleIndex = {
        "product": ProductName,
        "module_count": 0,
        "modules": []
    }

    for ModuleDir in sorted(ProductDir.iterdir()):
        if not ModuleDir.is_dir():
            continue

        DocIndexPath = ModuleDir / "doc_index.json"
        if not DocIndexPath.exists():
            continue

        with open(DocIndexPath, "r", encoding="utf-8") as F:
            DocIndex = json.load(F)

        # Aggregate keywords and summaries from all docs in this module
        AllKeywords = set()
        AllSummaries = []
        AllTopSections = []
        DocNames = []

        for Doc in DocIndex.get("documents", []):
            AllKeywords.update(Doc.get("keywords", []))
            Summary = Doc.get("summary", "")
            if Summary:
                AllSummaries.append(Summary[:150])
            AllTopSections.extend(Doc.get("top_sections", [])[:5])
            DocNames.append(Doc["name"])

        ModuleEntry = {
            "name": ModuleDir.name,
            "document_count": DocIndex.get("document_count", 0),
            "documents": DocNames,
            "top_sections": AllTopSections[:20],
            "summary": " ".join(AllSummaries)[:400],
            "keywords": sorted(AllKeywords)
        }
        ModuleIndex["modules"].append(ModuleEntry)

    ModuleIndex["module_count"] = len(ModuleIndex["modules"])

    IndexPath = ProductDir / "module_index.json"
    with open(IndexPath, "w", encoding="utf-8") as F:
        json.dump(ModuleIndex, F, indent=2, ensure_ascii=False)

    print(f"    module_index.json: {ModuleIndex['module_count']} modules")
    return ModuleIndex


# ─────────────────────────────────────────
# Layer 1: Product Index (root level)
# ─────────────────────────────────────────

def BuildProductIndex(OutputFolder: str) -> dict:
    """
    Build product_index.json at the root of the index folder.
    Reads each product's module_index.json and aggregates upward.
    This is the entry point for layered search.
    """
    OutputFolder = Path(OutputFolder)
    ProductIndex = {
        "created_at": datetime.now().isoformat(),
        "index_folder": str(OutputFolder.resolve()),
        "total_products": 0,
        "products": []
    }

    for ProductDir in sorted(OutputFolder.iterdir()):
        if not ProductDir.is_dir():
            continue

        ModuleIndexPath = ProductDir / "module_index.json"
        if not ModuleIndexPath.exists():
            continue

        with open(ModuleIndexPath, "r", encoding="utf-8") as F:
            ModuleIndex = json.load(F)

        # Aggregate from all modules in this product
        AllKeywords = set()
        ModuleNames = []
        AllSummaries = []
        TotalDocs = 0

        for Module in ModuleIndex.get("modules", []):
            AllKeywords.update(Module.get("keywords", []))
            ModuleNames.append(Module["name"])
            Summary = Module.get("summary", "")
            if Summary:
                AllSummaries.append(Summary[:150])
            TotalDocs += Module.get("document_count", 0)

        ProductEntry = {
            "name": ProductDir.name,
            "module_count": len(ModuleNames),
            "modules": ModuleNames,
            "total_documents": TotalDocs,
            "summary": " ".join(AllSummaries)[:500],
            "keywords": sorted(AllKeywords)
        }
        ProductIndex["products"].append(ProductEntry)

    ProductIndex["total_products"] = len(ProductIndex["products"])

    IndexPath = OutputFolder / "product_index.json"
    with open(IndexPath, "w", encoding="utf-8") as F:
        json.dump(ProductIndex, F, indent=2, ensure_ascii=False)

    print(f"  product_index.json: {ProductIndex['total_products']} products")
    return ProductIndex


# ─────────────────────────────────────────
# Main Batch Indexer
# ─────────────────────────────────────────

def BatchIndex(RootPath: str, OutputFolder: str, Model: str, Host: str,
               SkipExisting: bool = True):
    """
    Full hierarchical indexing pipeline:
      1. Scan folder tree to detect Product/Module/Document hierarchy
      2. Index each PDF (reuses existing index_pdf, skips if JSON exists)
      3. Build doc_index.json per module (Layer 3)
      4. Build module_index.json per product (Layer 2)
      5. Build product_index.json at root (Layer 1)

    All layer indexes are built purely from existing document JSONs — no extra LLM calls.
    """
    RootPath = Path(RootPath)
    OutputFolder = Path(OutputFolder)
    OutputFolder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Hierarchical Batch PDF Indexer")
    print(f"   Root:   {RootPath}")
    print(f"   Output: {OutputFolder}")
    print(f"   Model:  {Model}")
    print(f"   Skip existing: {SkipExisting}")
    print(f"{'='*60}\n")

    # ── Step 1: Scan hierarchy ─────────────────────
    print("Step 1: Scanning folder hierarchy...")
    Products = ScanHierarchy(str(RootPath))

    if not Products:
        print("  No products found! Expected structure: Root/Product/Module/*.pdf")
        sys.exit(1)

    TotalDocs = 0
    for Product in Products:
        ModuleCount = len(Product["modules"])
        DocCount = sum(len(M["documents"]) for M in Product["modules"])
        TotalDocs += DocCount
        print(f"  {Product['name']}: {ModuleCount} modules, {DocCount} docs")
        for Module in Product["modules"]:
            print(f"    {Module['name']}: {len(Module['documents'])} docs")

    # ── Step 2: Index each PDF ─────────────────────
    print(f"\nStep 2: Indexing {TotalDocs} PDFs...")
    StartTime = time.time()
    Indexed = 0
    Skipped = 0
    Failed = 0
    DocNum = 0

    for Product in Products:
        for Module in Product["modules"]:
            # Create output directory for this module
            ModuleOutputDir = OutputFolder / Product["name"] / Module["name"]
            ModuleOutputDir.mkdir(parents=True, exist_ok=True)

            for Doc in Module["documents"]:
                DocNum += 1
                PdfPath = Doc["pdf_path"]
                Stem = Doc["stem"]
                JsonOutputPath = ModuleOutputDir / f"{Stem}.json"

                # Resume support: skip if already indexed
                if SkipExisting and JsonOutputPath.exists():
                    print(f"  [{DocNum}/{TotalDocs}] SKIP: "
                          f"{Product['name']}/{Module['name']}/{Stem}")
                    Skipped += 1
                    continue

                print(f"\n  [{DocNum}/{TotalDocs}] Indexing: "
                      f"{Product['name']}/{Module['name']}/{Stem}")
                print(f"  {'─'*50}")

                try:
                    index_pdf(PdfPath, Model, Host, str(JsonOutputPath))
                    Indexed += 1
                except SystemExit:
                    print(f"  [FAILED] PDF not found: {PdfPath}")
                    Failed += 1
                except Exception as E:
                    print(f"  [FAILED] {E}")
                    Failed += 1

    ElapsedMinutes = (time.time() - StartTime) / 60
    print(f"\n  Indexing done in {ElapsedMinutes:.1f} minutes")
    print(f"    New: {Indexed}  Skipped: {Skipped}  Failed: {Failed}")

    # ── Step 3: Build layered indexes ──────────────
    print(f"\nStep 3: Building layered indexes...")

    # Layer 3: doc_index.json per module
    print("\n  Building Layer 3 (doc indexes)...")
    for Product in Products:
        for Module in Product["modules"]:
            print(f"    {Product['name']}/{Module['name']}:")
            BuildDocIndex(Product["name"], Module["name"], str(OutputFolder))

    # Layer 2: module_index.json per product
    print("\n  Building Layer 2 (module indexes)...")
    for Product in Products:
        print(f"  {Product['name']}:")
        BuildModuleIndex(Product["name"], str(OutputFolder))

    # Layer 1: product_index.json at root
    print("\n  Building Layer 1 (product index)...")
    ProductIndex = BuildProductIndex(str(OutputFolder))

    print(f"\n{'='*60}")
    print(f"  Hierarchical indexing complete!")
    print(f"   Products: {ProductIndex['total_products']}")
    for P in ProductIndex["products"]:
        print(f"     {P['name']}: {P['module_count']} modules, "
              f"{P['total_documents']} docs")
    print(f"   Index at: {OutputFolder / 'product_index.json'}")
    print(f"{'='*60}\n")

    return ProductIndex


if __name__ == "__main__":
    Parser = argparse.ArgumentParser(
        description="Hierarchical batch indexer: Root/Product/Module/*.pdf"
    )
    Parser.add_argument(
        "--input", required=True,
        help="Root folder (e.g., FlexCube/) containing Product/Module/Doc.pdf"
    )
    Parser.add_argument(
        "--output", default="index",
        help="Output folder for all indexes (default: ./index)"
    )
    Parser.add_argument(
        "--model", default=INDEXER_MODEL,
        help="Model name"
    )
    Parser.add_argument(
        "--host", default=INDEXER_HOST,
        help="Server URL or 'groq'"
    )
    Parser.add_argument(
        "--reindex", action="store_true",
        help="Re-index all PDFs even if JSON already exists"
    )
    Args = Parser.parse_args()

    BatchIndex(
        Args.input, Args.output,
        Args.model, Args.host,
        SkipExisting=not Args.reindex
    )
