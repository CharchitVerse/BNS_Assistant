#!/usr/bin/env python3
"""
Run the BNS ingestion pipeline.

Usage:
    python scripts/ingest.py --pdf-path ./data/bns_english.pdf

This script:
1. Extracts text from the BNS PDF
2. Parses into structured sections (1-358)
3. Chunks with section-aware splitting
4. Generates embeddings locally (all-MiniLM-L6-v2)
5. Stores in ChromaDB (vectors) + PostgreSQL (metadata)

Run on your MacBook (24GB RAM). Takes ~2-3 minutes.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.pipeline import run_ingestion_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="BNS PDF Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py --pdf-path ./data/bns_english.pdf
  python scripts/ingest.py --pdf-path ~/Downloads/250883_english_01042024.pdf
        """,
    )
    parser.add_argument(
        "--pdf-path",
        required=True,
        help="Path to the BNS PDF file",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        print(f"   Download from: https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf")
        sys.exit(1)

    print(f"📄 Starting ingestion: {pdf_path}")
    print(f"   This will take ~2-3 minutes on first run (model download + embedding)")
    print()

    stats = run_ingestion_pipeline(str(pdf_path))

    print()
    print("=" * 60)
    print("✅ INGESTION COMPLETE")
    print("=" * 60)
    print(f"  📄 PDF characters: {stats['pdf_chars']:,}")
    print(f"  📑 Sections parsed: {stats['sections_parsed']}")
    print(f"  🧩 Chunks created: {stats['chunks_created']}")
    print(f"  🧠 Embedding model: {stats['embedding_model']}")
    print(f"  💾 ChromaDB path: {stats['chroma_dir']}")
    print()
    print("Next steps:")
    print("  1. Start Postgres:  docker compose up postgres -d")
    print("  2. Start backend:   uvicorn app.api.main:app --reload")
    print("  3. Start frontend:  streamlit run frontend/app.py")


if __name__ == "__main__":
    main()
