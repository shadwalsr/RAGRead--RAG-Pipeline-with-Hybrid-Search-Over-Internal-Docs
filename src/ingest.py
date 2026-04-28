"""
Ingestion Script — Loads processed chunks and ingests them into the
HybridStore (ChromaDB + BM25) with deduplication.

Usage:
    python src/ingest.py                          # ingest all semantic chunks
    python src/ingest.py --strategy fixed_size    # ingest fixed-size chunks
    python src/ingest.py --strategy all           # ingest all strategies
"""

import os
import sys
import json
import glob
import argparse

from dotenv import load_dotenv

load_dotenv()

from storage import HybridStore

PROCESSED_DIR = "data/processed"


def load_processed_chunks(strategy: str) -> list:
    """
    Read chunk .txt / .json pairs from data/processed/ for a given strategy.
    """
    pattern = os.path.join(PROCESSED_DIR, f"*_{strategy}_chunk_*.txt")
    txt_files = sorted(glob.glob(pattern))

    if not txt_files:
        print(f"No chunks found for strategy '{strategy}' in {PROCESSED_DIR}")
        return []

    chunks = []
    for txt_path in txt_files:
        json_path = txt_path.replace(".txt", ".json")

        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata = {}
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        chunks.append({"content": content, "metadata": metadata})

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest chunks into HybridStore.")
    parser.add_argument(
        "--strategy",
        choices=["fixed_size", "structure_aware", "semantic", "all"],
        default="semantic",
        help="Which chunking strategy's output to ingest (default: semantic)",
    )
    args = parser.parse_args()

    strategies = (
        ["fixed_size", "structure_aware", "semantic"]
        if args.strategy == "all"
        else [args.strategy]
    )

    store = HybridStore()

    for strat in strategies:
        print(f"\n{'='*60}")
        print(f"Ingesting: {strat}")
        print(f"{'='*60}")

        chunks = load_processed_chunks(strat)
        if not chunks:
            continue

        print(f"Found {len(chunks)} chunks to ingest.\n")
        stats = store.ingest_chunks(chunks)
        print(f"\nStats: {json.dumps(stats, indent=2)}")

    # Final summary
    print(f"\n{'='*60}")
    print("Final Store Status")
    print(f"{'='*60}")
    final = store.get_stats()
    for k, v in final.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
