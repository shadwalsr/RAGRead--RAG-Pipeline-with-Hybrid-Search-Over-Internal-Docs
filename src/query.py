"""
Query Script — Test your Hybrid Search (Vector + BM25) from the CLI.

Usage:
    python src/query.py "What are the project management skills?"
    python src/query.py --mode bm25 "Agile"
"""

import argparse
import json
from dotenv import load_dotenv
from storage import HybridStore

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Query the RAG HybridStore.")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument(
        "--mode", 
        choices=["vector", "bm25", "both"], 
        default="both",
        help="Search mode (default: both)"
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of results")
    
    args = parser.parse_args()
    store = HybridStore()
    
    if args.mode in ["vector", "both"]:
        print(f"\n--- SEMANTIC (VECTOR) SEARCH: '{args.query}' ---")
        results = store.vector_search(args.query, n_results=args.top_k)
        for i, r in enumerate(results):
            print(f"[{i+1}] Distance: {r['distance']:.4f} | Strategy: {r['metadata'].get('chunking_strategy')}")
            print(f"    Content: {r['content'][:200]}...")
            
    if args.mode in ["bm25", "both"]:
        print(f"\n--- KEYWORD (BM25) SEARCH: '{args.query}' ---")
        results = store.bm25_search(args.query, n_results=args.top_k)
        if not results:
            print("    No exact keyword matches found.")
        for i, r in enumerate(results):
            print(f"[{i+1}] Score: {r['bm25_score']:.4f}")
            print(f"    Content: {r['content'][:200]}...")

if __name__ == "__main__":
    main()
