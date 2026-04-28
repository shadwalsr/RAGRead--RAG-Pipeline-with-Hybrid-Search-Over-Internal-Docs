"""
One-off script: rebuild the BM25 index using the new smart tokenizer.
Run this once after upgrading storage.py's _tokenize method.
"""
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, 'src')
from storage import HybridStore

store = HybridStore()

# Pull all existing docs from ChromaDB
result = store._collection.get(include=['documents', 'metadatas'])
ids  = result['ids']
docs = result['documents']

print(f"Rebuilding BM25 index for {len(ids)} documents with new smart tokenizer...")

# Demo: show what the new tokenizer does to spaced PDF text
sample = "W h y s c h o o l A c a d e m y"
print(f"  Old naive tokenization: {sample.lower().split()[:4]}...")
print(f"  New smart tokenization: {store._tokenize(sample)}")
print()

# Rebuild corpus with new tokenizer
store._bm25_corpus = [store._tokenize(d) for d in docs]
store._bm25_ids    = list(ids)
store._rebuild_bm25()
store._save_bm25()

print(f"Done. Index rebuilt with {len(ids)} documents.")
print()
print("--- Spot-check: BM25 search for 'whyschool' ---")
results = store.bm25_search('whyschool', n_results=3)
if results:
    for r in results:
        print(f"  Score: {r['bm25_score']:.4f} | {r['content'][:120]}")
else:
    print("  No results — tokenizer may need further tuning.")

print()
print("--- Spot-check: BM25 search for 'skills' ---")
results2 = store.bm25_search('skills', n_results=3)
if results2:
    for r in results2:
        print(f"  Score: {r['bm25_score']:.4f} | {r['content'][:120]}")
