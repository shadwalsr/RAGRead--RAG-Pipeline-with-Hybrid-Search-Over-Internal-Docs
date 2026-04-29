import os
from dotenv import load_dotenv
from retriever import HybridRetriever

# quick script to test retrieval without the whole AI generation part
def quick_test():
    load_dotenv()
    r = HybridRetriever()
    
    q = "Shadwal Singh experience"
    print(f"Searching for: {q}\n")
    
    # testing hybrid search with reranker on
    hits = r.retrieve(q, top_k=3, use_reranker=True)
    
    for i, h in enumerate(hits):
        print(f"[{i+1}] {h['metadata'].get('source')} (Score: {h.get('rrf_score')})")
        print(f"   {h['content'][:150]}...\n")

if __name__ == "__main__":
    quick_test()
