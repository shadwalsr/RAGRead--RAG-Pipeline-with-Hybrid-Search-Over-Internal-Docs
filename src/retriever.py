import os
import json
from typing import List, Dict, Any
from google import genai
from google.genai import types
from storage import HybridStore
from rate_limiter import call_with_retry

class HybridRetriever:
    def __init__(self, db: HybridStore = None):
        self.db = db or HybridStore()

    def merge_results(self, dense_hits, sparse_hits, k=60, alpha=0.5):
        # basic reciprocal rank fusion (RRF)
        # alpha=0.5 means equal weight for both search types
        scores = {}
        lookup = {} # for quick access to the actual content

        def score_list(hits, weight):
            for rank, hit in enumerate(hits):
                hid = hit["id"]
                if hid not in scores:
                    scores[hid] = 0.0
                    lookup[hid] = hit
                # RRF math: higher rank = higher score
                # using k=60 to dampen high ranks
                scores[hid] += weight * (1.0 / (k + rank + 1))

        score_list(dense_hits, weight=alpha)
        score_list(sparse_hits, weight=(1.0 - alpha))

        # sort by the new combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        final = []
        for hid in sorted_ids:
            chunk = lookup[hid]
            chunk["rrf_score"] = round(scores[hid], 4)
            final.append(chunk)
            
        return final

    def smart_rerank(self, query, candidates, top_n):
        # asks an LLM to pick the best chunks from a candidate list
        # basically a second pass to fix any ranking weirdness
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            # if we don't have a key, just return the raw candidates
            return candidates[:top_n]

        client = genai.Client(api_key=key)

        # build a context string for the LLM to judge
        text_to_judge = ""
        for c in candidates:
            text_to_judge += f"--- ID: {c['id']} ---\n{c['content']}\n\n"

        prompt = f"""You are a judge. Which of these chunks actually help answer this query?
Query: "{query}"

Return a JSON list of chunk IDs in order of relevance. Max {top_n} IDs.
Chunks:
{text_to_judge}
"""
        try:
            resp = call_with_retry(
                client.models.generate_content,
                model='gemini-flash-latest',
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type='application/json')
            )
            
            best_ids = json.loads(resp.text)
            
            # map the IDs back to the actual data
            chunk_map = {c['id']: c for c in candidates}
            reranked = []
            for bid in best_ids:
                if bid in chunk_map:
                    c = chunk_map[bid]
                    c['reranked'] = True
                    reranked.append(c)
            return reranked
            
        except Exception as e:
            # if the AI reranker fails, don't crash, just use the RRF results
            print(f"Reranking went wrong: {e}")
            return candidates[:top_n]

    def retrieve(self, query, top_k=5, alpha=0.5, use_reranker=False, strategy=None):
        # step 1: get candidates from both worlds
        # we grab 10 from each to have a good pool for fusion
        f = {"chunking_strategy": strategy} if strategy else None
        
        dense_results = self.db.dense_search(query, n_results=10, where=f)
        sparse_results = self.db.sparse_search(query, n_results=10, strategy=strategy)
        
        # step 2: mash them together
        merged = self.merge_results(dense_results, sparse_results, alpha=alpha)
        
        # step 3: optional slow-but-smart reranking
        if use_reranker:
            return self.smart_rerank(query, merged, top_k)
            
        return merged[:top_k]

# test block
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    r = HybridRetriever()
    q = "Shadwal management skills"
    
    print(f"Testing search for: {q}")
    res = r.retrieve(q, top_k=2)
    for i, chunk in enumerate(res):
        print(f"{i+1}. Score: {chunk.get('rrf_score')} | {chunk['content'][:100]}...")
