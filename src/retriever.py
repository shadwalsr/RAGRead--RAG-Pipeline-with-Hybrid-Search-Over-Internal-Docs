"""
Hybrid Retrieval Engine (Phase 2)

Fuses Dense (Vector) and Sparse (BM25) search results using 
Reciprocal Rank Fusion (RRF) for high-precision retrieval.
"""

from typing import List, Dict, Any
from storage import HybridStore

class HybridRetriever:
    """
    Combines ChromaDB vector search with BM25 keyword search using RRF.
    """
    
    def __init__(self, store: HybridStore = None):
        self.store = store or HybridStore()
        
    def _reciprocal_rank_fusion(
        self, 
        vector_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]],
        k: int = 60,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Fuses two ranked lists using the RRF algorithm.
        Score = weight * (1 / (k + rank))
        
        alpha: Weight given to dense (vector) search. 
               (1 - alpha) is given to sparse (BM25) search.
               Example: alpha=0.7 means 70% dense, 30% sparse.
        """
        fused_scores = {}
        chunks_by_id = {}
        
        # Helper to process a ranked list
        def add_to_fusion(results: List[Dict[str, Any]], weight: float):
            for rank, result in enumerate(results):
                doc_id = result["id"]
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                    chunks_by_id[doc_id] = result
                
                # RRF Formula with custom weighting
                fused_scores[doc_id] += weight * (1.0 / (k + rank + 1))
                
        # Process both lists with adjustable weighting
        add_to_fusion(vector_results, weight=alpha)
        add_to_fusion(bm25_results, weight=(1.0 - alpha))
        
        # Sort by fused score descending
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        # Build final ranked list
        final_results = []
        for doc_id in sorted_ids:
            chunk = chunks_by_id[doc_id]
            chunk["rrf_score"] = round(fused_scores[doc_id], 4)
            final_results.append(chunk)
            
        return final_results

    def _llm_rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Cross-encoder style reranking using Gemini Flash.
        Sends the candidate chunks to the LLM and asks it to identify
        the most relevant ones for the specific query.
        """
        import os
        import json
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("WARNING: No API key found for reranking. Returning RRF results.")
            return candidates[:top_k]

        client = genai.Client(api_key=api_key)

        chunks_text = ""
        for chunk in candidates:
            chunks_text += f"--- Chunk ID: {chunk['id']} ---\n{chunk['content']}\n\n"

        prompt = f'''You are an expert relevance ranking engine.
Query: "{query}"

Evaluate the following document chunks and rank them by how relevant they are to answering the query.
Return the result as a JSON array of chunk IDs, ordered from most relevant to least relevant.
Max {top_k} chunk IDs. Only include IDs of chunks that are actually relevant to the query.
If none are relevant, return an empty array [].

Chunks:
{chunks_text}
'''
        try:
            response = client.models.generate_content(
                model='gemini-flash-latest',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                )
            )
            
            ranked_ids = json.loads(response.text)
            
            # Rebuild the final list in the order specified by the LLM
            chunk_map = {c['id']: c for c in candidates}
            final_reranked = []
            
            for doc_id in ranked_ids:
                if doc_id in chunk_map:
                    # Give it a special flag so we know it was reranked
                    chunk = chunk_map[doc_id]
                    chunk['reranked'] = True
                    final_reranked.append(chunk)
                    
            return final_reranked
            
        except Exception as e:
            print(f"WARNING: LLM Reranking failed ({e}). Falling back to RRF.")
            return candidates[:top_k]

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5, use_reranker: bool = False, strategy: str = None) -> List[Dict[str, Any]]:
        """
        Executes a hybrid search for the given query.
        
        alpha: Weight given to vector search (0.0 to 1.0). Default 0.5 (equal weight).
        use_reranker: If True, sends the fused top 20 results to an LLM to re-evaluate and filter.
        strategy: If provided, only returns chunks from this specific strategy (fixed, structural, semantic).
        """
        # Step 1: Fetch candidates (k=10 as required for production)
        where_filter = {"chunking_strategy": strategy} if strategy else None
        
        vector_candidates = self.store.vector_search(query, n_results=10, where=where_filter)
        bm25_candidates = self.store.bm25_search(query, n_results=10, strategy=strategy)
        
        # Step 2: Fuse with RRF and apply weights
        fused_results = self._reciprocal_rank_fusion(
            vector_candidates, 
            bm25_candidates, 
            alpha=alpha
        )
        
        # Step 3: LLM Reranking (Cross-Encoder Pass)
        if use_reranker:
            # Pass the fused top 20 to the reranker
            return self._llm_rerank(query, fused_results, top_k)
            
        # Or Return top K from RRF directly
        return fused_results[:top_k]

# For testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    retriever = HybridRetriever()
    query = "skills and management"
    
    print(f"--- HYBRID SEARCH TEST (RRF Only): '{query}' ---\n")
    results = retriever.retrieve(query, top_k=3, use_reranker=False)
    for i, r in enumerate(results):
        print(f"[{i+1}] RRF Score: {r['rrf_score']} | ID: {r['id']}")
        print(f"    Content: {r['content'][:150]}...\n")
        
    print(f"--- HYBRID SEARCH TEST (With LLM Reranker): '{query}' ---\n")
    reranked_results = retriever.retrieve(query, top_k=3, use_reranker=True)
    for i, r in enumerate(reranked_results):
        print(f"[{i+1}] RRF Score: {r.get('rrf_score')} | ID: {r['id']} (Reranked: {r.get('reranked', False)})")
        print(f"    Content: {r['content'][:150]}...\n")
