import os
import pickle
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from google import genai
from typing import List, Dict, Any

class HybridStore:
    def __init__(self, db_path="data/vectorstore", bm25_path="data/bm25_index.pkl"):
        # setup chroma - it's where we keep the actual embeddings
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # for bm25, we just load the whole thing from a pickle file
        self.bm25_path = bm25_path
        self.corpus = [] # the raw text chunks
        self.bm25 = None
        self.doc_ids = [] # mapped to the corpus index
        
        self._load_bm25()

    def _load_bm25(self):
        # if the pickle exists, load it. otherwise we'll start fresh
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                data = pickle.load(f)
                self.corpus = data["corpus"]
                self.doc_ids = data["doc_ids"]
                # rebuild the index from the loaded corpus
                self.bm25 = BM25Okapi(self.corpus)

    def _save_bm25(self):
        # persistent storage for the keyword index
        with open(self.bm25_path, "wb") as f:
            pickle.dump({"corpus": self.corpus, "doc_ids": self.doc_ids}, f)

    def dense_search(self, query, n_results=5, where=None):
        # this handles the "meaning" based search
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        emb_resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query
        )
        q_emb = emb_resp.embeddings[0].values

        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            where=where
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "dist": results["distances"][0][i]
            })
        return formatted

    def sparse_search(self, query, n_results=5, strategy=None):
        # keywords search - good for specific terms like "Python" or names
        if not self.bm25:
            return []

        q_tokens = query.lower().split()
        scores = self.bm25.get_scores(q_tokens)
        
        # get the top indices
        top_idx = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_idx:
            if len(results) >= n_results:
                break
                
            chunk_text = " ".join(self.corpus[idx])
            chunk_id = self.doc_ids[idx]
            
            # manual metadata pull from Chroma because BM25 doesn't store it
            # this is a bit slow, but keeps the BM25 index small
            meta_res = self._collection.get(ids=[chunk_id])
            metadata = meta_res["metadatas"][0]
            
            # filter by strategy if requested
            if strategy and metadata.get("chunking_strategy") != strategy:
                continue
                
            results.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": metadata,
                "bm25_score": scores[idx]
            })
        return results

    def ingest_chunks(self, chunks: List[Dict[str, Any]]):
        # batch add to both vector and keyword search
        added_count = 0
        skip_count = 0
        
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        
        # dedup check - don't want to double index the same text
        for chunk in chunks:
            # check for existing
            # Note: simplified dedup for now, just checking ID
            existing = self._collection.get(ids=[chunk["id"]])
            if existing["ids"]:
                skip_count += 1
                continue
                
            # embed and add to chroma
            emb_resp = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk["content"]
            )
            vec = emb_resp.embeddings[0].values
            
            self._collection.add(
                ids=[chunk["id"]],
                embeddings=[vec],
                documents=[chunk["content"]],
                metadatas=[chunk["metadata"]]
            )
            
            # add to bm25
            self.corpus.append(chunk["content"].lower().split())
            self.doc_ids.append(chunk["id"])
            added_count += 1
            
        # refresh bm25 and save
        if added_count > 0:
            self.bm25 = BM25Okapi(self.corpus)
            self._save_bm25()
            
        return {"total": len(chunks), "added": added_count, "duplicates_skipped": skip_count}
