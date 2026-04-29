"""
Storage & Hybrid Retrieval Module  (Step 5 + 6)

Provides:
1. Deduplication Filter  – skips near-duplicate chunks (cosine sim > 0.95)
2. Vector Storage        – ChromaDB with Gemini embeddings + rich metadata
3. BM25 Keyword Index    – rank_bm25 kept in perfect sync with the vector store
4. Smart Tokenizer       – collapses spaced PDF text before BM25 indexing
   (e.g. 'W h y s c h o o l' → 'whyschool')
"""

import os
import json
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Gemini Embedding Helper
# ---------------------------------------------------------------------------

def get_embeddings(
    texts: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gemini-embedding-001",
) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using the Gemini embedding model.
    """
    from google import genai

    api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY to generate embeddings."
        )

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=model_name,
        contents=texts,
    )
    return [e.values for e in result.embeddings]


# ---------------------------------------------------------------------------
# Cosine Similarity (for deduplication check)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


# ---------------------------------------------------------------------------
# Hybrid Store
# ---------------------------------------------------------------------------

class HybridStore:
    """
    Production-grade storage layer that keeps a ChromaDB vector store and
    a BM25 keyword index in perfect sync, with deduplication.

    Usage:
        store = HybridStore()
        stats = store.ingest_chunks(chunks)   # chunks from chunker.py
        print(stats)
    """

    DEDUP_THRESHOLD = 0.95  # cosine similarity above this → near-duplicate

    def __init__(
        self,
        persist_dir: str = "data/vectorstore",
        collection_name: str = "rag_documents",
        bm25_path: str = "data/bm25_index.pkl",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.bm25_path = bm25_path

        os.makedirs(persist_dir, exist_ok=True)

        # ---- ChromaDB ----
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # ---- BM25 (loaded from disk if available) ----
        self._bm25_corpus: List[List[str]] = []  # tokenised docs
        self._bm25_ids: List[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._load_bm25()

    # ------------------------------------------------------------------
    # BM25 persistence
    # ------------------------------------------------------------------

    def _load_bm25(self):
        """Restore BM25 index from disk."""
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                data = pickle.load(f)
            self._bm25_corpus = data["corpus"]
            self._bm25_ids = data["ids"]
            if self._bm25_corpus:
                self._bm25 = BM25Okapi(self._bm25_corpus)
            print(f"[BM25] Loaded existing index with {len(self._bm25_ids)} documents.")
        else:
            print("[BM25] No existing index found. Starting fresh.")

    def _save_bm25(self):
        """Persist BM25 index to disk."""
        os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump({"corpus": self._bm25_corpus, "ids": self._bm25_ids}, f)

    def _rebuild_bm25(self):
        """Rebuild BM25Okapi from current corpus."""
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)
        else:
            self._bm25 = None

    # Characters that should break a spaced-letter run instead of being merged
    _BREAK_CHARS = set('/-.,;:!?@#$%^&*()[]{}|<>=+\\_~`\'"\\\\')

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Smart tokeniser for BM25.

        Problem: PDFs exported from design tools store text with spaces between
        every character:  'W h y s c h o o l A c a d e m y 2 0 2 6 - P r e s e n t'

        Strategy:
          1. Split on whitespace to get raw single-char tokens (preserving case).
          2. Walk tokens accumulating letters/digits into a 'run'.
             - Punctuation chars flush the run and are discarded.
             - Multi-char real words (len > 1) flush the run and are added as-is.
          3. When flushing a run, it may form a merged word like 'WhyschoolAcademy'.
             Use regex to split CamelCase and PascalCase boundaries before lowercasing.

        Examples:
            'W h y s c h o o l A c a d e m y'  -> ['whyschool', 'academy']
            'S K I L L S Leadership'            -> ['skills', 'leadership']
            'c o m / i n / s h a d w a l'      -> ['com', 'in', 'shadwal']
            '2 0 2 6 - P r e s e n t'          -> ['2026', 'present']
            'Python developer'                  -> ['python', 'developer']
        """
        import re
        raw = text.split()          # preserve case for boundary detection
        result: List[str] = []
        run: List[str] = []

        def split_camel_case(word: str) -> List[str]:
            # Split lower->Upper (e.g. WhyschoolAcademy -> Whyschool Academy)
            s1 = re.sub(r'([a-z])([A-Z])', r'\1 \2', word)
            # Split UPPER->UpperLower (e.g. SKILLSLeadership -> SKILLS Leadership)
            s2 = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', s1)
            return s2.lower().split()

        def flush_run():
            if run:
                merged = ''.join(run)
                result.extend(split_camel_case(merged))
                run.clear()

        for token in raw:
            if len(token) == 1 and token in HybridStore._BREAK_CHARS:
                flush_run()                           # punctuation → word break
            elif len(token) == 1:
                run.append(token)                     # accumulate
            else:
                flush_run()                           # real multi-char word
                result.extend(split_camel_case(token))

        flush_run()
        return [t for t in result if len(t) > 0]



    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _is_duplicate(self, embedding: List[float], threshold: float = None) -> bool:
        """
        Check if a chunk's embedding is a near-duplicate of something
        already in the vector store.
        """
        threshold = threshold or self.DEDUP_THRESHOLD

        # If the collection is empty, nothing to compare against
        if self._collection.count() == 0:
            return False

        # Query for the single most similar existing chunk
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
        )

        if results and results["distances"] and results["distances"][0]:
            # ChromaDB cosine distance = 1 - cosine_similarity
            distance = results["distances"][0][0]
            similarity = 1.0 - distance
            if similarity >= threshold:
                return True

        return False

    # ------------------------------------------------------------------
    # Chunk ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(content: str, metadata: Dict[str, Any]) -> str:
        """Deterministic ID from content + key metadata."""
        key = f"{metadata.get('source', '')}:{metadata.get('chunk_index', '')}:{metadata.get('chunking_strategy', '')}"
        return hashlib.sha256((key + content[:100]).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 20,
    ) -> Dict[str, int]:
        """
        Ingest a list of chunks (from chunker.py output or processed files).

        Each chunk is a dict with keys: "content", "metadata".

        Returns stats: {"total", "added", "duplicates_skipped"}.
        """
        stats = {"total": len(chunks), "added": 0, "duplicates_skipped": 0}

        # Process in batches to be kind to the embedding API
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            texts = [c["content"] for c in batch]

            # Skip empty chunks
            texts_clean = [t if t.strip() else " " for t in texts]

            print(f"  Embedding batch {batch_start // batch_size + 1} "
                  f"({len(texts_clean)} chunks)...")
            embeddings = get_embeddings(texts_clean)

            for chunk, embedding in zip(batch, embeddings):
                content = chunk["content"]
                metadata = chunk.get("metadata", {})
                doc_id = self._make_id(content, metadata)

                # ---- Deduplication check ----
                if self._is_duplicate(embedding):
                    stats["duplicates_skipped"] += 1
                    print(f"    [SKIP] Duplicate detected for chunk {metadata.get('chunk_index', '?')}")
                    continue

                # ---- Add to ChromaDB ----
                # ChromaDB metadata values must be str, int, float, or bool
                safe_meta = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        safe_meta[k] = v
                    elif isinstance(v, list):
                        safe_meta[k] = json.dumps(v)
                    elif v is None:
                        safe_meta[k] = ""
                    else:
                        safe_meta[k] = str(v)

                self._collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[safe_meta],
                )

                # ---- Add to BM25 (synced) ----
                tokens = self._tokenize(content)
                self._bm25_corpus.append(tokens)
                self._bm25_ids.append(doc_id)

                stats["added"] += 1

        # Rebuild BM25 with the complete corpus and persist
        self._rebuild_bm25()
        self._save_bm25()

        print(f"\n  Ingestion complete: "
              f"{stats['added']} added, "
              f"{stats['duplicates_skipped']} duplicates skipped "
              f"(out of {stats['total']} total).")
        print(f"  ChromaDB collection size: {self._collection.count()}")
        print(f"  BM25 index size: {len(self._bm25_ids)}")

        return stats

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def vector_search(self, query: str, n_results: int = 10, where: Dict = None) -> List[Dict[str, Any]]:
        """
        Dense semantic search via ChromaDB.
        where: Optional metadata filter dict (e.g. {"chunking_strategy": "fixed"})
        """
        # Cap n_results to the collection size to avoid ChromaDB errors
        n_results = min(n_results, self._collection.count())
        if n_results == 0:
            return []

        query_emb = get_embeddings([query])
        results = self._collection.query(
            query_embeddings=query_emb,
            n_results=n_results,
            where=where
        )
        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": round(1.0 - results["distances"][0][i], 4),
            })
        return out

    def bm25_search(self, query: str, n_results: int = 5, strategy: str = None) -> List[Dict[str, Any]]:
        """
        Keyword search via BM25.
        strategy: If provided, only returns chunks from this strategy.
        """
        if not self._bm25:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        
        # Sort all indices by score descending
        top_indices = np.argsort(scores)[::-1]

        out = []
        for idx in top_indices:
            if len(out) >= n_results:
                break
                
            if scores[idx] > 0:
                doc_id = self._bm25_ids[idx]
                
                # Fetch metadata to check strategy
                result = self._collection.get(ids=[doc_id])
                if not result or not result["documents"]:
                    continue
                    
                meta = result["metadatas"][0]
                if strategy and meta.get("chunking_strategy") != strategy:
                    continue

                out.append({
                    "id": doc_id,
                    "content": result["documents"][0],
                    "metadata": meta,
                    "bm25_score": round(float(scores[idx]), 4),
                })
        return out

    def get_stats(self) -> Dict[str, Any]:
        """Return current store statistics."""
        return {
            "chromadb_count": self._collection.count(),
            "bm25_count": len(self._bm25_ids),
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
        }
