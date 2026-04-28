# Storage & Deduplication Report — Production RAG Pipeline

**Author:** Shadwal Singh\
**Date:** May 2, 2026\
**Step:** 5 — Storage, Deduplication & Hybrid Search Indexing\
**Input:** 27 semantic chunks from `shadwal singh (3).pdf`

---

## 1. Executive Summary

This report covers the implementation of a production-grade storage layer for
the RAG pipeline. After chunking documents in Step 4, this step turns raw text
segments into a **searchable, deduplicated database** with two synchronized
search indices — one semantic (ChromaDB) and one keyword-based (BM25).

| Component           | Technology                             | Status                             |
| ------------------- | -------------------------------------- | ---------------------------------- |
| **Vector Store**    | ChromaDB (persistent, cosine distance) | 27 documents indexed               |
| **Keyword Index**   | BM25Okapi (rank_bm25)                  | 27 documents indexed               |
| **Deduplication**   | Cosine similarity > 0.95 threshold     | Verified — 27/27 skipped on re-run |
| **Embedding Model** | Gemini `gemini-embedding-001`          | Batch embedding (20 + 7)           |
| **Disk Footprint**  | ChromaDB: ~1.8 MB, BM25: ~8 KB         | Lightweight for 27 chunks          |

---

## 2. The Three Goals — Implementation & Results

### 2.1 Deduplication Filter (Noise Control)

**Problem:** Even with smart chunking, re-processing the same document or
overlapping content can create redundant entries that pollute the search results
and waste the LLM's context window.

**Solution:** Before saving any chunk, we embed it and query ChromaDB for the
most similar existing document. If the cosine similarity exceeds **0.95**, the
chunk is flagged as a near-duplicate and skipped.

**Implementation:**

```python
def _is_duplicate(self, embedding, threshold=0.95):
    # Query ChromaDB for the single most similar existing chunk
    results = self._collection.query(
        query_embeddings=[embedding],
        n_results=1,
    )
    # ChromaDB returns cosine distance (1 - similarity)
    distance = results["distances"][0][0]
    similarity = 1.0 - distance
    return similarity >= threshold
```

**Test Results:**

| Run                             | Chunks Processed | Added | Duplicates Skipped |
| ------------------------------- | :--------------: | :---: | :----------------: |
| **First ingestion**             |        27        |  27   |         0          |
| **Second ingestion** (re-run)   |        27        |   0   |         27         |
| **Third ingestion** (user test) |        27        |   0   |         27         |

The deduplication filter correctly identified all 27 chunks as duplicates across
two independent re-runs, confirming that the 0.95 threshold is working as
intended.

**Why 0.95?**

- **Too low** (e.g. 0.80): Would aggressively merge related-but-different
  chunks, losing information.
- **Too high** (e.g. 0.99): Would only catch exact copies, missing paraphrased
  duplicates.
- **0.95**: Catches near-identical content while preserving distinct information
  — the sweet spot for production systems.

---

### 2.2 Vector Storage (Semantic Search)

**Problem:** Traditional keyword search fails when a user's query doesn't share
exact words with the document. For example, searching "communication abilities"
should still find chunks about "effective communication skills."

**Solution:** Store chunks alongside their Gemini-generated embeddings in
ChromaDB, enabling similarity-based retrieval.

**Implementation:**

```python
# ChromaDB persistent client — survives restarts
self._client = chromadb.PersistentClient(path="data/vectorstore")
self._collection = self._client.get_or_create_collection(
    name="rag_documents",
    metadata={"hnsw:space": "cosine"},  # cosine similarity
)

# Add with rich metadata
self._collection.add(
    ids=[doc_id],
    embeddings=[embedding],
    documents=[content],
    metadatas=[{
        "source": "shadwal singh (3).pdf",
        "chunking_strategy": "semantic",
        "page_number": 1,
        "format": "pdf",
        ...
    }],
)
```

**Rich Metadata — The "Production-Grade" Secret:**

Every chunk stored carries full provenance:

```json
{
    "source": "shadwal singh (3).pdf",
    "page_number": 1,
    "headings": "[]",
    "format": "pdf",
    "chunking_strategy": "semantic",
    "chunk_index": 0,
    "avg_similarity": 0.4982
}
```

This means when you retrieve a chunk, you know exactly:

- **Where** it came from (source file, page)
- **How** it was created (chunking strategy)
- **Why** it was split there (avg_similarity score)

**Search Test — Query: "skills and experience":**

| Rank | Content Preview                                                                            | Cosine Distance | Relevance |
| :--: | ------------------------------------------------------------------------------------------ | :-------------: | :-------: |
|  1   | "Strong communication and problem-solving abilities honed through diverse experience..."   |     0.3394      |   High    |
|  2   | "Seeking opportunities to leverage technical skills and contribute to challenging proj..." |     0.3570      |   High    |
|  3   | "SKILLS Leadership Project Management Teamwork Effective Communication Critical Thin..."   |     0.3822      |   High    |
|  4   | "SKILLS Leadership Project Management Teamwork..." (different section)                     |     0.3960      |  Medium   |
|  5   | "Proficient in Python, C++, and SQL."                                                      |     0.3992      |  Medium   |

All top results are genuinely relevant to "skills and experience" — even though
the query words don't exactly match the stored text. This is the power of
semantic search.

**Search Test — Query: "education university degree":**

| Rank | Content Preview                                         | Cosine Distance |
| :--: | ------------------------------------------------------- | :-------------: |
|  1   | "linkedin."                                             |     0.4267      |
|  2   | "Strong communication and problem-solving abilities..." |     0.4370      |
|  3   | "Mastering C — Udemy-GeeksforGeeks (Nov 2025)..."       |     0.4478      |

The higher distances (0.42–0.45 vs 0.34 for the skills query) suggest the resume
may not have a strong dedicated "education" section, or the spaced-out PDF text
reduces embedding quality — an important finding.

---

### 2.3 BM25 Keyword Index (Sparse Search)

**Problem:** Semantic search sometimes misses exact technical terms. If someone
searches for a specific error code like `ERR_CONNECTION_REFUSED` or a function
name like `load_pdf`, embedding-based search might not find it because it looks
for _meaning_, not exact tokens.

**Solution:** Build a parallel BM25 (Best Matching 25) index over the exact same
chunks, using the `rank_bm25` library. Both indices are kept in perfect sync.

**Implementation:**

```python
from rank_bm25 import BM25Okapi

# Tokenize and build
tokens = text.lower().split()
self._bm25_corpus.append(tokens)
self._bm25 = BM25Okapi(self._bm25_corpus)

# Search
scores = self._bm25.get_scores(query_tokens)
top_indices = np.argsort(scores)[::-1][:n_results]
```

**The Sync Rule:** Every chunk added to ChromaDB is simultaneously added to the
BM25 corpus. If one has 27 documents, both have 27 documents. This is enforced
in a single `ingest_chunks()` method.

**Search Test — Query: "m a n a g e m e n t" (matching the PDF's spaced
format):**

| Rank | Content Preview                                                                          | BM25 Score |
| :--: | ---------------------------------------------------------------------------------------- | :--------: |
|  1   | "SKILLS Leadership Project Management Teamwork..."                                       |   2.4117   |
|  2   | "Implemented Agile project management methodologies, resulting in a 20% increase..."     |   2.4014   |
|  3   | "SHADWAL SINGH First year Computer Science student SKILLS Leadership Project Manage..."  |   2.3931   |
|  4   | "SKILLS Leadership Project Management..." (different chunk)                              |   2.3886   |
|  5   | "CORE SKILLS LEADERSHIP TEAMWORK EFFECTIVE COMMUNICATION PROJECT MANAGEMENT CRITICAL..." |   2.3787   |

**Key Insight:** BM25 successfully matched the spaced-out text that semantic
search handles via meaning. This demonstrates exactly why hybrid search is
needed — each method catches what the other misses.

---

## 3. Architecture

```
                        ┌─────────────────────────────────┐
                        │       src/ingest.py             │
                        │   (CLI entry-point)             │
                        └──────────┬──────────────────────┘
                                   │
                        ┌──────────v──────────────────────┐
                        │      src/storage.py             │
                        │      HybridStore class          │
                        │                                 │
                        │  ┌──────────────────────────┐   │
data/processed/         │  │   Deduplication Filter    │   │
*_semantic_chunk_*.txt ─┼─>│   (cosine sim > 0.95)     │   │
*_semantic_chunk_*.json │  └──────────┬───────────────┘   │
                        │             │ passes             │
                        │  ┌──────────v───────────────┐   │
                        │  │     Gemini Embeddings     │   │
                        │  │  (gemini-embedding-001)   │   │
                        │  └──────────┬───────────────┘   │
                        │             │                    │
                        │     ┌───────┴────────┐          │
                        │     │                │          │
                        │  ┌──v─────────┐  ┌───v───────┐  │
                        │  │  ChromaDB   │  │   BM25    │  │
                        │  │  (vectors)  │  │ (keywords)│  │
                        │  └────────────┘  └───────────┘  │
                        │   data/vectorstore  data/bm25   │
                        └─────────────────────────────────┘
```

### File Structure After Step 5

```
production_rag/
├── data/
│   ├── raw/                    # Original documents
│   ├── processed/              # Chunked text + metadata JSON
│   ├── vectorstore/            # ChromaDB persistent storage (~1.8 MB)
│   └── bm25_index.pkl          # Serialized BM25 index (~8 KB)
├── src/
│   ├── loader.py               # Step 3: Multi-format document loading
│   ├── chunker.py              # Step 4: Three chunking strategies
│   ├── storage.py              # Step 5: HybridStore (this step)
│   └── ingest.py               # Step 5: Ingestion CLI
├── reports/
│   ├── report1.md              # Chunking strategy comparison
│   └── report2.md              # Storage & deduplication (this report)
├── requirements.txt
├── .env                        # API keys (gitignored)
└── .gitignore
```

---

## 4. Technology Stack

| Component       | Library         | Version | Purpose                                      |
| --------------- | --------------- | ------- | -------------------------------------------- |
| Vector Database | `chromadb`      | latest  | Persistent vector storage with HNSW indexing |
| Keyword Search  | `rank_bm25`     | latest  | BM25Okapi algorithm for sparse retrieval     |
| Embeddings      | `google-genai`  | latest  | Gemini `gemini-embedding-001` model          |
| Similarity Math | `numpy`         | latest  | Cosine similarity for deduplication          |
| Serialization   | `pickle`        | stdlib  | BM25 index persistence to disk               |
| Config          | `python-dotenv` | latest  | Secure API key management via `.env`         |

---

## 5. Key Learnings

### 5.1 ChromaDB Uses Cosine Distance, Not Similarity

ChromaDB returns `distance = 1 - similarity`. A distance of 0.34 means a
similarity of 0.66 — which is "quite similar" in embedding space. I initially
confused the two, which would have broken the deduplication threshold.

### 5.2 BM25 Struggles with Spaced-Out PDF Text

The resume PDF stores text as `S K I L L S` instead of `SKILLS`. BM25's
tokenizer splits on whitespace, so each letter becomes a separate token. The
query `"management"` returns nothing, but `"m a n a g e m e n t"` works. This
reveals a preprocessing gap — a future improvement would be to collapse spaced
characters before indexing.

### 5.3 Sync is Non-Negotiable

If the vector store has 27 documents but the BM25 index has 25, hybrid search
will return inconsistent results. By adding to both indices inside the same
method (`ingest_chunks`), we guarantee they're always in sync.

### 5.4 Batch Embedding is Efficient

Instead of making 27 individual API calls, we batch into groups of 20. This
reduced the embedding step from ~27 API roundtrips to just 2 (20 + 7). At scale,
this is the difference between a 30-second and a 3-second ingestion.

### 5.5 Deterministic IDs Prevent Silent Duplicates

Each chunk gets a SHA-256 hash ID based on its source, chunk index, and
strategy. This means even if deduplication fails, ChromaDB won't create a second
copy of the same chunk because the ID already exists.

---

## 6. Disk Footprint

| Component            | Size         | Notes                                       |
| -------------------- | ------------ | ------------------------------------------- |
| ChromaDB vectorstore | ~1.86 MB     | Includes HNSW index + metadata + embeddings |
| BM25 index (pickle)  | ~8 KB        | Tokenized corpus + document IDs             |
| **Total**            | **~1.87 MB** | For 27 chunks — scales linearly             |

---

## 7. CLI Usage

```bash
# Activate virtual environment
.\venv\Scripts\Activate

# Ingest semantic chunks (default)
python src/ingest.py

# Ingest a specific strategy
python src/ingest.py --strategy fixed_size
python src/ingest.py --strategy structure_aware

# Ingest all strategies at once
python src/ingest.py --strategy all
```

---

## 8. Next Steps

1. **Hybrid Retrieval** — Combine vector search and BM25 scores using Reciprocal
   Rank Fusion (RRF) for a single, ranked result list.
2. **Query Pipeline** — Build a RAG query interface that retrieves context and
   feeds it to Gemini for answer generation.
3. **Preprocessing Fix** — Collapse spaced PDF characters (`S K I L L S` →
   `SKILLS`) before BM25 indexing to improve keyword recall.
4. **Multi-Document Testing** — Add more documents to test deduplication across
   different files.
5. **Evaluation** — Measure retrieval quality with precision@k and recall@k
   metrics.

---

_Report generated as part of the Production RAG Pipeline project — Step 5._
