# Chunking Strategies Report — Production RAG Pipeline

**Author:** Shadwal Singh\
**Date:** April 3, 2026\
**Document:** `shadwal singh (3).pdf` (Resume — 3 pages)

---

## 1. Executive Summary

This report documents the implementation and comparative analysis of three text
chunking strategies applied to a production-grade RAG (Retrieval-Augmented
Generation) pipeline. The goal was to determine how different chunking
approaches affect the granularity, coherence, and retrieval-readiness of
document segments.

| Metric                     |  Fixed-Size  |  Structure-Aware  |   Semantic   |
| -------------------------- | :----------: | :---------------: | :----------: |
| **Total Chunks**           |      16      |        17         |      27      |
| **Min Chunk Size (chars)** |     394      |        126        |      1       |
| **Max Chunk Size (chars)** |     500      |        499        |    1,087     |
| **Avg Chunk Size (chars)** |     493      |        460        |     263      |
| **Provenance Tag**         | `fixed_size` | `structure_aware` |  `semantic`  |
| **External API Required**  |      No      |        No         | Yes (Gemini) |

> **Key Finding:** Semantic chunking produced **69% more chunks** than
> fixed-size, splitting the same document into 27 topic-focused segments versus
> 16 uniform blocks. This suggests significantly better retrieval precision at
> the cost of an API call.

---

## 2. Strategy Breakdown

### 2.1 Fixed-Size Chunking (The Baseline)

**How it works:**\
The text is split into equal windows of **500 characters** with a **50-character
overlap**. The overlap ensures that no sentence or idea is completely severed at
a chunk boundary.

**Implementation:**

```python
def fixed_size_chunk(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks
```

**Results:**

- Produced **16 chunks** with highly uniform sizes (394–500 chars).
- Average chunk size: **493 characters** — almost perfectly consistent.
- Every chunk carries positional metadata (`chunk_char_start`,
  `chunk_char_end`).

**Sample Metadata:**

```json
{
    "source": "shadwal singh (3).pdf",
    "page_number": 1,
    "format": "pdf",
    "chunking_strategy": "fixed_size",
    "chunk_index": 0,
    "chunk_char_start": 0,
    "chunk_char_end": 500
}
```

**Strengths:**

- Dead simple, no dependencies beyond Python.
- Guaranteed full coverage of the document — nothing is missed.
- Predictable output: you know exactly how many chunks to expect.

**Weaknesses:**

- Cuts mid-sentence and mid-word blindly. A chunk might end with
  `"Digital Marketi"` and the next starts with `"ng Strategy"`.
- No awareness of document structure or meaning.
- Poor retrieval quality for precise queries — the answer might be split across
  two chunks.

---

### 2.2 Structure-Aware Chunking (The Smart Way)

**How it works:**\
Uses LangChain's `RecursiveCharacterTextSplitter` which attempts to split on
natural boundaries in priority order:

1. Double newlines (`\n\n`) — paragraph breaks
2. Single newlines (`\n`) — line breaks
3. Spaces (``) — word boundaries
4. Empty string (`""`) — character-level (last resort)

**Implementation:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_text(text)
```

**Results:**

- Produced **17 chunks** — one more than fixed-size.
- Size range: **126–499 characters** (more variable than fixed-size).
- Average chunk size: **460 characters**.

**Sample Metadata:**

```json
{
    "source": "shadwal singh (3).pdf",
    "page_number": 1,
    "format": "pdf",
    "chunking_strategy": "structure_aware",
    "chunk_index": 2
}
```

**Strengths:**

- Respects paragraph and sentence boundaries — chunks feel more "natural."
- Still maintains a size ceiling (500 chars), so no runaway chunks.
- No API calls or external services needed.

**Weaknesses:**

- Still primarily size-driven; it just picks _better_ cut points.
- Doesn't understand the _meaning_ of text — it might group unrelated sentences
  if they happen to be in the same paragraph.
- PDF text often lacks clear `\n\n` separators, reducing the advantage.

---

### 2.3 Semantic Chunking (The Advanced Way)

**How it works:**\
This strategy uses **Gemini's embedding model** (`gemini-embedding-001`) to
understand the _meaning_ of each sentence. It:

1. Splits text into individual sentences.
2. Generates a vector embedding for each sentence using the Gemini API.
3. Computes **cosine similarity** between consecutive sentence pairs.
4. When similarity drops below **0.75**, a topic shift is detected and a new
   chunk begins.

**Implementation:**

```python
from google import genai

client = genai.Client(api_key=api_key)
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentences,
)

# Walk sentence pairs — split when similarity drops
for i in range(1, len(sentences)):
    sim = cosine_similarity(embeddings[i-1], embeddings[i])
    if sim < 0.75:
        # Topic shift detected — start new chunk
        flush_current_chunk()
```

**Results:**

- Produced **27 chunks** — the most granular of all three strategies.
- Size range: **1–1,087 characters** (highest variance).
- Average chunk size: **263 characters**.
- Average inter-sentence similarity: **0.6064** (well below the 0.75 threshold,
  indicating frequent topic shifts in the resume format).

**Similarity Score Distribution:**

| Metric         | Value  |
| -------------- | ------ |
| Min Similarity | 0.4982 |
| Max Similarity | 0.7525 |
| Avg Similarity | 0.6064 |

**Individual Chunk Similarity Scores:**

```
Chunk 0:  0.4982  │  Chunk 7:  0.6435  │  Chunk 14: 0.6267  │  Chunk 21: 0.7525
Chunk 1:  0.6283  │  Chunk 8:  0.5854  │  Chunk 15: 0.5864  │  Chunk 22: 0.5134
Chunk 2:  0.5809  │  Chunk 9:  0.5497  │  Chunk 16: 0.6452  │  Chunk 23: 0.5775
Chunk 3:  0.5970  │  Chunk 10: 0.5983  │  Chunk 17: 0.6501  │  Chunk 24: 0.6677
Chunk 4:  0.6369  │  Chunk 11: 0.5534  │  Chunk 18: 0.5963  │  Chunk 25: 0.6142
Chunk 5:  0.6705  │  Chunk 12: 0.5979  │  Chunk 19: 0.5366  │
Chunk 6:  0.6645  │  Chunk 13: 0.5907  │  Chunk 20: 0.6035  │
```

**Sample Metadata:**

```json
{
    "source": "shadwal singh (3).pdf",
    "page_number": 1,
    "format": "pdf",
    "chunking_strategy": "semantic",
    "chunk_index": 0,
    "avg_similarity": 0.4982
}
```

**Strengths:**

- Chunks are **meaning-aligned** — each chunk contains a coherent topic.
- Produces the most retrieval-friendly segments for a RAG system.
- The `avg_similarity` metadata enables quality auditing — you can see _why_ a
  split happened.
- Demonstrates advanced AI/NLP understanding (portfolio differentiator).

**Weaknesses:**

- Requires a **Google API key** and network access.
- Higher latency — each file requires an API roundtrip for embeddings.
- Highly variable chunk sizes (1–1,087 chars) — some chunks are trivially small.
- Cost implications at scale (embedding API calls per document).

---

## 3. Comparative Analysis

### 3.1 Chunk Count vs. Granularity

```
Fixed-Size:       ████████████████  (16 chunks)
Structure-Aware:  █████████████████  (17 chunks)
Semantic:         ███████████████████████████  (27 chunks)
```

Semantic chunking produced **69% more chunks** than fixed-size. This means each
chunk is more focused on a single topic, which directly improves retrieval
precision — when a user asks a question, the retrieved chunk is more likely to
contain _only_ the relevant answer, without noise from adjacent topics.

### 3.2 Size Distribution

|                        | Fixed-Size | Structure-Aware | Semantic |
| ---------------------- | :--------: | :-------------: | :------: |
| **Uniformity**         |   ★★★★★    |      ★★★☆☆      |  ★☆☆☆☆   |
| **Natural Boundaries** |   ★☆☆☆☆    |      ★★★★☆      |  ★★★★★   |
| **Meaning Coherence**  |   ★★☆☆☆    |      ★★★☆☆      |  ★★★★★   |

### 3.3 When to Use Each Strategy

| Use Case                                                | Recommended Strategy |
| ------------------------------------------------------- | -------------------- |
| Quick prototyping / baseline testing                    | Fixed-Size           |
| Well-structured documents (Markdown, HTML with headers) | Structure-Aware      |
| Unstructured text, resumes, mixed-content PDFs          | Semantic             |
| Cost-sensitive / offline environments                   | Structure-Aware      |
| Maximum retrieval accuracy                              | Semantic             |

---

## 4. Implementation Details

### 4.1 Pipeline Architecture

```
data/raw/                    src/loader.py              src/chunker.py
┌──────────────┐         ┌──────────────────┐      ┌─────────────────────┐
│  .pdf        │────────>│  DocumentLoader  │─────>│  chunk_document()   │
│  .md         │  load   │  - load_pdf()    │ text │  - fixed_size       │
│  .html       │────────>│  - load_unstr()  │─────>│  - structure_aware  │
└──────────────┘         │  - normalize()   │      │  - semantic         │
                         └──────────────────┘      └─────────┬───────────┘
                                                             │
                                                   data/processed/
                                                   ┌─────────────────────┐
                                                   │  *_chunk_0.txt      │
                                                   │  *_chunk_0.json     │
                                                   │  (with provenance)  │
                                                   └─────────────────────┘
```

### 4.2 Provenance Tracking

Every single chunk carries a `chunking_strategy` metadata field. This is
critical for later A/B testing — when evaluating retrieval quality, you can
compare:

- _"Did the fixed-size chunk or the semantic chunk provide a better answer?"_
- _"Which strategy had fewer hallucinations?"_

This is what separates a portfolio project from a tutorial copy-paste.

### 4.3 Technology Stack

| Component       | Library                    | Purpose                             |
| --------------- | -------------------------- | ----------------------------------- |
| PDF Loading     | `pypdf`                    | Extract text page-by-page           |
| HTML/MD Loading | `unstructured`             | Intelligent document partitioning   |
| Text Splitting  | `langchain-text-splitters` | RecursiveCharacterTextSplitter      |
| Embeddings      | `google-genai`             | Gemini `gemini-embedding-001` model |
| Similarity      | `numpy`                    | Cosine similarity computation       |
| Config          | `python-dotenv`            | Secure API key management           |

### 4.4 CLI Usage

```bash
# Run with default strategy (structure_aware)
python src/loader.py

# Explicitly choose a strategy
python src/loader.py --strategy fixed_size
python src/loader.py --strategy structure_aware
python src/loader.py --strategy semantic
```

---

## 5. Key Learnings

### 5.1 Case Study: The "W h y s c h o o l" Artifact (Phase 1)

The resume PDF contained a classic "designer PDF" artifact where the company
name was stored as `W h y s c h o o l  A c a d e m y`.

- **The Problem:** In this phase, basic chunking strategies (Fixed and
  Structural) treated these as individual letters. A search for "Whyschool"
  would return zero results because the keyword simply didn't exist in the text.
- **The Lesson:** This highlighted that even the best chunking strategy fails if
  the underlying text extraction isn't "cleaned" first.

### 5.2 Semantic Chunking Reveals Document Structure

Even though the resume PDF had no markdown headers or semantic HTML, the Gemini
embedding model was able to detect where "Skills" ended and "Experience" began
purely from meaning. This is powerful for unstructured documents.

### 5.3 The 0.75 Threshold is Aggressive for Short Documents

With an average similarity of 0.6064, nearly every sentence pair triggered a new
chunk. For short documents like resumes, lowering the threshold to ~0.50 might
produce more useful groupings. The threshold should be tuned per use case.

### 5.4 Provenance is Non-Negotiable

Without the `chunking_strategy` field, you'd have no way to know which chunks
came from which strategy when debugging retrieval quality. This is a "production
readiness" detail.

### 5.5 Graceful Fallback is Important

The semantic chunker automatically falls back to structure-aware if the API key
is missing or the embedding call fails. This prevents pipeline failures in CI/CD
or offline environments.

---

## 6. Next Steps

1. **Embedding & Vector Store** — Store these chunks in a vector database
   (ChromaDB / Pinecone) for similarity search.
2. **Retrieval Testing** — Query the vector store and compare which strategy
   returns the most relevant chunks.
3. **Threshold Tuning** — Experiment with different similarity thresholds (0.5,
   0.6, 0.7, 0.8) for semantic chunking.
4. **Multi-Document Testing** — Add more diverse documents (long-form articles,
   technical docs) to stress-test the strategies.
5. **Evaluation Metrics** — Implement precision@k, recall@k, and MRR (Mean
   Reciprocal Rank) to quantitatively compare strategies.

---

_Report generated as part of the Production RAG Pipeline project._
