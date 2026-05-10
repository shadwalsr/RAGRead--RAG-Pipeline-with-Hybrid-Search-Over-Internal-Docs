## 📊 Visual Assets (ASCII Graphs)

### 1. The Benchmark Results

A visual breakdown of how the three chunking strategies performed against the
50+ golden Q&A dataset.

```text
            [ Chunking Strategy Benchmark Results ]
                  (50+ Golden Q&A Pairs)

Strategy          Correctness   Citation Cov.   Latency
─────────────────────────────────────────────────────────
Fixed-size      │ ██████ 0.33 │ ████████ 100% │ 5.46s
Structure-aware │ ██████ 0.33 │ ████████ 100% │ 5.37s  <-- Chosen Standard
Semantic        │ 0.00        │ ████████ 100% │ 5.11s  <-- Fastest, but failed
```

### 2. Why Semantic Failed (The Resume Problem)

A diagram illustrating exactly why semantic chunking broke down on the
"Whyschool Academy" query compared to structure-aware chunking.

```text
                 [ The Source Document Text ]
 "Shadwal worked as a Software Engineer at Whyschool Academy"

    Semantic Chunking                   Structure-Aware Chunking
  (Splits too Aggressively)               (Respects Paragraphs)
            │                                       │
   ┌────────┴────────┐                              ▼
   ▼                 ▼                     ┌─────────────────┐
 [Chunk 1]         [Chunk 2]               │    [Chunk 1]    │
"...Software      "...at Whyschool         │ "...Software    │
 Engineer..."      Academy..."             │  Engineer at    │
   (Role)          (Company)               │  Whyschool..."  │
                                           └─────────────────┘
 Result: ❌                              Result: ✅
 Context is scattered.                   Context stays intact.
 Retriever misses the link.              Retriever finds the answer.
```

### 3. Engineering vs. Tinkering (The Broader Point)

A visual showing how the evaluation benchmark forces data-driven decisions over
theoretical preference.

```[]
       The Theory                            The Data
  (What "makes sense")               (What the Benchmark says)

   "Semantic Chunking                  "Structure-Aware is
    is topic-aligned                    the most accurate
    and meaning-aware!"                 on this dataset."
           │                                    │
           ▼                                    ▼
     [ Tinkering ]                       [ Engineering ]
   (Guessing based on                 (Deciding based on the
    personal preference)               objective delta)
```

---
