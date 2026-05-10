## 📊 Visual Assets (ASCII Graphs)

### 1. The Interactive Chat Loop (`chat.py`)

This diagram shows the shift from a static script to an interactive loop where
every query triggers the full end-to-end pipeline.

```text
           [ User Types Question ]
                     │
                     ▼
             ┌───────────────┐
             │ chat.py Loop  │<────────────────────┐
             └───────┬───────┘                     │
                     │                             │
                     ▼                             │
┌─────────────────────────────────────────┐        │
│          The Full RAG Pipeline          │        │
│                                         │        │
│  1. Hybrid Retrieval                    │        │
│  2. RRF Fusion                          │        │
│  3. Gemini Reranker                     │        │
│  4. Grounded Generation                 │        │
│  5. Citation Verification               │        │
└────────────────────┬────────────────────┘        │
                     │                             │
                     ▼                             │
         [ Output: Answer + Stats ]                │
       (Confidence Score, Citations)               │
                     │                             │
                     └─────────────────────────────┘
                          (Next Question)
```

### 2. The Golden Dataset Breakdown (`golden_qa.json`)

A visualization of the 50+ hand-curated question-answer pairs and their specific
roles in evaluating the system.

```text
                 [ golden_qa.json ]
                 (50+ Q&A Pairs)
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ▼                ▼                ▼
 ┌──────────┐     ┌──────────┐     ┌──────────┐
 │ Fact Ret.│     │ Multi-hop│     │ Negative │
 │ (30 Qs)  │     │ (10 Qs)  │     │ (10 Qs)  │
 └────┬─────┘     └────┬─────┘     └────┬─────┘
      │                │                │
      ▼                ▼                ▼
Basic lookups,    Connecting 2+    Tests structured
roles, dates,     parts of the     refusal logic
and skills.       documents.       (stays honest).
```

### 3. The Value of a Benchmark (Tinkering vs. Engineering)

A flowchart illustrating how a benchmark transforms the development process from
guesswork to evidence-based engineering.

```text
     Tinkering                      Engineering
(Without Benchmark)              (With Benchmark)
                              
  Change Strategy                 Change Strategy
         │                               │
         ▼                               ▼
     Guessing                    Run Full Dataset
  "Looks better?"                        │
                                         ▼
                                  Objective Score
                                 (Evidence-based)
```

---
