<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini_AI-Powered-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">📑 RAG-READ</h1>
<h3 align="center">A Production-Grade RAG Pipeline with Hybrid Search Over Internal Documents</h3>

<p align="center">
  <em>Upload your documents. Ask questions. Get cited, verified answers — backed by real evidence from your own files.</em>
</p>

<p align="center">
  <strong>Built by <a href="https://github.com/shadwalsr">Shadwal Singh</a></strong> · April – May 2026
</p>

---

## 🤔 What Is This? (The Simple Version)

Imagine you have a stack of PDFs, notes, or documents — maybe a resume, a company handbook, research papers, or meeting notes. Now imagine you could just **ask a question** and get an accurate answer **with citations pointing to exactly where the info came from**.

That's what **RAG-READ** does.

> **RAG** stands for **R**etrieval-**A**ugmented **G**eneration.  
> Instead of an AI just guessing an answer, it first **searches your documents** for relevant information, then **generates an answer based only on what it found** — and **cites its sources**.

### 🎯 Why Does This Matter?

| Problem | How RAG-READ Solves It |
|---|---|
| AI makes stuff up ("hallucinations") | Only answers from **your actual documents** — refuses if it can't find proof |
| You don't know where info came from | Every claim has a **[1], [2]** citation pointing to the source chunk |
| Keyword search misses context | Uses **both** keyword matching AND meaning-based search together |
| No way to verify AI output | Built-in **citation verifier** checks every claim against the source |

---

## 🏗️ How It Works — The Big Picture

The pipeline has **5 stages**. Think of it like an assembly line where your documents go in one end, and verified answers come out the other:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAG-READ PIPELINE                                │
│                                                                         │
│  ① INGEST     ② RETRIEVE      ③ GENERATE      ④ VERIFY     ⑤ SERVE    │
│                                                                         │
│  Your Files → Search Index → Find Matches → AI Answer → Check Facts → You │
│                                                                         │
│  PDF/MD/TXT   Vector + BM25   Hybrid Fusion   Gemini AI   Citation     API │
│  HTML         Dual Index      RRF + Rerank    w/ Sources  Auditor      Dashboard │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage-by-Stage Breakdown

<details>
<summary><strong>① INGEST — "Eating" Your Documents</strong></summary>

```
                    ┌──────────┐
  📄 PDF ──────────▶│          │      ┌─────────────┐     ┌──────────────┐
  📝 Markdown ────▶│  LOADER  │─────▶│   CHUNKER   │────▶│   STORAGE    │
  🌐 HTML ────────▶│          │      │             │     │              │
  📃 TXT ─────────▶│ (PyMuPDF │      │ Fixed-size  │     │ ChromaDB     │
                    │  + Unstr)│      │ Structural  │     │ (vectors)    │
                    └──────────┘      │ Semantic    │     │     +        │
                                      └─────────────┘     │ BM25 Index  │
                                                          │ (keywords)  │
                                                          └──────────────┘
```

**What happens:**
1. **Loader** reads your files (PDFs via PyMuPDF, everything else via Unstructured)
2. **Chunker** splits them into bite-sized pieces using one of 3 strategies
3. **Storage** saves each chunk into two search indexes simultaneously

**Three chunking strategies:**

| Strategy | How It Splits | Best For |
|---|---|---|
| **Fixed** | Every 500 characters with 50-char overlap | Simple, predictable splits |
| **Structural** | By paragraph breaks (double newlines) | Documents with clear sections |
| **Semantic** | AI finds natural topic boundaries | Highest quality, but slower |

</details>

<details>
<summary><strong>② RETRIEVE — Finding the Right Information</strong></summary>

```
                         ┌─────────────────────┐
                    ┌───▶│   DENSE SEARCH       │───┐
                    │    │ (Gemini Embeddings)   │   │
  "What is X?" ────┤    │ Understands MEANING   │   │     ┌──────────┐
                    │    └─────────────────────┘   ├────▶│  RRF     │──▶ Top K
                    │    ┌─────────────────────┐   │     │  FUSION  │    Chunks
                    └───▶│   SPARSE SEARCH      │───┘     └──────────┘
                         │ (BM25 Keywords)      │
                         │ Matches EXACT WORDS  │
                         └─────────────────────┘
```

**Why two search methods?** Each catches things the other misses:

- **Dense (Semantic):** Understands that "management experience" and "led a team" mean the same thing
- **Sparse (BM25 Keywords):** Catches exact names, dates, and technical terms the AI might miss

**Reciprocal Rank Fusion (RRF)** combines both result lists into one ranked list. You control the blend with the `alpha` slider (0 = all keywords, 1 = all semantic).

**Optional Reranker:** An AI judge reviews the merged results and picks the truly relevant ones — slower but smarter.

</details>

<details>
<summary><strong>③ GENERATE — Producing the Answer</strong></summary>

```
  Top Chunks ──▶ ┌────────────────────────────┐ ──▶ Cited Answer
                 │        GEMINI AI            │
                 │                              │     "Based on [1], the
                 │  Rules:                      │      candidate has 3 years
                 │  • Only use the context      │      of experience [2]..."
                 │  • Cite every claim [1],[2]  │
                 │  • Refuse if unsure          │
                 │  • Score own confidence 0-10 │
                 └────────────────────────────┘
```

The AI doesn't just answer — it returns a **structured JSON response** including:
- `retrieval_confidence_score` — How confident it is in the retrieved context (0-10)
- `can_answer` — Whether it has enough info to answer at all
- `structured_refusal` — If it can't answer, it explains what's missing
- `answer` — The actual cited answer

</details>

<details>
<summary><strong>④ VERIFY — Fact-Checking the AI's Own Work</strong></summary>

```
  Answer with [1],[2] ──▶ ┌─────────────────────┐ ──▶ Verified Answer
                          │   CITATION VERIFIER   │
                          │                       │     [1] ✅ Supported
                          │ For each [n] citation:│     [2] ✅ Supported
                          │ • Extract the claim   │     [3] ⚠️ UNVERIFIED
                          │ • Pull chunk [n]      │
                          │ • Ask: "Does this     │
                          │   chunk support this  │
                          │   claim? YES/NO"      │
                          └─────────────────────┘
```

This is the **self-auditing** layer. Every single citation `[n]` gets checked:
- A separate AI call compares the claim to the actual source chunk
- If the chunk doesn't support the claim → the citation is flagged `[n UNVERIFIED]`
- The final `citation_coverage_pct` tells you what % of citations passed

</details>

<details>
<summary><strong>⑤ SERVE — Delivering Results to You</strong></summary>

Two ways to interact:

| Interface | What It Is | How to Start |
|---|---|---|
| **FastAPI** | REST API on `localhost:8000` | `python src/api.py` |
| **Streamlit Dashboard** | Visual UI with upload, search, and results | `streamlit run src/dashboard.py` |

The dashboard features a striking **yellow-and-black brand identity** with glassmorphism panels, pipeline visualization, and side-by-side search comparison (Hybrid vs Dense-only).

</details>

---

## 📁 Project Structure

```
RAG-READ/
│
├── src/                        # ← All source code lives here
│   ├── loader.py               # Reads PDFs, Markdown, HTML, TXT files
│   ├── chunker.py              # Splits documents into searchable pieces
│   ├── storage.py              # Dual index: ChromaDB vectors + BM25 keywords
│   ├── retriever.py            # Hybrid search + RRF fusion + AI reranker
│   ├── generator.py            # Answer generation + citation verification
│   ├── evaluator.py            # LLM-as-a-judge benchmarking system
│   ├── api.py                  # FastAPI REST server (3 endpoints)
│   ├── dashboard.py            # Streamlit visual interface
│   ├── ingest.py               # CLI script to index all documents at once
│   ├── chat.py                 # Terminal-based Q&A chat interface
│   ├── query.py                # Quick retrieval test script
│   ├── compare_strategies.py   # Head-to-head chunking strategy benchmark
│   └── rebuild_bm25.py         # Utility to rebuild keyword index from vectors
│
├── data/
│   ├── raw/                    # ← Drop your documents here (PDF, MD, HTML, TXT)
│   ├── processed/              # Auto-generated: chunked data (gitignored)
│   ├── vectorstore/            # Auto-generated: ChromaDB embeddings (gitignored)
│   ├── bm25_index.pkl          # Auto-generated: keyword search index (gitignored)
│   └── eval/
│       ├── golden_qa.json      # Hand-written Q&A pairs for benchmarking
│       ├── evaluation_report.csv
│       └── eval_*.csv          # Per-strategy evaluation results
│
├── reports/                    # Engineering journey reports (12 iterations)
├── .streamlit/config.toml      # Dashboard theme configuration
├── requirements.txt            # Python dependencies
├── .env                        # Your Google API key (never committed)
└── .gitignore                  # Keeps secrets and generated data out of git
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** installed on your machine
- A **Google AI API key** (free tier works) — [Get one here](https://aistudio.google.com/apikey)

### Step-by-Step Setup

**1. Clone the repository**

```bash
git clone https://github.com/shadwalsr/RAGRead--RAG-Pipeline-with-Hybrid-Search-Over-Internal-Docs.git
cd RAGRead--RAG-Pipeline-with-Hybrid-Search-Over-Internal-Docs
```

**2. Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up your API key**

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

**5. Add your documents**

Drop any PDF, Markdown, HTML, or TXT files into the `data/raw/` folder.

**6. Index your documents** (first time only)

```bash
cd src
python ingest.py
```

**7. Start the API server**

```bash
python api.py
```

> The API will be running at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

**8. Launch the dashboard** (in a new terminal)

```bash
streamlit run src/dashboard.py
```

> The dashboard opens at `http://localhost:8501`.

---

## 🖥️ API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns `{"status": "online"}` |
| `GET` | `/v1/documents` | Lists all indexed documents with sizes |
| `POST` | `/v1/ask` | Ask a question and get a cited answer |
| `POST` | `/v1/ingest` | Upload and index a new document |

### Example: Asking a Question

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What experience does Shadwal have?",
    "top_k": 5,
    "alpha": 0.5,
    "use_reranker": true,
    "strategy": null
  }'
```

**Response:**

```json
{
  "answer": "Based on the documents, Shadwal has experience in... [1] [2]",
  "confidence_score": 8,
  "citation_coverage": 100,
  "can_answer": true,
  "sources": [
    {
      "id": "resume_struct_0",
      "content_preview": "Shadwal Singh — Software Engineer...",
      "source": "resume.pdf",
      "score": 0.0164
    }
  ]
}
```

### Query Parameters Explained

| Parameter | Type | Default | What It Does |
|---|---|---|---|
| `query` | string | *required* | Your question in plain English |
| `top_k` | int | 5 | How many source chunks to retrieve (1–10) |
| `alpha` | float | 0.5 | Search blend: 0.0 = keywords only, 1.0 = semantic only |
| `use_reranker` | bool | true | Enable AI-powered result reranking (slower but smarter) |
| `strategy` | string | null | Filter by chunking strategy: `"fixed"`, `"structural"`, `"semantic"`, or null for all |

---

## 🧪 Evaluation System

RAG-READ includes a full **LLM-as-a-judge** evaluation framework to measure answer quality:

```
golden_qa.json ──▶ ┌──────────────┐ ──▶ evaluation_report.csv
(hand-written       │  EVALUATOR   │
 Q&A pairs)         │              │     Per question:
                    │ For each Q:  │     • correctness_score (1-5)
                    │ 1. Retrieve  │     • retrieval_confidence
                    │ 2. Generate  │     • citation_coverage
                    │ 3. Judge     │     • latency
                    └──────────────┘
```

**Run the evaluator:**

```bash
cd src
python evaluator.py
```

**Compare chunking strategies head-to-head:**

```bash
python compare_strategies.py
```

This runs the same questions against Fixed, Structural, and Semantic chunking and saves a combined comparison CSV.

---

## 🔧 Tech Stack

| Component | Technology | Why |
|---|---|---|
| **Document Parsing** | PyMuPDF + Unstructured | Fast PDF reading + robust multi-format support |
| **Embeddings** | Gemini Embedding 001 | High-quality semantic vectors from Google |
| **Vector Store** | ChromaDB (persistent) | Lightweight, local-first, cosine similarity |
| **Keyword Index** | BM25Okapi (rank_bm25) | Classic information retrieval for exact term matching |
| **LLM** | Gemini Flash | Fast inference for generation, verification, and judging |
| **API Server** | FastAPI + Uvicorn | Modern async Python API with auto-generated docs |
| **Dashboard** | Streamlit | Rapid visual prototyping with custom CSS theming |
| **Fusion** | Reciprocal Rank Fusion | Proven method to merge heterogeneous search results |

---

## 🗺️ Architecture Diagram

```
                              ┌─────────────────────────────────────┐
                              │          USER INTERFACES             │
                              │                                     │
                              │  ┌──────────┐    ┌───────────────┐  │
                              │  │ Streamlit │    │  FastAPI /docs│  │
                              │  │ Dashboard │    │  Swagger UI   │  │
                              │  └─────┬─────┘    └───────┬───────┘  │
                              │        │                  │          │
                              └────────┼──────────────────┼──────────┘
                                       │    HTTP / REST   │
                              ┌────────▼──────────────────▼──────────┐
                              │          FastAPI SERVER (api.py)      │
                              │                                      │
                              │  /v1/ask    /v1/ingest   /v1/documents│
                              └──────┬───────────┬───────────────────┘
                                     │           │
                         ┌───────────▼───┐  ┌────▼──────────┐
                         │  RETRIEVER    │  │   INGESTION   │
                         │               │  │               │
                         │ Dense Search  │  │ Loader →      │
                         │ Sparse Search │  │ Chunker →     │
                         │ RRF Fusion    │  │ Storage       │
                         │ AI Reranker   │  └───────────────┘
                         └───────┬───────┘
                                 │
                         ┌───────▼───────┐
                         │  GENERATOR    │
                         │               │
                         │ Build Context │
                         │ Generate Ans  │
                         │ Verify Cites  │
                         └───────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      STORAGE LAYER       │
                    │                          │
                    │  ┌────────┐  ┌────────┐  │
                    │  │ChromaDB│  │  BM25  │  │
                    │  │Vectors │  │Keywords│  │
                    │  └────────┘  └────────┘  │
                    └──────────────────────────┘
```

---

## 💡 Usage Tips

- **First time?** Drop a PDF into `data/raw/`, run `python src/ingest.py`, start the API, then use the dashboard.
- **Reranker** improves quality but adds ~2-3 seconds per query. Toggle it off for speed.
- **Alpha slider**: Start at 0.5 (balanced). Move toward 1.0 if your queries are conceptual, toward 0.0 if you're searching for specific names/dates.
- **Citation coverage below 100%?** Some claims weren't fully supported — check the `[UNVERIFIED]` tags.
- **BM25 out of sync?** Run `python src/rebuild_bm25.py` to reconstruct it from ChromaDB.

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Built with curiosity and caffeine ☕ by <a href="https://github.com/shadwalsr">Shadwal Singh</a></strong>
</p>
