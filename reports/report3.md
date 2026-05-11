# Report 3: The Case for Hybrid Retrieval — Bridging Meaning and Exact Wording

**Author:** Shadwal Singh\
**Date:** april 5, 2026\
**Context:** Case Study of query "whyschool" on PDF-sourced data.

---

## 1. The Observation: A Tale of Two Searches

During testing of the Phase 1 storage implementation, a query for
**"whyschool"** was executed against the database. The results revealed a
critical divergence in how our two search engines interpret data:

| Search Engine         | Query       | Result    | Finding                                      |
| --------------------- | ----------- | --------- | -------------------------------------------- |
| **Semantic (Vector)** | "whyschool" | **MATCH** | Found "Whyschool Academy" at Distance 0.4090 |
| **Keyword (BM25)**    | "whyschool" | **FAIL**  | "No exact keyword matches found"             |

### The "Whyschool" Paradox

The document clearly contains the name "Whyschool Academy". However, due to the
way the PDF was formatted (designed with spaced lettering), the underlying text
was extracted as: `W h y s c h o o l   A c a d e m y`

---

## 2. Why Semantic Search Succeeded

**The Intelligence of Embeddings:**\
The Gemini embedding model doesn't just look at characters; it transforms text
into a high-dimensional mathematical representation of its _intent_.

Even though the letters are spaced out (`W h y s...`), the vector for
"whyschool" in the query was mathematically close enough to the vector for the
spaced-out version in the database. It understood that these represent the same
entity.

---

## 3. Why BM25 (Keyword Search) Failed

**The Limitation of Strict Matching:**\
BM25 is a "bag-of-words" model. It tokenizes text based on whitespace. For the
database entry `W h y s c h o o l`, BM25 sees eight separate one-letter
documents: `W`, `h`, `y`, `s`, `c`, `h`, `o`, `o`, `l`.

When you search for the single word `"whyschool"`, BM25 looks for that exact
continuous string. Since it only has individual letters in its index, it finds
**zero** matches.

---

## 4. The Solution: Phase 2 — The Hybrid Retrieval Engine

This experiment proves that relying on either engine alone is risky for a
production system:

1. **Vector-Only** can sometimes be "too fuzzy" and retrieve irrelevant but
   semantically similar concepts.
2. **Keyword-Only** is "too brittle" and fails on formatting issues like our
   spaced PDF text.

### The Hybrid Strategy:

In Phase 2, we will implement a **Hybrid Retrieval Engine** using **Reciprocal
Rank Fusion (RRF)**.

#### How it will work:

1. We run **both** searches simultaneously.
2. We take the top results from both.
3. We use a mathematical formula (RRF) to "vote" on the best results.
   - If a result is found by both (even if one ranks it lower), it gets a
     massive boost.
   - If a result is only found by one, it still has a chance to appear if its
     score is high enough.

---

## 5. Conclusion: From "Toy" to "Production"

The "whyschool" test is the "Smoking Gun" evidence we needed. It shows that in
real-world data (especially PDFs from Canva/InDesign), text is often messy.

**Hybrid Retrieval is not a "nice-to-have"; it is the bridge that ensures our AI
can find "Whyschool Academy" regardless of whether the user types it perfectly
or the document formats it strangely.**

---

_Report generated following the successful observation of retrieval divergence
in Step 5._
