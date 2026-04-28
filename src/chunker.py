"""
Strategic Chunking Module

Provides three switchable chunking strategies with provenance tracking:
1. Fixed-Size Chunking   - 500 chars, 50-char overlap (baseline)
2. Structure-Aware       - LangChain RecursiveCharacterTextSplitter
3. Semantic Chunking     - Gemini embedding-based topic segmentation
"""

import re
import os
import numpy as np
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _split_into_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter that handles common abbreviations.
    Splits on period/question-mark/exclamation followed by whitespace.
    """
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# 1. Fixed-Size Chunking
# ---------------------------------------------------------------------------

def fixed_size_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    base_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Splits *text* into fixed-size windows with overlap.

    Returns a list of dicts: {"content": str, "metadata": dict}
    """
    chunks: List[Dict[str, Any]] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        meta = dict(base_metadata) if base_metadata else {}
        meta["chunking_strategy"] = "fixed_size"
        meta["chunk_index"] = idx
        meta["chunk_char_start"] = start
        meta["chunk_char_end"] = min(end, len(text))

        chunks.append({"content": chunk_text, "metadata": meta})

        start += chunk_size - overlap
        idx += 1

    return chunks


# ---------------------------------------------------------------------------
# 2. Structure-Aware Chunking
# ---------------------------------------------------------------------------

def structure_aware_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    base_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Uses LangChain's RecursiveCharacterTextSplitter which tries to split on
    double newlines → single newlines → spaces → characters, keeping
    related paragraphs together.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)
    chunks: List[Dict[str, Any]] = []

    for idx, chunk_text in enumerate(raw_chunks):
        meta = dict(base_metadata) if base_metadata else {}
        meta["chunking_strategy"] = "structure_aware"
        meta["chunk_index"] = idx

        chunks.append({"content": chunk_text, "metadata": meta})

    return chunks


# ---------------------------------------------------------------------------
# 3. Semantic Chunking
# ---------------------------------------------------------------------------

def _get_embeddings(sentences: List[str], model_name: str = "gemini-embedding-001") -> List[np.ndarray]:
    """
    Generates embeddings for a list of sentences using the Gemini
    embedding model via the google-genai SDK.
    Requires GOOGLE_API_KEY to be set.
    """
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable "
            "to use semantic chunking."
        )

    client = genai.Client(api_key=api_key)

    # Embed in a single batch for efficiency
    result = client.models.embed_content(
        model=model_name,
        contents=sentences,
    )

    return [np.array(e.values) for e in result.embeddings]


def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.75,
    base_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Splits text into chunks based on *meaning*.

    1. Split text into sentences.
    2. Embed each sentence with Gemini.
    3. Walk through sentence pairs; when the cosine similarity drops
       below *similarity_threshold*, start a new chunk.

    Falls back to structure-aware chunking if embedding fails.
    """
    sentences = _split_into_sentences(text)

    if len(sentences) <= 1:
        meta = dict(base_metadata) if base_metadata else {}
        meta["chunking_strategy"] = "semantic"
        meta["chunk_index"] = 0
        return [{"content": text, "metadata": meta}]

    try:
        embeddings = _get_embeddings(sentences)
    except Exception as e:
        print(f"[semantic_chunk] Embedding failed ({e}), falling back to structure_aware.")
        return structure_aware_chunk(text, base_metadata=base_metadata)

    # Walk pairs and decide where to split
    chunks: List[Dict[str, Any]] = []
    current_group: List[str] = [sentences[0]]
    idx = 0

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])

        if sim < similarity_threshold:
            # Topic shift detected — flush current group
            meta = dict(base_metadata) if base_metadata else {}
            meta["chunking_strategy"] = "semantic"
            meta["chunk_index"] = idx
            meta["avg_similarity"] = round(float(
                np.mean([
                    _cosine_similarity(embeddings[j], embeddings[j + 1])
                    for j in range(max(0, i - len(current_group)), i - 1)
                ]) if len(current_group) > 1 else sim
            ), 4)

            chunks.append({
                "content": " ".join(current_group),
                "metadata": meta,
            })
            current_group = []
            idx += 1

        current_group.append(sentences[i])

    # Flush final group
    if current_group:
        meta = dict(base_metadata) if base_metadata else {}
        meta["chunking_strategy"] = "semantic"
        meta["chunk_index"] = idx
        chunks.append({"content": " ".join(current_group), "metadata": meta})

    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

STRATEGIES = {
    "fixed_size": fixed_size_chunk,
    "structure_aware": structure_aware_chunk,
    "semantic": semantic_chunk,
}


def chunk_document(
    text: str,
    strategy: str = "structure_aware",
    base_metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Public entry-point. Pick a strategy by name and chunk the text.

    Args:
        text:          The plain-text content to chunk.
        strategy:      One of "fixed_size", "structure_aware", or "semantic".
        base_metadata: Dict of fields to copy into every chunk's metadata.
        **kwargs:      Extra keyword args forwarded to the strategy function
                       (e.g. chunk_size, overlap, similarity_threshold).
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {list(STRATEGIES.keys())}"
        )

    fn = STRATEGIES[strategy]
    return fn(text, base_metadata=base_metadata, **kwargs)
