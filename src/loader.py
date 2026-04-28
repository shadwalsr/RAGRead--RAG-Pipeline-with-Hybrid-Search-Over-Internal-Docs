import os
import re
import json
import argparse
from typing import List, Dict, Any

from dotenv import load_dotenv
from pypdf import PdfReader
from unstructured.partition.auto import partition

from chunker import chunk_document, STRATEGIES

# Load .env file (GOOGLE_API_KEY etc.)
load_dotenv()


class DocumentLoader:
    """
    A class to load, normalize, and extract metadata from multiple file formats.
    After loading, documents are passed through a configurable chunking strategy.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        chunking_strategy: str = "structure_aware",
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.chunking_strategy = chunking_strategy

        # Ensure directories exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Text normalisation
    # ------------------------------------------------------------------

    def normalize_text(self, text: str) -> str:
        """
        Strips extra whitespace and converts to clean plaintext.
        """
        # Replace multiple spaces/newlines with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a PDF using pypdf.
        """
        documents = []
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    metadata = {
                        "source": os.path.basename(file_path),
                        "page_number": i + 1,
                        "headings": [],  # pypdf doesn't provide structured headings easily
                        "format": "pdf",
                    }
                    documents.append({
                        "content": self.normalize_text(text),
                        "metadata": metadata,
                    })
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
        return documents

    def load_unstructured(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads HTML or Markdown using the unstructured library.
        """
        documents = []
        try:
            elements = partition(filename=file_path)

            # Grouping by "Title" to approximate sections/headings
            current_chunk = []
            current_headings = []
            source_name = os.path.basename(file_path)

            for el in elements:
                el_text = str(el)
                if el.category == "Title":
                    # If we have accumulated text, save it as a chunk before the new heading
                    if current_chunk:
                        documents.append({
                            "content": self.normalize_text(" ".join(current_chunk)),
                            "metadata": {
                                "source": source_name,
                                "headings": list(current_headings),
                                "page_number": getattr(el.metadata, 'page_number', None),
                                "format": os.path.splitext(file_path)[1][1:],
                            },
                        })
                        current_chunk = []

                    current_headings.append(el_text)

                current_chunk.append(el_text)

            # Final chunk
            if current_chunk:
                documents.append({
                    "content": self.normalize_text(" ".join(current_chunk)),
                    "metadata": {
                        "source": source_name,
                        "headings": list(current_headings),
                        "page_number": None,
                        "format": os.path.splitext(file_path)[1][1:],
                    },
                })
        except Exception as e:
            print(f"Error loading {file_path} with unstructured: {e}")

        return documents

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_all(self):
        """
        Scans raw_dir, loads each file, applies the chosen chunking
        strategy, and writes results to processed_dir.
        """
        files = os.listdir(self.raw_dir)
        print(f"Found {len(files)} file(s) in {self.raw_dir}")
        print(f"Chunking strategy: {self.chunking_strategy}\n")

        for filename in files:
            file_path = os.path.join(self.raw_dir, filename)
            ext = os.path.splitext(filename)[1].lower()

            print(f"Processing {filename}...")

            if ext == ".pdf":
                docs = self.load_pdf(file_path)
            elif ext in [".html", ".htm", ".md"]:
                docs = self.load_unstructured(file_path)
            else:
                print(f"  Skipping unsupported format: {ext}")
                continue

            # ---- Chunking stage ----
            # Concatenate all loaded sections into one text block per file,
            # then chunk the combined text with provenance metadata.
            full_text = "\n\n".join(doc["content"] for doc in docs)

            # Build base metadata from the first document section
            base_meta = docs[0]["metadata"].copy() if docs else {"source": filename}

            chunks = chunk_document(
                text=full_text,
                strategy=self.chunking_strategy,
                base_metadata=base_meta,
            )

            # ---- Persist chunks ----
            base_name = os.path.splitext(filename)[0]
            for i, chunk in enumerate(chunks):
                output_base = f"{base_name}_{self.chunking_strategy}_chunk_{i}"

                # Save processed text
                text_path = os.path.join(self.processed_dir, f"{output_base}.txt")
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(chunk["content"])

                # Save metadata (includes chunking_strategy provenance)
                meta_path = os.path.join(self.processed_dir, f"{output_base}.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(chunk["metadata"], f, indent=4)

            print(f"  -> {len(chunks)} chunks saved ({self.chunking_strategy}).")

        print("\nDone.")


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load documents and chunk them with a chosen strategy."
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="structure_aware",
        help="Chunking strategy to use (default: structure_aware)",
    )
    args = parser.parse_args()

    loader = DocumentLoader(chunking_strategy=args.strategy)
    loader.process_all()
