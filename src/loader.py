import os
import re
import fitz # pymupdf - the best one for speed
from typing import List, Dict, Any
from unstructured.partition.auto import partition

class DocumentLoader:
    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = raw_dir

    def clean_text(self, text):
        # basic cleaning to remove excessive whitespace and weird characters
        if not text:
            return ""
        # replace multiple newlines/spaces with single ones
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_pdf(self, path):
        # using fitz because it's way faster than pypdf
        docs = []
        try:
            doc = fitz.open(path)
            fname = os.path.basename(path)
            
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    docs.append({
                        "content": self.clean_text(text),
                        "metadata": {
                            "source": fname,
                            "page": i + 1,
                            "ext": "pdf"
                        }
                    })
        except Exception as e:
            print(f"pdf load failed for {path}: {e}")
        return docs

    def load_other(self, path):
        # handles md, html, txt using unstructured
        # it's a bit heavy but very robust
        docs = []
        try:
            elements = partition(filename=path)
            fname = os.path.basename(path)
            
            # just group everything into one block for now
            # could split by headings if we wanted to be fancy
            full_text = "\n".join([str(e) for e in elements])
            if full_text.strip():
                docs.append({
                    "content": self.clean_text(full_text),
                    "metadata": {
                        "source": fname,
                        "ext": os.path.splitext(path)[1][1:]
                    }
                })
        except Exception as e:
            print(f"unstructured load failed for {path}: {e}")
        return docs

    def get_all_docs(self, limit=None):
        # scans the raw dir and returns list of chunks
        # this is what we call from the ingestion script
        all_files = os.listdir(self.raw_dir)
        if limit:
            all_files = all_files[:limit]
            
        results = []
        for f in all_files:
            if f.startswith("."): continue
            
            p = os.path.join(self.raw_dir, f)
            ext = os.path.splitext(f)[1].lower()
            
            print(f"loading: {f}")
            if ext == ".pdf":
                results.extend(self.load_pdf(p))
            else:
                results.extend(self.load_other(p))
        return results
