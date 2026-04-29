import os
from dotenv import load_dotenv
from storage import HybridStore
from loader import DocumentLoader
from chunker import split_text

def run_ingest(strategy="structural"):
    # entry point for indexing all raw files
    load_dotenv()
    
    db = HybridStore()
    loader = DocumentLoader()
    
    # 1. pull everything from the raw folder
    print("scaning data/raw...")
    raw_docs = loader.get_all_docs()
    
    if not raw_docs:
        print("nothing to ingest!")
        return

    # 2. group by source and chunk
    # we group because split_text expects the full doc string
    sources = {}
    for d in raw_docs:
        src = d["metadata"]["source"]
        if src not in sources:
            sources[src] = []
        sources[src].append(d["content"])
        
    for src, contents in sources.items():
        print(f"processing: {src}...")
        full_text = "\n\n".join(contents)
        
        chunks = split_text(full_text, src, strategy=strategy)
        
        # 3. push to db
        stats = db.ingest_chunks(chunks)
        print(f"  done: {stats['added']} added, {stats['duplicates_skipped']} skipped")

if __name__ == "__main__":
    # change this to "fixed" or "semantic" if you want to experiment
    run_ingest(strategy="structural")
