import os
import pickle
import chromadb
from rank_bm25 import BM25Okapi

# run this if the keyword index gets out of sync with the vector store
def reset_bm25():
    db_path = "data/vectorstore"
    bm25_path = "data/bm25_index.pkl"
    
    print("rebuilding bm25 from chroma documents...")
    
    # 1. pull every doc from chroma
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection("rag_documents")
    all_docs = col.get()
    
    ids = all_docs["ids"]
    texts = all_docs["documents"]
    
    # 2. tokenize
    corpus = [t.lower().split() for t in texts]
    
    # 3. save
    with open(bm25_path, "wb") as f:
        pickle.dump({"corpus": corpus, "doc_ids": ids}, f)
        
    print(f"done. indexed {len(ids)} chunks.")

if __name__ == "__main__":
    reset_bm25()
