import os
import time
from dotenv import load_dotenv
from retriever import HybridRetriever
from generator import RAGGenerator

def main():
    load_dotenv()
    
    # setup the brain
    r = HybridRetriever()
    g = RAGGenerator()
    
    print("\n--- RAGRead CLI Chat ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        q = input("Question: ").strip()
        if not q or q.lower() in ["exit", "quit", "q"]:
            break
            
        t0 = time.time()
        
        # 1. search
        print(" searching...")
        chunks = r.retrieve(q, top_k=3, use_reranker=True)
        
        # 2. answer
        print(" thinking...")
        # we use the 'comprehensive' one because it does verification automatically
        res = g.generate_comprehensive_response(q, chunks)
        
        dt = round(time.time() - t0, 2)
        
        print(f"\n({dt}s) AI Answer:")
        if res.get("can_answer"):
            print(res.get("verified_answer"))
            print(f"\nConfidence: {res.get('retrieval_confidence_score')}/10 | Coverage: {res.get('citation_coverage_pct')}%")
        else:
            # handle the refusal case
            ref = res.get("structured_refusal", {})
            print("Sorry, I don't have enough info.")
            print(f"Missing: {ref.get('what_is_missing')}")
            
        print("-" * 50)
        # small delay so the console doesn't feel frantic
        time.sleep(0.5)

if __name__ == "__main__":
    main()
