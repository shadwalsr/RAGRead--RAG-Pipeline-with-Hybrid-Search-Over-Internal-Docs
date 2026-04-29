"""
Interactive RAG Chat Interface
Ties together the Hybrid Retriever and the Grounded Generator.
"""

import os
import sys
import time
from dotenv import load_dotenv
from retriever import HybridRetriever
from generator import RAGGenerator

def main():
    load_dotenv()
    
    print("\n" + "="*60)
    print("  ----------RAGREAD- RAG INTERACTIVE CHAT ENGINE----------")
    print("  Hybrid Search + RRF + Reranker + Grounded Generation")
    print("="*60)
    
    # Initialize components
    try:
        retriever = HybridRetriever()
        generator = RAGGenerator()
    except Exception as e:
        print(f"❌ Error initializing engine: {e}")
        return

    print("\nSystem ready. Type 'exit' or 'quit' to stop.")
    
    while True:
        query = input("\n👤 User: ").strip()
        
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if not query:
            continue

        print("\n🔍 Thinking...")
        start_time = time.time()
        
        try:
            # 1. RETRIEVAL PHASE
            # We use alpha=0.5 (balanced) and use_reranker=True for max accuracy
            chunks = retriever.retrieve(query, top_k=5, alpha=0.5, use_reranker=True)
            
            # 2. GENERATION & VERIFICATION PHASE
            response_data = generator.generate_comprehensive_response(query, chunks)
            
            if response_data.get("status") == "error":
                print(f"❌ API Error: {response_data.get('message')}")
                continue
                
            elapsed = time.time() - start_time
            
            # 3. DISPLAY RESULTS
            print("-" * 40)
            print(f"📊 CONFIDENCE REPORT (Processed in {elapsed:.1f}s)")
            print(f"Confidence: {response_data.get('retrieval_confidence_score')}/10")
            print(f"Coverage:   {response_data.get('citation_coverage_pct', 0)}%")
            print("-" * 40)
            
            if not response_data.get("can_answer"):
                print("\n🤖 AI (Refusal):")
                refusal = response_data.get("structured_refusal", {})
                print(f"I couldn't find enough info. {refusal.get('what_is_missing')}")
                print(f"\n💡 Suggestion: {refusal.get('suggested_documents')}")
            else:
                print("\n🤖 AI Answer:")
                print(response_data.get("verified_answer"))
            
            # Source Traceability
            print("\n📚 Sources Used:")
            seen_sources = set()
            for i, chunk in enumerate(chunks):
                source = chunk["metadata"].get("source", "Unknown Document")
                if source not in seen_sources:
                    print(f"   [{i+1}] {source}")
                    seen_sources.add(source)
                    
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            print("Wait a moment and try again (it might be a rate limit).")

if __name__ == "__main__":
    main()
