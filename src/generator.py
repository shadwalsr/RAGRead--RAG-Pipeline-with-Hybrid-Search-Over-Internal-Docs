"""
Phase 3: Generation and Citation Layer

Takes the top results from the Hybrid Retriever and uses a strict
"Grounded Generation Prompt" to force the LLM to answer using ONLY 
the provided context, while requiring numbered citations.
"""

import os
from typing import List, Dict, Any
from google import genai

class RAGGenerator:
    """
    Handles the final LLM generation step of the RAG pipeline.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GOOGLE_API_KEY to use the Generator.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates a grounded answer with citations based strictly on the context chunks.
        """
        if not context_chunks:
            return "I could not find any relevant information to answer your question."
            
        # 1. Build Numbered Context Blocks
        # We format them as "Context Block X" so the LLM knows exactly what number to cite.
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            # i+1 to make it 1-indexed for the prompt
            context_text += f"Context Block {i+1}:\n{chunk['content']}\n\n"
            
        # 2. Design the Grounded Generation Prompt
        prompt = f"""You are a helpful, accurate, and precise assistant. 
Your task is to answer the user's question based strictly and exclusively on the provided context blocks.

STRICT INSTRUCTIONS:
1. You must answer the question ONLY using the facts from the provided Context Blocks. 
2. If the Context Blocks do not contain enough information to answer the question, clearly state: "I don't have enough information to answer that based on the provided documents." Do NOT use your general knowledge.
3. Every factual claim you make MUST be followed by a citation to the specific Context Block it came from.
4. Use bracketed references for citations, matching the Context Block number (e.g., [1], [2]). 
5. Do not hallucinate, invent, or assume any information outside of the context.

CONTEXT BLOCKS:
{context_text}

USER QUESTION:
{query}

ANSWER:
"""

        # 3. Generate the response
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text
            
        except Exception as e:
            return f"Error generating response: {e}"

    def verify_citations(self, answer: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        The 'Quality Layer'. Parses generated answers for citations like [1],
        and uses an LLM-as-a-Judge to verify if the cited chunk actually supports the claim.
        Flags unsupported citations with a warning.
        """
        import re
        
        print("    [Verifier] Checking citations...")
        
        # Split text into sentences for granular claim checking
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        verified_answer = answer
        
        for sentence in sentences:
            # Find all citations like [1] or [2]
            cites = re.findall(r'\[(\d+)\]', sentence)
            if not cites:
                continue
                
            original_sentence = sentence
            new_sentence = sentence
            
            for cite in set(cites):
                cite_idx = int(cite) - 1
                if 0 <= cite_idx < len(context_chunks):
                    chunk_content = context_chunks[cite_idx]["content"]
                    
                    prompt = f"""You are a strict citation verifier.
Given a CLAIM and a CONTEXT chunk, determine if the CONTEXT provides enough information to fully support the CLAIM.

CLAIM: "{sentence}"

CONTEXT: "{chunk_content}"

Does the CONTEXT fully support the CLAIM? Answer strictly with a single word: YES or NO.
"""
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=prompt
                        )
                        result = response.text.strip().upper()
                        
                        # If the LLM Judge says NO, flag it
                        if not result.startswith("YES"):
                            print(f"      -> Flagged [Chunk {cite}] for unsupported claim: '{sentence[:50]}...'")
                            new_sentence = new_sentence.replace(f"[{cite}]", f"[{cite} ⚠️ UNVERIFIED]")
                            
                    except Exception as e:
                        print(f"      -> API Error during verification: {e}")
                        
            # Apply any flagged warnings back to the main answer
            if original_sentence != new_sentence:
                verified_answer = verified_answer.replace(original_sentence, new_sentence)
                
        return verified_answer

    def generate_comprehensive_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Phase 3, Steps 3 & 4: Generates a grounded answer alongside a Confidence Report,
        and gracefully handles the 'I Don't Know' case with a structured refusal.
        Uses structured JSON output to force the LLM to evaluate its own context.
        """
        import json
        from google.genai import types

        if not context_chunks:
            return {
                "status": "refused",
                "answer": "I don't have any documents to search.",
                "report": {"retrieval_confidence": 0, "completeness": "None"}
            }

        context_text = ""
        for i, chunk in enumerate(context_chunks):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            context_text += f"Context Block {i+1} (Source: {source}):\n{chunk['content']}\n\n"

        prompt = f"""You are an elite enterprise AI assistant.
Evaluate the CONTEXT BLOCKS against the USER QUERY.

INSTRUCTIONS:
1. Evaluate if the context contains enough information to answer the query. Assign a 'retrieval_confidence_score' from 0 to 10.
2. If the score is < 5, set 'can_answer' to false. Provide a 'structured_refusal' explaining what you found, what is missing, and what type of documents the user should check.
3. If the score is >= 5, set 'can_answer' to true. Generate the 'answer' using strict bracketed citations (e.g., [1]) pointing to the Context Blocks.
4. Evaluate 'answer_completeness': Does your answer address all parts of the user's query?

Respond strictly in this JSON schema:
{{
  "retrieval_confidence_score": int,
  "can_answer": bool,
  "structured_refusal": {{
    "what_was_found": str,
    "what_is_missing": str,
    "suggested_documents": str
  }},
  "answer": str,
  "answer_completeness": str
}}

CONTEXT BLOCKS:
{context_text}

USER QUERY:
{query}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            result = json.loads(response.text)
            
            # Step 2: The Citation Quality Layer (Coverage)
            if result.get("can_answer") and result.get("answer"):
                # Run the LLM-as-a-judge verifier on the generated answer
                raw_answer = result["answer"]
                verified_answer = self.verify_citations(raw_answer, context_chunks)
                result["verified_answer"] = verified_answer
                
                # Calculate Citation Coverage stats
                import re
                total_cites = len(re.findall(r'\[\d+\]', raw_answer))
                failed_cites = len(re.findall(r'⚠️ UNVERIFIED', verified_answer))
                
                if total_cites > 0:
                    coverage_pct = round(((total_cites - failed_cites) / total_cites) * 100)
                else:
                    coverage_pct = 100 # No citations used, so technically 100% valid
                    
                result["citation_coverage_pct"] = coverage_pct
                
            return result
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

# For testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    from retriever import HybridRetriever
    import time
    import json
    
    load_dotenv()
    
    # Query 1: Should be answerable
    query1 = "What role did Shadwal have at Whyschool Academy?"
    # Query 2: Unanswerable (testing Structured Refusal)
    query2 = "What is Shadwal's favorite ice cream flavor?"
    
    retriever = HybridRetriever()
    generator = RAGGenerator()
    
    for q in [query1, query2]:
        print(f"\n{'='*50}\nTESTING QUERY: '{q}'\n{'='*50}")
        
        chunks = retriever.retrieve(q, top_k=3, use_reranker=False)
        print(f"Retrieved {len(chunks)} chunks.")
        
        response_data = generator.generate_comprehensive_response(q, chunks)
        
        # We might hit rate limits, so handle errors cleanly
        if response_data.get("status") == "error":
            print(f"API Error: {response_data.get('message')}")
            time.sleep(15) # cooldown
            continue
            
        print(f"\n--- CONFIDENCE REPORT ---")
        print(f"Retrieval Confidence: {response_data.get('retrieval_confidence_score')}/10")
        print(f"Answer Completeness:  {response_data.get('answer_completeness')}")
        
        if not response_data.get("can_answer"):
            print("\n--- STRUCTURED REFUSAL ---")
            refusal = response_data.get("structured_refusal", {})
            print(f"Found:   {refusal.get('what_was_found')}")
            print(f"Missing: {refusal.get('what_is_missing')}")
            print(f"Suggest: {refusal.get('suggested_documents')}")
        else:
            print(f"Citation Coverage:    {response_data.get('citation_coverage_pct')}%")
            print("\n--- FINAL VERIFIED ANSWER ---")
            print(response_data.get("verified_answer"))
            
        time.sleep(15) # prevent API rate limiting between queries
