import os
import re
import json
import time
from typing import List, Dict, Any
from google import genai
from google.genai import types
from rate_limiter import call_with_retry


class RAGGenerator:

    def __init__(self, model_name: str = "gemini-flash-latest"):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model_name

    def _build_context_block(self, chunks: List[Dict[str, Any]], include_source=False) -> str:
        # formats the chunks into a numbered list the LLM can cite
        ctx = ""
        for i, chunk in enumerate(chunks):
            if include_source:
                src = chunk.get("metadata", {}).get("source", "unknown")
                ctx += f"Context Block {i+1} (from {src}):\n{chunk['content']}\n\n"
            else:
                ctx += f"Context Block {i+1}:\n{chunk['content']}\n\n"
        return ctx

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant documents found."

        ctx = self._build_context_block(chunks)

        prompt = f"""You are a helpful assistant. Answer the question using only the context below.
Every claim must cite the context block it came from using brackets like [1], [2].
If the context doesn't contain the answer, say so - don't guess.

CONTEXT:
{ctx}

QUESTION: {query}

ANSWER:"""

        try:
            resp = call_with_retry(self.client.models.generate_content, model=self.model, contents=prompt)
            return resp.text
        except Exception as e:
            return f"Generation error: {e}"

    def verify_citations(self, answer: str, chunks: List[Dict[str, Any]]) -> str:
        # goes sentence by sentence and checks that each [n] citation is actually supported
        # this is the "self-auditing" part - catches hallucinated citations
        print("  checking citations...")

        sentences = re.split(r'(?<=[.!?])\s+', answer)
        checked = answer

        for sent in sentences:
            cited_nums = re.findall(r'\[(\d+)\]', sent)
            if not cited_nums:
                continue

            updated_sent = sent

            for num in set(cited_nums):
                idx = int(num) - 1
                if not (0 <= idx < len(chunks)):
                    continue

                chunk_text = chunks[idx]["content"]

                judge_prompt = f"""Does the following CONTEXT support the CLAIM?
Answer with one word only: YES or NO.

CLAIM: "{sent}"
CONTEXT: "{chunk_text}"
"""
                try:
                    resp = call_with_retry(self.client.models.generate_content, model=self.model, contents=judge_prompt)
                    verdict = resp.text.strip().upper()

                    if not verdict.startswith("YES"):
                        print(f"  flagged [{num}]: '{sent[:60]}...'")
                        updated_sent = updated_sent.replace(f"[{num}]", f"[{num} UNVERIFIED]")

                except Exception as e:
                    # if the judge call fails just skip - better than crashing the whole response
                    print(f"  verifier error on [{num}]: {e}")

            if updated_sent != sent:
                checked = checked.replace(sent, updated_sent)

        return checked

    def generate_comprehensive_response(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {
                "status": "refused",
                "can_answer": False,
                "answer": "No documents available.",
                "retrieval_confidence_score": 0
            }

        ctx = self._build_context_block(chunks, include_source=True)

        # asking for JSON output directly - cleaner than parsing free text
        prompt = f"""You are an AI assistant. Read the context blocks and answer the user's query.

Rules:
1. Score your confidence in the retrieved context from 0-10 as retrieval_confidence_score
2. If score < 5, set can_answer to false and fill in structured_refusal
3. If score >= 5, write an answer with bracketed citations like [1], [2]
4. Fill in answer_completeness: did you cover everything in the query?

Return valid JSON matching this schema exactly:
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

CONTEXT:
{ctx}

QUERY: {query}"""

        try:
            resp = call_with_retry(
                self.client.models.generate_content,
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(resp.text)

            # run the citation verifier if there's an answer
            if result.get("can_answer") and result.get("answer"):
                raw = result["answer"]
                verified = self.verify_citations(raw, chunks)
                result["verified_answer"] = verified

                # count how many citations survived the verification pass
                total = len(re.findall(r'\[\d+\]', raw))
                failed = len(re.findall(r'UNVERIFIED', verified))
                result["citation_coverage_pct"] = round(((total - failed) / total) * 100) if total > 0 else 100

            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    from dotenv import load_dotenv
    from retriever import HybridRetriever

    load_dotenv()

    retriever = HybridRetriever()
    generator = RAGGenerator()

    test_questions = [
        "What role did Shadwal have at Whyschool Academy?",
        "What is Shadwal's favorite ice cream flavor?"  # this one should get refused
    ]

    for q in test_questions:
        print(f"\n--- {q} ---")
        chunks = retriever.retrieve(q, top_k=3, use_reranker=False)
        out = generator.generate_comprehensive_response(q, chunks)

        if out.get("status") == "error":
            print(f"error: {out['message']}")
        elif not out.get("can_answer"):
            refusal = out.get("structured_refusal", {})
            print(f"refused - missing: {refusal.get('what_is_missing')}")
        else:
            print(f"confidence: {out['retrieval_confidence_score']}/10")
            print(f"coverage: {out.get('citation_coverage_pct')}%")
            print(out.get("verified_answer"))

        time.sleep(15)
