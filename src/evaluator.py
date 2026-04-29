import os
import json
import time
from typing import List, Dict, Any
from google import genai


class RAGEvaluator:
    def __init__(self):
        # reuse the same retriever and generator across all test runs
        from retriever import HybridRetriever
        from generator import RAGGenerator
        self.retriever = HybridRetriever()
        self.gen = RAGGenerator()
        self._judge_client = genai.Client()

    def score_answer(self, question: str, ai_answer: str, golden: str) -> Dict[str, Any]:
        # simple llm-as-a-judge: compare ai output to our hand-written ground truth
        prompt = f"""Compare the AI Answer to the Golden Answer for this question.
Score from 1-5 where 5 = perfect match, 1 = completely wrong.

Question: {question}
Golden Answer: {golden}
AI Answer: {ai_answer}

Return JSON only:
{{"score": int, "reasoning": "one sentence"}}"""

        try:
            resp = self._judge_client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            return json.loads(resp.text)
        except Exception as e:
            return {"score": 0, "reasoning": str(e)}

    def run_benchmark(self, dataset_path: str, out_path: str, limit: int = None, strategy: str = None):
        import pandas as pd

        if not os.path.exists(dataset_path):
            print(f"dataset not found: {dataset_path}")
            return

        with open(dataset_path, "r") as f:
            qa_pairs = json.load(f)

        if limit:
            qa_pairs = qa_pairs[:limit]

        strat_label = strategy or "all"
        print(f"running {len(qa_pairs)} questions | strategy: {strat_label}")

        rows = []

        for i, item in enumerate(qa_pairs):
            q = item["question"]
            gold = item["answer"]
            q_type = item.get("type", "fact")

            print(f"[{i+1}/{len(qa_pairs)}] {q[:55]}...")

            t0 = time.time()

            # retrieve then generate
            chunks = self.retriever.retrieve(q, top_k=3, use_reranker=False, strategy=strategy)
            gen_result = self.gen.generate_comprehensive_response(q, chunks)
            ai_ans = gen_result.get("verified_answer") or gen_result.get("answer", "")

            # small pause before calling the judge to avoid 429s
            time.sleep(2)
            score_result = self.score_answer(q, ai_ans, gold)

            rows.append({
                "question": q,
                "type": q_type,
                "strategy": strat_label,
                "golden_answer": gold,
                "ai_answer": ai_ans,
                "correctness_score": score_result.get("score"),
                "reasoning": score_result.get("reasoning"),
                "retrieval_confidence": gen_result.get("retrieval_confidence_score"),
                "citation_coverage": gen_result.get("citation_coverage_pct"),
                "latency_s": round(time.time() - t0, 2)
            })

            # 15 rpm free tier = ~4s per request but we're being safe with 10
            time.sleep(10)

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"done. saved to {out_path}")
        return df


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    ev = RAGEvaluator()
    ev.run_benchmark(
        dataset_path="data/eval/golden_qa.json",
        out_path="data/eval/evaluation_report.csv",
        limit=5
    )
