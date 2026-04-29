"""
Phase 4: Evaluation Layer
Automated Evaluation Harness for RAG Performance.
"""

import os
import json
import time
import pandas as pd
from typing import List, Dict, Any
from retriever import HybridRetriever
from generator import RAGGenerator

class RAGEvaluator:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = RAGGenerator()
        
    def evaluate_correctness(self, question: str, ai_answer: str, golden_answer: str) -> Dict[str, Any]:
        """
        LLM-as-Judge to compare AI answer against the Ground Truth.
        """
        from google import genai
        
        client = genai.Client()
        prompt = f"""You are an objective AI evaluator. 
Compare the AI Answer against the Golden Answer for the given Question.

Question: {question}
Golden Answer: {golden_answer}
AI Answer: {ai_answer}

Grade the AI Answer on a scale of 1 to 5 based on:
1. Accuracy: Does it match the Golden Answer facts?
2. Completeness: Does it include all necessary details?

Return strictly in JSON format:
{{
  "score": int,
  "reasoning": "Brief explanation of the score"
}}
"""
        try:
            response = client.models.generate_content(
                model='gemini-flash-latest',
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            return json.loads(response.text)
        except Exception as e:
            return {"score": 0, "reasoning": f"Error during evaluation: {e}"}

    def run_benchmark(self, dataset_path: str, output_path: str, limit: int = None, strategy: str = None):
        """
        Runs the full RAG pipeline on the dataset and scores the results.
        strategy: If provided, filters retrieval to only use chunks from this strategy.
        """
        if not os.path.exists(dataset_path):
            print(f"Error: {dataset_path} not found.")
            return

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        if limit:
            data = data[:limit]

        results = []
        strat_name = strategy if strategy else "all"
        print(f"Starting Evaluation on {len(data)} pairs (Strategy: {strat_name})...")

        for i, pair in enumerate(data):
            question = pair['question']
            golden_answer = pair['answer']
            q_type = pair.get('type', 'fact')

            print(f"[{i+1}/{len(data)}] Testing: {question[:50]}...")
            
            try:
                # 1. RETRIEVE
                start_time = time.time()
                # Pass strategy to retrieval
                chunks = self.retriever.retrieve(question, top_k=3, use_reranker=False, strategy=strategy)
                
                # 2. GENERATE COMPREHENSIVE RESPONSE
                response_data = self.generator.generate_comprehensive_response(question, chunks)
                ai_answer = response_data.get("verified_answer", response_data.get("answer", ""))
                
                # 3. SCORE CORRECTNESS (LLM-AS-JUDGE)
                # Note: We pause slightly to avoid 429 rate limits
                time.sleep(2)
                eval_report = self.evaluate_correctness(question, ai_answer, golden_answer)
                
                results.append({
                    "question": question,
                    "type": q_type,
                    "strategy": strat_name,
                    "golden_answer": golden_answer,
                    "ai_answer": ai_answer,
                    "correctness_score": eval_report.get("score"),
                    "reasoning": eval_report.get("reasoning"),
                    "retrieval_confidence": response_data.get("retrieval_confidence_score"),
                    "citation_coverage": response_data.get("citation_coverage_pct"),
                    "latency": round(time.time() - start_time, 2)
                })

            except Exception as e:
                print(f"   Error on item {i+1}: {e}")
                
            # Wait between items to stay within 15 RPM free tier limits
            time.sleep(10)

        # Save results to CSV for analysis
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\nEvaluation Complete! Report saved to: {output_path}")
        return df

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    evaluator = RAGEvaluator()
    # We test with a limit of 5 to avoid long wait times during the first run
    evaluator.run_benchmark(
        dataset_path='data/eval/golden_qa.json',
        output_path='data/eval/evaluation_report.csv',
        limit=5
    )
