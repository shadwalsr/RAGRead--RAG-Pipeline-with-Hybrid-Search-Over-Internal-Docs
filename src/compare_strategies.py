import os
import pandas as pd
from dotenv import load_dotenv
from evaluator import RAGEvaluator

def run_strategy_bench():
    # compare our three chunking strategies head-to-head
    load_dotenv()
    ev = RAGEvaluator()
    
    strategies = ["fixed", "structural", "semantic"]
    
    for s in strategies:
        print(f"\n--- TESTING STRATEGY: {s} ---")
        out_path = f"data/eval/results_{s}.csv"
        
        # run a small subset (3 questions) for speed
        # change limit=None for full run
        ev.run_benchmark(
            dataset_path="data/eval/golden_qa.json",
            out_path=out_path,
            limit=3,
            strategy=s
        )
        
    # combine results for final report
    all_res = []
    for s in strategies:
        p = f"data/eval/results_{s}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            all_res.append(df)
            
    if all_res:
        final_df = pd.concat(all_res)
        final_df.to_csv("data/eval/strategy_comparison.csv", index=False)
        print("\nFinal comparison saved to data/eval/strategy_comparison.csv")

if __name__ == "__main__":
    run_strategy_bench()
