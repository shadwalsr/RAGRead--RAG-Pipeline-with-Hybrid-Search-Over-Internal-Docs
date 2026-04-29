"""
Head-to-Head Chunking Strategy Comparison
Runs the evaluation loop across all three strategies and generates a comparison report.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from evaluator import RAGEvaluator

def run_comparison():
    load_dotenv()
    
    evaluator = RAGEvaluator()
    strategies = ["fixed", "structural", "semantic"]
    
    # We'll test with a small limit (e.g. 3 questions) to show the proof of concept
    # without hitting massive rate limits or waiting hours.
    limit = 3
    dataset_path = 'data/eval/golden_qa.json'
    
    all_summaries = []
    
    print("="*60)
    print("  HEAD-TO-HEAD CHUNKING STRATEGY COMPARISON")
    print("="*60)
    
    for strategy in strategies:
        output_path = f'data/eval/eval_{strategy}.csv'
        results_df = evaluator.run_benchmark(
            dataset_path=dataset_path,
            output_path=output_path,
            limit=limit,
            strategy=strategy
        )
        
        # Calculate summary for this strategy
        summary = {
            "Strategy": strategy.capitalize(),
            "Avg Correctness": round(results_df['correctness_score'].mean(), 2),
            "Avg Confidence": round(results_df['retrieval_confidence'].mean(), 2),
            "Avg Citation Coverage": round(results_df['citation_coverage'].mean(), 2),
            "Avg Latency (s)": round(results_df['latency'].mean(), 2)
        }
        all_summaries.append(summary)
        
        # Cooldown between strategies
        print("\nCooldown for 30s to reset API quotas...")
        import time
        time.sleep(30)

    # Generate the Comparison Table
    comparison_df = pd.DataFrame(all_summaries)
    print("\n" + "="*60)
    print("  FINAL COMPARISON RESULTS")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("="*60)
    
    # Save comparison to CSV
    comparison_df.to_csv('data/eval/strategy_comparison.csv', index=False)
    
    # Determine the winner
    winner = comparison_df.iloc[comparison_df['Avg Correctness'].idxmax()]
    print(f"\nWINNER: {winner['Strategy']} Chunking Strategy!")
    print(f"It achieved the highest Correctness score of {winner['Avg Correctness']}/5.0")

if __name__ == "__main__":
    run_comparison()
