import time
from pathlib import Path
import sys

from src.online_ranker.purge import run_purge
from src.online_ranker.ner_heuristics import run_ner_heuristics
from src.online_ranker.vector_slicing import run_vector_scoring
from src.online_ranker.behavioral_blend import run_behavioral_blend
from src.online_ranker.reasoning import generate_reasoning_and_export

def main():
    start_time = time.time()
    print("Initializing Stage 0: Data Paths...")
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    cands_path = base_dir / "data" / "raw" / "candidates.jsonl"
    rubric_path = base_dir / "data" / "precomputed" / "recruiter_rubric.json"
    team_name = "Compilation_Error" # Update this to your actual team name

    try:
        # STAGE 0: Structural Purge (Filter out Honeypots & IT Firms)
        valid_ids = run_purge(cands_path, rubric_path)
        
        # STAGE 1: NER Heuristics (Extract Tech & Filter to Top 1500)
        top_candidates_map = run_ner_heuristics(valid_ids, top_k=1500)
        
        # STAGE 2: Vector Slicing (Semantic Similarity via pre-computed embeddings)
        scored_candidates_map = run_vector_scoring(top_candidates_map)
        
        # STAGE 3: Behavioral Blend (Static metrics & Sequence-wise normalization)
        final_top_100 = run_behavioral_blend(scored_candidates_map, top_n=100)
        
        # STAGE 4: Deterministic Reasoning & CSV Export
        generate_reasoning_and_export(final_top_100, team_name=team_name)

    except Exception as e:
        print(f"CRITICAL PIPELINE FAILURE: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"⏱Total Execution Time: {elapsed:.2f} seconds.")
    
    if elapsed > 300:
        print("WARNING: Execution exceeded the 5-minute sandbox limit!")
    else:
        print("SUCCESS: Execution safely under the 5-minute limit.")

if __name__ == "__main__":
    main()