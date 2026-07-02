import json
import gzip
from pathlib import Path

def precompute_static_scores():
    """
    Calculates the baseline, non-semantic scores for all 100k candidates offline.
    This includes Behavioral signaling, Job-hopper penalties, and Product-company boosts.
    The online script will only need to add the cosine similarity score to this baseline.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_data_path = base_dir / "data" / "raw" / "candidates.jsonl"
    precomputed_dir = base_dir / "data" / "precomputed"
    rubric_path = precomputed_dir / "recruiter_rubric.json"
    output_file = precomputed_dir / "candidate_static_scores.json"

    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric = json.load(f)

    b_weights = rubric["stage3_behavioral_signals"]["weights"]
    exp_scoring = rubric["stage2_experience_scoring"]
    
    candidate_scores = {}

    print("Processing static features from raw candidate data...")
    with open(raw_data_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            
            signals = cand.get("redrob_signals", {})
            
            gh_raw = signals.get("github_activity_score", 0)
            gh_score = max(0, gh_raw) / 100.0  
            
            response_rate = signals.get("recruiter_response_rate", 0)
            open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
            
            behavioral_score = (
                (response_rate * b_weights["recruiter_response_rate"]) +
                (gh_score * b_weights["github_activity_score"]) +
                (open_to_work * b_weights["open_to_work_flag"])
            )

            notice_days = signals.get("notice_period_days", 30)
            if notice_days > rubric["stage3_behavioral_signals"]["max_acceptable_notice_period"]:
                behavioral_score *= 0.5  

            years_exp = cand.get("profile", {}).get("years_of_experience", 0)
            company_count = len(cand.get("career_history", []))
            
            exp_modifier = 1.0
            if company_count > 0 and (years_exp / company_count) < exp_scoring["job_hopper_threshold_years"]:
                exp_modifier *= exp_scoring["job_hopper_penalty"]
            
            candidate_scores[cid] = {
                "behavioral_score": round(behavioral_score, 4),
                "experience_modifier": round(exp_modifier, 4),
                "total_years_exp": years_exp
            }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(candidate_scores, f, indent=4)

    print(f"Precomputed static scores for {len(candidate_scores)} candidates.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    precompute_static_scores()