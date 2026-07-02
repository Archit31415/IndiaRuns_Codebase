import json
import gzip
from pathlib import Path

def run_purge(candidates_file_path: Path, rubric_path: Path) -> set:
    """
    Executes the O(N) structural purge over the raw candidate pool.
    
    This script is the first step of the 5-minute online ranking phase. 
    It eliminates honeypots, keyword stuffers, and candidates who violate 
    the strict hard-filters defined in the recruiter rubric.
    
    Returns:
        valid_candidate_ids (set): A set of candidate IDs that survive the purge.
    """
    print("[Stage 0] Starting Structural Purge & Honeypot Detection...")
    
    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric = json.load(f)
        
    hard_filters = rubric.get("stage0_hard_filters", {})
    min_exp = hard_filters.get("min_years_experience", 4.0)
    blacklisted_firms = set([
        firm.lower() for firm in hard_filters.get("blacklisted_companies_if_exclusive", [])
    ])
    incompatible_domains = set([
        dom.lower() for dom in hard_filters.get("incompatible_domains", [])
    ])

    valid_candidate_ids = set()
    total_processed = 0
    honeypots_caught = 0
    filter_drops = 0

    open_func = gzip.open if str(candidates_file_path).endswith('.gz') else open
    mode = "rt" if str(candidates_file_path).endswith('.gz') else "r"

    with open_func(candidates_file_path, mode, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
                
            total_processed += 1
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            
            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            skills = cand.get("skills", [])
            
            claimed_years_exp = profile.get("years_of_experience", 0.0)

            is_honeypot = False
            
            # Honeypot detection
            # Trap A: "Expert proficiency with 0 years used"
            for skill in skills:
                if skill.get("proficiency") == "expert" and skill.get("duration_months", 1) == 0:
                    is_honeypot = True
                    break
            
            # Trap B: "Impossible Timelines" - Claiming 8 years exp, but sum of all job durations is < 2 years
            total_career_months = sum(job.get("duration_months", 0) for job in career)
            if claimed_years_exp > 0 and (total_career_months / 12.0) < (claimed_years_exp * 0.5):
                is_honeypot = True

            if is_honeypot:
                honeypots_caught += 1
                continue

            # Rubric filters
            # Rule A: Does not meet absolute minimum seniority
            if claimed_years_exp < min_exp:
                filter_drops += 1
                continue
            
            # Rule B: Pure Service Company Career
            # Candidate is dropped if EVERY company they have worked for is blacklisted
            companies_worked = [job.get("company", "").lower() for job in career]
            if companies_worked and all(any(bl_firm in comp for bl_firm in blacklisted_firms) for comp in companies_worked):
                filter_drops += 1
                continue
                
            # Rule C: Exclusive Incompatible Domain
            # Drops purely CV/Speech/Robotics researchers with 0 NLP/Product history
            job_titles = [job.get("title", "").lower() for job in career]
            if job_titles and all(any(bad_dom in title for bad_dom in incompatible_domains) for title in job_titles):
                filter_drops += 1
                continue

            # Passed all checks
            valid_candidate_ids.add(cid)

    print(f"   - Total Processed: {total_processed}")
    print(f"   - Honeypots Exterminated: {honeypots_caught}")
    print(f"   - JD Hard-Filter Drops: {filter_drops}")
    print(f"[Stage 0] Candidates surviving purge: {len(valid_candidate_ids)}")
    
    return valid_candidate_ids

if __name__ == "__main__":
    # Local manual testing execution
    base_dir = Path(__file__).resolve().parent.parent.parent
    cands_path = base_dir / "data" / "raw" / "candidates.jsonl"
    rubric_path = base_dir / "data" / "precomputed" / "recruiter_rubric.json"
    if cands_path.exists() and rubric_path.exists():
        survivors = run_purge(cands_path, rubric_path)
