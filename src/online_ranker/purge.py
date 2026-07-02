import json
import gzip
from pathlib import Path
from datetime import datetime

def parse_date(date_str):
    """Safely converts ISO date strings into datetime objects for chronological verification."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def run_purge(candidates_file_path: Path, rubric_path: Path) -> set:
    print("[Stage 0] Executing Hardened Structural Purge & Adversarial Honeypot Verification...")
    
    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric = json.load(f)
        
    hard_filters = rubric.get("stage0_hard_filters", {})
    min_exp = hard_filters.get("min_years_experience", 4.0)
    blacklisted_firms = {firm.lower().strip() for firm in hard_filters.get("blacklisted_companies_if_exclusive", [])}
    incompatible_domains = {dom.lower().strip() for dom in hard_filters.get("incompatible_domains", [])}

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

            for skill in skills:
                proficiency = str(skill.get("proficiency", "")).lower().strip()
                duration = skill.get("duration_months", 0)
                
                if proficiency in ["advanced", "expert"] and duration <= 0:
                    is_honeypot = True
                    break

            total_claimed_months = 0
            earliest_start = None
            latest_end = None
            
            for job in career:
                start_dt = parse_date(job.get("start_date"))
                end_dt = parse_date(job.get("end_date")) if not job.get("is_current") else datetime.now()
                
                job_duration_months = job.get("duration_months", 0)
                total_claimed_months += job_duration_months
                
                if start_dt:
                    if earliest_start is None or start_dt < earliest_start:
                        earliest_start = start_dt
                    if end_dt:
                        if latest_end is None or end_dt > latest_end:
                            latest_end = end_dt
                            
                if start_dt and end_dt and job_duration_months > 0:
                    real_job_delta_months = ((end_dt - start_dt).days / 30.44) + 1.5
                    if job_duration_months > real_job_delta_months * 1.5:
                        is_honeypot = True

            if earliest_start and latest_end:
                max_possible_calendar_months = ((latest_end - earliest_start).days / 30.44) + 1.0
                
                if total_claimed_months > (max_possible_calendar_months * 1.3):
                    is_honeypot = True
                    
                if (claimed_years_exp * 12.0) > (max_possible_calendar_months * 1.1):
                    is_honeypot = True

            if is_honeypot:
                honeypots_caught += 1
                continue

            if claimed_years_exp < min_exp:
                filter_drops += 1
                continue
            
            companies_worked = [str(job.get("company", "")).lower().strip() for job in career]
            if companies_worked and all(any(bl_firm in comp for bl_firm in blacklisted_firms) for comp in companies_worked):
                filter_drops += 1
                continue
                
            job_titles = [str(job.get("title", "")).lower().strip() for job in career]
            if job_titles and all(any(bad_dom in title for bad_dom in incompatible_domains) for title in job_titles):
                filter_drops += 1
                continue

            valid_candidate_ids.add(cid)

    print(f"   - Total Processed: {total_processed}")
    print(f"   - Honeypots Exterminated: {honeypots_caught}")
    print(f"   - JD Hard-Filter Drops: {filter_drops}")
    print(f"[Stage 0] Clean Candidates Surviving: {len(valid_candidate_ids)}")
    
    return valid_candidate_ids