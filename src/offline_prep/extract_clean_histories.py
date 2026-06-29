import json
import gzip
from pathlib import Path

def extract_clean_histories():
    """
    Parses candidates.jsonl.gz to extract validated career histories.
    
    This script intentionally ignores 'profile.summary' and the static 'skills' 
    array to bypass keyword-stuffing traps. It outputs a pre-processed JSON 
    containing the clean text for embeddings and the structural metadata 
    needed for the Stage 0/1 heuristic filters.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_data_path = base_dir / "data" / "raw" / "candidates.jsonl"
    output_dir = base_dir / "data" / "precomputed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "clean_candidate_histories.json"

    extracted_data = {}
    
    print(f"Reading raw candidate data from: {raw_data_path}")
    
    try:
        with open(raw_data_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                cand_data = json.loads(line)
                cand_id = cand_data.get("candidate_id")
                if not cand_id:
                    continue
                
                history_text_blocks = []
                industries_worked = set()
                companies_worked = set()
                
                for role in cand_data.get("career_history", []):
                    title = role.get("title", "")
                    description = role.get("description", "")
                    industry = role.get("industry", "")
                    company = role.get("company", "")
                    
                    if title or description:
                        block = f"Title: {title}. Description: {description}."
                        history_text_blocks.append(block.strip())
                    
                    if industry:
                        industries_worked.add(industry)
                    if company:
                        companies_worked.add(company.lower())

                full_history_text = " ".join(history_text_blocks)
                
                extracted_data[cand_id] = {
                    "history_text": full_history_text,
                    "industries": list(industries_worked),
                    "companies": list(companies_worked)
                }

    except FileNotFoundError:
        print(f"Error: Could not find raw data at {raw_data_path}")
        print("Please ensure 'candidates.jsonl.gz' is placed in 'data/raw/'")
        return

    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(extracted_data, out_f, indent=4)
        
    print(f"Successfully extracted and cleaned histories for {len(extracted_data)} candidates.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    print("Starting Phase 1: Cross-Field Verification Extraction...")
    extract_clean_histories()