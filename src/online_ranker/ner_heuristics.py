import json
import spacy
from pathlib import Path

def run_ner_heuristics(valid_candidate_ids: set, top_k: int = 1500) -> dict:
    """
    Runs ultra-fast spaCy NER over the purged candidate pool to extract exact 
    matches for infrastructure, ML tools, and ranking frameworks. 
    
    Cross-references these extracted entities against the rubric weights and 
    the pre-computed static experience multipliers to isolate the top 1,500 
    candidates for the final dense vector similarity check.

    Args:
        valid_candidate_ids (set): IDs that survived Stage 0.
        top_k (int): Number of candidates to pass to Stage 2.

    Returns:
        dict: A mapping of the top_k candidate IDs to their extracted NER entities 
              and heuristic scores (useful for Stage 4 Reasoning).
    """
    print(f"[Stage 1] Running NER Heuristics on {len(valid_candidate_ids)} candidates...")

    base_dir = Path(__file__).resolve().parent.parent.parent
    precomputed_dir = base_dir / "data" / "precomputed"
    
    rubric_path = precomputed_dir / "recruiter_rubric.json"
    histories_path = precomputed_dir / "clean_candidate_histories.json"
    static_scores_path = precomputed_dir / "candidate_static_scores.json"
    spacy_model_path = precomputed_dir / "local_models" / "spacy_ner_pipeline"

    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric = json.load(f)
        
    with open(histories_path, "r", encoding="utf-8") as f:
        histories = json.load(f)
        
    with open(static_scores_path, "r", encoding="utf-8") as f:
        static_scores = json.load(f)

    target_weights = {}
    for category, config in rubric.get("stage1_text_targets", {}).items():
        target_weights[category.upper()] = config.get("weight", 1.0)

    print("   - Loading local spaCy EntityRuler...")
    nlp = spacy.load(spacy_model_path)

    # We only process candidates that survived the Stage 0 Purge
    processing_batch = []
    for cid in valid_candidate_ids:
        text = histories.get(cid, {}).get("history_text", "")
        if text:
            processing_batch.append((text, cid))

    candidate_scores = []
    extracted_entities_map = {}

    print("   - Executing bulk entity extraction...")
    # nlp.pipe is heavily optimized in C and bypasses Python loop overhead
    for doc, cid in nlp.pipe(processing_batch, as_tuples=True, batch_size=512):
        score = 0.0
        found_ents = set()
        
        for ent in doc.ents:
            label = ent.label_
            text_val = ent.text.lower()
            
            if text_val not in found_ents:
                found_ents.add(text_val)
                score += target_weights.get(label, 1.0)
                
        # Store what we found so Stage 4 can use it to write the reasoning sentences
        extracted_entities_map[cid] = list(found_ents)
        
        # Scale the pure NER score by their static experience modifier 
        # (This penalizes job hoppers and boosts product-company engineers immediately)
        exp_modifier = static_scores.get(cid, {}).get("experience_modifier", 1.0)
        final_heuristic_score = score * exp_modifier
        
        candidate_scores.append((final_heuristic_score, cid))

    # 5. Sort and truncate to the top 1,500
    print("   - Sorting and shortlisting top candidates...")
    # Sort descending by score
    candidate_scores.sort(key=lambda x: (-x[0], x[1]))
    
    top_candidates = {}
    for score, cid in candidate_scores[:top_k]:
        top_candidates[cid] = {
            "heuristic_score": score,
            "entities_found": extracted_entities_map[cid]
        }

    print(f"✅ [Stage 1] Shortlist complete. Filtered down to top {len(top_candidates)} candidates.")
    return top_candidates

if __name__ == "__main__":
    # Local manual testing execution
    # Assuming valid_ids comes from stage0
    dummy_valid_ids = {"CAND_0000031", "CAND_0000001"} # Ela Singh vs Ira Vora test
    results = run_ner_heuristics(dummy_valid_ids, top_k=2)
    print(json.dumps(results, indent=2))