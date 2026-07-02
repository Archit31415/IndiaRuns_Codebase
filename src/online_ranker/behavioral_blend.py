import json
from pathlib import Path

def run_behavioral_blend(scored_candidates: dict, top_n: int = 100) -> list:
    """
    Blends the dynamic semantic/heuristic scores with the precomputed static 
    behavioral and experience scores.
    
    Applies cohort normalization (sequence-wise context) to scale the scores 
    relative to the shortlist pool before applying the final JD rubric weights.
    
    Args:
        scored_candidates (dict): Output from Stage 2 containing semantic & heuristic scores.
        top_n (int): The final number of candidates to output (default: 100).
        
    Returns:
        list: The finalized Top 100 candidates, sorted by rank.
    """
    print(f"[Stage 3] Blending Behavioral Signals & Computing Final Ranks...")

    base_dir = Path(__file__).resolve().parent.parent.parent
    precomputed_dir = base_dir / "data" / "precomputed"
    
    rubric_path = precomputed_dir / "recruiter_rubric.json"
    static_scores_path = precomputed_dir / "candidate_static_scores.json"

    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric = json.load(f)
        
    with open(static_scores_path, "r", encoding="utf-8") as f:
        static_scores_map = json.load(f)

    weights = rubric.get("stage4_final_ranking_weights", {})
    w_semantic = weights.get("retrieval_ranking_score", 0.25)
    w_heuristic = weights.get("production_system_score", 0.15)
    w_behavioral = weights.get("behavioral_score", 0.10)
    w_exp = weights.get("experience_score", 0.25)

    raw_semantics = [data["semantic_score"] for data in scored_candidates.values()]
    raw_heuristics = [data["heuristic_score"] for data in scored_candidates.values()]
    
    min_sem, max_sem = min(raw_semantics), max(raw_semantics)
    min_heu, max_heu = min(raw_heuristics), max(raw_heuristics)
    
    range_sem = (max_sem - min_sem) if max_sem > min_sem else 1.0
    range_heu = (max_heu - min_heu) if max_heu > min_heu else 1.0

    final_results = []

    for cid, dynamic_data in scored_candidates.items():
        norm_semantic = (dynamic_data["semantic_score"] - min_sem) / range_sem
        norm_heuristic = (dynamic_data["heuristic_score"] - min_heu) / range_heu
        
        static_data = static_scores_map.get(cid, {})
        behavioral_score = static_data.get("behavioral_score", 0.6) # Baseline default
        exp_modifier = static_data.get("experience_modifier", 1.0)
        
        base_exp_score = static_data.get("target_exp_score", 0.7)
        norm_exp = base_exp_score * exp_modifier

        final_score = (
            (norm_semantic * w_semantic) +
            (norm_heuristic * w_heuristic) +
            (behavioral_score * w_behavioral) +
            (norm_exp * w_exp)
        )
        
        boosted_score = 0.50 + (final_score * 0.50)
        
        final_results.append({
            "candidate_id": cid,
            "score": boosted_score,
            "years_exp": static_data.get("total_years_exp", 0.0),
            "entities_found": dynamic_data["entities_found"]
        })
    
    print("   - Sorting candidates and enforcing deterministic tie-breaks...")
    final_results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    top_n_results = final_results[:top_n]
    for rank_idx, candidate in enumerate(top_n_results):
        candidate["rank"] = rank_idx + 1 

    print(f"[Stage 3] Behavioral blend complete. Extracted final Top {len(top_n_results)}.")
    return top_n_results