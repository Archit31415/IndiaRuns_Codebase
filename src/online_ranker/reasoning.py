import csv
from pathlib import Path

def generate_reasoning_and_export(final_candidates: list, team_name: str = "team_redrob"):
    """
    Generates deterministic, hallucination-free reasoning for the Top 100 candidates
    and writes the final submission CSV file required by the hackathon evaluators.
    
    Args:
        final_candidates (list): The sorted list of dictionaries from Stage 3.
        team_name (str): Used to format the output CSV filename.
    """
    print(f"[Stage 4] Generating deterministic reasoning and exporting CSV...")

    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_dir / f"{team_name}_submission.csv"

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for cand in final_candidates:
            cid = cand["candidate_id"]
            rank = cand["rank"]
            score = round(cand["score"], 4)
            
            entities = cand.get("entities_found", [])
            years_exp = cand.get("years_exp", 0.0)
            behavioral_score = cand.get("behavioral_score", 0.0)

            # Template A: Rich Entity Match
            if len(entities) >= 2:
                formatted_entities = ", ".join([e.title() for e in entities[:3]]) 
                reasoning = (
                    f"Profile strongly aligns with core requirements across {years_exp} years of verified experience. "
                    f"Candidate demonstrates explicit hands-on expertise with key target technologies ({formatted_entities})."
                )
            
            # Template B: High Semantic/Behavioral, Lighter on Specific Entities
            elif behavioral_score > 0.7:
                reasoning = (
                    f"Candidate possesses {years_exp} years of relevant product engineering history with high semantic relevance to the JD. "
                    f"Furthermore, strong platform engagement signals indicate high availability and responsiveness."
                )
            
            # Template C: Fallback for generic high semantic matches
            else:
                reasoning = (
                    f"Selected based on a strong semantic vector alignment between the job description and the candidate's {years_exp} years of chronological work history."
                )

            writer.writerow([cid, rank, score, reasoning])

    print(f"[Stage 4] Successfully wrote {len(final_candidates)} candidates to {output_csv}")
    print("Ranking Pipeline Complete.")

if __name__ == "__main__":
    # Local mock execution
    dummy_results = [
        {
            "candidate_id": "CAND_0000031",
            "rank": 1,
            "score": 0.9432,
            "entities_found": ["pinecone", "xgboost", "codeforces"],
            "years_exp": 6.0,
            "behavioral_score": 0.85
        },
        {
            "candidate_id": "CAND_0000015",
            "rank": 2,
            "score": 0.7811,
            "entities_found": ["python"],
            "years_exp": 4.5,
            "behavioral_score": 0.91
        }
    ]
    generate_reasoning_and_export(dummy_results, "demo_team")