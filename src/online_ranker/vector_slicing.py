import numpy as np
import warnings
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

def run_vector_scoring(top_candidates: dict) -> dict:
    """
    Executes lightning-fast semantic similarity scoring on the shortlisted candidates.
    
    1. Loads the Job Description and embeds it using a locally bundled model.
    2. Slices into the pre-computed binary embeddings matrix (.npy) using memory-mapping.
    3. Computes the cosine similarity between the JD and the candidate's actual history.
    
    Args:
        top_candidates (dict): The shortlisted candidates from Stage 1.
                               Format: {cand_id: {"heuristic_score": float, "entities_found": list}}
                               
    Returns:
        dict: The updated candidate dictionary including "semantic_score".
    """
    print(f"Stage 2] Running Dense Vector Slicing for {len(top_candidates)} candidates...")

    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    
    jd_path = data_dir / "raw" / "job_description.md"
    precomputed_dir = data_dir / "precomputed"
    model_path = precomputed_dir / "local_models" / "all-MiniLM-L6-v2"
    embeddings_path = precomputed_dir / "project_embeddings.npy"
    ids_path = precomputed_dir / "candidate_ids.txt"

    id_to_index = {}
    with open(ids_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            cid = line.strip()
            id_to_index[cid] = idx

    shortlist_cids = list(top_candidates.keys())
    indices_to_slice = [id_to_index[cid] for cid in shortlist_cids if cid in id_to_index]

    print("   - Loading local SentenceTransformer model...")
    model = SentenceTransformer(str(model_path))
    
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print("   - Embedding Job Description...")
    jd_vector = model.encode(jd_text, convert_to_numpy=True)

    print("   - Memory-mapping dense embeddings matrix and slicing...")
    full_embeddings_matrix = np.load(embeddings_path, mmap_mode='r')
    
    sliced_vectors = full_embeddings_matrix[indices_to_slice]

    print("   - Computing vector alignment...")
    
    jd_norm = np.linalg.norm(jd_vector)
    jd_vector_normalized = jd_vector / jd_norm if jd_norm > 0 else jd_vector

    cand_norms = np.linalg.norm(sliced_vectors, axis=1, keepdims=True)
    sliced_vectors_normalized = sliced_vectors / np.where(cand_norms > 0, cand_norms, 1)

    similarities = np.dot(sliced_vectors_normalized, jd_vector_normalized)

    for i, cid in enumerate(shortlist_cids):
        top_candidates[cid]["semantic_score"] = float(similarities[i])

    print(f"[Stage 2] Semantic scoring complete for {len(top_candidates)} candidates.")
    return top_candidates

if __name__ == "__main__":
    dummy_input = {
        "CAND_0000031": {"heuristic_score": 5.0, "entities_found": ["pinecone", "xgboost"]},
        "CAND_0000001": {"heuristic_score": 1.0, "entities_found": []}
    }
    results = run_vector_scoring(dummy_input)
    print(results)