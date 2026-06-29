import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def precompute_embeddings():
    """
    Reads cleansed candidate histories, generates dense vector embeddings, 
    and saves them as a binary NumPy array (.npy) alongside the local model.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    precomputed_dir = base_dir / "data" / "precomputed"
    input_file = precomputed_dir / "clean_candidate_histories.json"
    
    model_save_path = precomputed_dir / "local_models" / "all-MiniLM-L6-v2"
    embeddings_output_path = precomputed_dir / "history_embeddings.npy"
    ids_output_path = precomputed_dir / "candidate_ids.txt"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run extract_clean_histories.py first.")
        return

    print("Loading cleansed history data...")
    with open(input_file, "r", encoding="utf-8") as f:
        extracted_data = json.load(f)

    candidate_ids = []
    history_texts = []

    for cand_id, data in extracted_data.items():
        candidate_ids.append(cand_id)
        history_texts.append(data.get("history_text", ""))

    print("Downloading/Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    model_save_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_save_path))
    print(f"Model weights saved locally at: {model_save_path}")

    print(f"Generating embeddings for {len(history_texts)} candidates...")
    embeddings = model.encode(
        history_texts, 
        batch_size=256, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )

    print("Saving binary matrix and ID index...")
    np.save(embeddings_output_path, embeddings)
    
    with open(ids_output_path, "w", encoding="utf-8") as f:
        for cid in candidate_ids:
            f.write(f"{cid}\n")

    print(f"Embeddings matrix shape {embeddings.shape} saved to: {embeddings_output_path}")

if __name__ == "__main__":
    precompute_embeddings()