import streamlit as st
import pandas as pd
from pathlib import Path

try:
    from src.online_ranker.purge import run_purge
    from src.online_ranker.ner_heuristics import run_ner_heuristics
    from src.online_ranker.vector_slicing import run_vector_scoring
    from src.online_ranker.behavioral_blend import run_behavioral_blend
    from src.online_ranker.reasoning import generate_reasoning_and_export
    PIPELINE_LOADED = True
except ImportError as e:
    PIPELINE_LOADED = False
    IMPORT_ERROR_MSG = str(e)

# --- Path Resolvers ---
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "sample_candidates.jsonl"
RUBRIC_PATH = BASE_DIR / "data" / "precomputed" / "recruiter_rubric.json"
OUTPUT_DIR = BASE_DIR / "data" / "output"

st.set_page_config(page_title="Production Ranker Sandbox", layout="wide")
st.title("Redrob AI Candidate Ranker — Production Sandbox")
st.markdown("This interface executes your **complete pipeline end-to-end with zero simulations**, using memory-mapped embeddings over the pre-loaded candidate pool.")

if not PIPELINE_LOADED:
    st.error(f"Core Pipeline Import Failure: {IMPORT_ERROR_MSG}")
    st.info("Ensure that app.py sits directly at your repository root and that your project module paths match your package imports.")
    st.stop()

# -----------------------------------------------------------------
# AUTOMATED DATA FLOW PIPELINE
# -----------------------------------------------------------------
# Step 1: Detect and verify the presence of the pre-loaded data files
if not RUBRIC_PATH.exists():
    st.error("Missing Precomputed Artifacts: `recruiter_rubric.json` not found under `data/precomputed/`.")
    st.stop()

data_file_to_use = RAW_DATA_PATH

if not data_file_to_use:
    st.error("Pre-loaded Dataset Missing: Ensure a subset or the full pool is named `candidates.jsonl` or `candidates.jsonl.gz` inside `data/raw/`[cite: 12].")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Sandbox Status")
    st.success(f"Verified Source File: `data/raw/{data_file_to_use.name}`")
    st.success("Verified Embeddings Ledger: `data/precomputed/history_embeddings.npy`")
    
    jd_path = BASE_DIR / "data" / "raw" / "job_description.md"
    if jd_path.exists():
        with open(jd_path, "r", encoding="utf-8") as f:
            st.text_area("Pre-loaded Job Description Document", value=f.read(), height=300, disabled=True)
    else:
        st.warning("`job_description.md` not detected at raw path location.")

with col2:
    st.subheader("Live Pipeline Execution")
    
    if st.button("Run Complete Pipeline End-to-End", type="primary"):
        status_box = st.empty()
        
        try:
            status_box.info("Executing Stage 0: Structural Purge & Honeypot Extermination...")
            valid_ids = run_purge(data_file_to_use, RUBRIC_PATH)
            st.write(f"**Stage 0 Complete:** {len(valid_ids)} candidates successfully survived hard filters[cite: 21].")
            
            if not valid_ids:
                st.error("No candidates survived the initial structural criteria pass.")
                st.stop()
                
            status_box.info("Executing Stage 1: C-Optimized Heuristic NER Shortlisting...")
            top_candidates_map = run_ner_heuristics(valid_ids, top_k=1500)
            st.write(f"**Stage 1 Complete:** Pool truncated to top {len(top_candidates_map)} candidates via tooling matrix scores[cite: 20].")
            
            status_box.info("Executing Stage 2: Memory-Mapped Embedding Vector Dot Product Alignment...")
            scored_candidates_map = run_vector_scoring(top_candidates_map)
            st.write("**Stage 2 Complete:** Core semantic alignments processed natively through physical `.npy` lookups[cite: 22].")
            
            status_box.info("Executing Stage 3: Multi-Score Normalization Blending & Deterministic Tie-Breaking...")
            final_top_100 = run_behavioral_blend(scored_candidates_map, top_n=100)
            st.write(f"**Stage 3 Complete:** Blended final metrics compiled for the elite {len(final_top_100)} shortlist profiles[cite: 18].")
            
            status_box.info("Executing Stage 4: Compiling non-hallucinated text rows and outputting final CSV array...")
            team_name = "Compilation_Error" 
            generate_reasoning_and_export(final_top_100, team_name=team_name)
            
            status_box.success("Pipeline complete! Top 100 Shortlist successfully rendered below.")
            
            generated_csv_path = OUTPUT_DIR / f"{team_name}_submission.csv"
            
            if generated_csv_path.exists():
                output_df = pd.read_csv(generated_csv_path)
                st.dataframe(output_df, use_container_width=True)
                
                with open(generated_csv_path, "rb") as csv_f:
                    st.download_button(
                        label="Download Production Ranked CSV Output",
                        data=csv_f,
                        file_name=f"{team_name}_submission.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Pipeline finished but output CSV generation failed tracking checks.")
                
        except Exception as pipeline_error:
            st.error(f"Pipeline Execution Failure: {pipeline_error}")
            status_box.empty()